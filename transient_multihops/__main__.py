import multiprocessing as mp
import os
import time
from pathlib import Path

import numpy as np
import polars as pl
import pandas as pd
from pydantic import PrivateAttr
from loguru import logger
from qsimpy.core import Model, Sink, Entity, Task
from qsimpy.discrete import CapacityQueue, Rayleigh
from qsimpy.gym import GymSink, GymSource
from qsimpy.polar import pandas_to_polars

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# create the sinks
class MultiHopGymSink(GymSink):
    type: str = "multihopgymsink"
    _traffic_outs = PrivateAttr()

    def run(self):
        while True:
            task = yield self._store.get()
            is_last_main = task.is_last_main

            # EVENT task_reception
            task = self.add_records(task=task, event_name="task_reception")

            self.attributes["tasks_received"] += 1
            if self._debug:
                print(task)

            self._received_tasks.append(task)

            if len(self._received_tasks) >= self.batch_size:
                pddf = pd.DataFrame(self._received_tasks)
                # save the received task into a Polars dataframe
                if self._pl_received_tasks is None:
                    self._pl_received_tasks = pandas_to_polars(
                        pddf, self._post_process_fn
                    )
                else:
                    self._pl_received_tasks = self._pl_received_tasks.vstack(
                        pandas_to_polars(pddf, self._post_process_fn)
                    )
                del task, pddf
                self._received_tasks = []

            if (self.out is not None) and (is_last_main):
                # first notify the traffics
                for traffic_out in self._traffic_outs:
                    traffic_out.put(Task(id=0, task_type="start_msg"))
                # send the start message to the source
                self._out.put(Task(id=0, task_type="start_msg"))

def create_run_graph(params):
    # Create the QSimPy environment
    # a class for keeping all of the entities and accessing their attributes
    model = Model(name=f"Gym Rayleigh benchmark #{params['run_number']}")

    # first lets have sink name
    sink_name = "gym-sink"
    start_source_name = "start-node"
    start_source = None
    drop_sink_name = "drop-sink"

    queue_old_name = None
    last_queue = None
    queue_names = []
    traffic_sources = []
    for queue_num in reversed(range(params["num_queues"])):

        # create the traffic sources
        if queue_num == 0:
            main_task_num = 1
            source_name = start_source_name
        else:
            main_task_num = 0
            source_name = f"source_{str(queue_num)}"
        
        source = GymSource(
            name=source_name,
            main_task_num=main_task_num,
            main_task_type="main",
            traffic_task_type="traffic",
            traffic_task_num=int(params["traffic_tasks"]/params["num_queues"]),
        )
        model.add_entity(source)
        if queue_num != 0:
            traffic_sources.append(source)
        else:
            start_source = source

        # create the queue
        queue_name = f"queue_{str(queue_num)}"
        queue_names.append(queue_name)
        # service process is Rayleigh channel capacity
        service = Rayleigh(
            seed=params["service_seed"]+queue_num*1002,
            snr=5,  # in db
            bandwidth=25e3,  # in hz
            time_slot_duration=1e-3,  # in seconds
            dtype="float64",
        )
        # a queue
        queue = CapacityQueue(
            name=f"queue_{queue_num}",
            service_rp=service,
            queue_limit=None,
        )
        model.add_entity(queue)

        source.out = queue_name
        queue.drop = drop_sink_name
        if queue_num == params["num_queues"]-1:
            last_queue = queue
            last_queue_name = queue.name
            queue.out = sink_name
        else:
            queue.out = queue_old_name

        queue_old_name = queue.name


    # create the sinks
    sink = MultiHopGymSink(
        name=sink_name,
        batch_size=10000,
    )

    def user_fn(df):
        # df is pandas dataframe in batch_size
        df["end2end_delay"] = df["end_time"] - df["start_time"]
        df["queue_length"] = params["traffic_tasks"]
        return df

    sink._post_process_fn = user_fn
    model.add_entity(sink)

    drop_sink = Sink(
        name=drop_sink_name,
    )
    model.add_entity(drop_sink)

    # make the last connection
    sink.out = start_source_name
    sink._traffic_outs = traffic_sources

    # Setup task records
    model.set_task_records(
        {
            "timestamps": {
                start_source_name: {
                    "task_generation": "start_time",
                },
                last_queue_name: {
                    "service_time": "end_time",
                },
            },
            "attributes": {
                start_source_name: {
                    "task_generation": {
                        name: {
                            "queue_length": name + "_length",
                        } for name in queue_names
                    },
                },
            },
        }
    )

    modeljson = model.json()
    with open(
        params["records_path"] + f"{params['run_number']}_model.json",
        "w",
        encoding="utf-8",
    ) as f:
        f.write(modeljson)

    # prepare for run
    model.prepare_for_run(debug=False)

    # report timesteps
    def report_state(time_step):
        yield model.env.timeout(time_step)
        logger.info(
            f"{params['run_number']}:"
            + " Simulation progress"
            + f" {100.0*float(model.env.now)/float(params['until'])}% done"
        )

    for step in np.arange(
        0, params["until"], params["until"] * params["report_state"], dtype=int
    ):
        model.env.process(report_state(step))

    # Run!
    start = time.time()
    model.env.run(until=params["until"])
    end = time.time()
    logger.info(f"{params['run_number']}: Run finished in {end - start} seconds")

    logger.info(
        "{0}: Source generated {1} main tasks".format(
            params["run_number"], source.get_attribute("main_tasks_generated")
        )
    )
    logger.info(
        "{0}: Last queue completed {1}, dropped {2}".format(
            params["run_number"],
            last_queue.get_attribute("tasks_completed"),
            last_queue.get_attribute("tasks_dropped"),
        )
    )
    logger.info(
        "{0}: Sink received {1} main tasks".format(
            params["run_number"], sink.get_attribute("tasks_received")
        )
    )

    start = time.time()

    # Process the collected data
    df = sink.received_tasks
    # df_dropped = df.filter(pl.col("end_time") == -1)
    df_finished = df.filter(pl.col("end_time") >= 0)
    df = df_finished

    #print(df)

    end = time.time()

    df.write_parquet(
        file=params["records_path"] + f"{params['run_number']}_records.parquet",
        compression="snappy",
    )

    logger.info(
        "{0}: Data processing finished in {1} seconds".format(
            params["run_number"], end - start
        )
    )


if __name__ == "__main__":

    # project folder setting
    p = Path(__file__).parents[0]
    project_path = str(p) + "/projects/transient/"

    # simulation parameters
    # bench_params = {str(n): n for n in range(15)}
    bench_params = {
        #"20": [20,6000],
        #"40": [40,3800],
        "500": [500,200],
        #"1000": [1000,100],
        #"1500": [1500,100],
    }

    sequential_runs = 1  # 5
    parallel_runs = 18  # 18
    for j in range(sequential_runs):

        processes = []
        for i in range(parallel_runs):  # range(parallel_runs):

            # parameter figure out
            keys = list(bench_params.keys())
            # remember to modify this line
            key_this_run = keys[i % len(keys)]

            # create and prepare the results directory
            results_path = project_path + key_this_run + "_results/"
            records_path = results_path + "records/"
            os.makedirs(records_path, exist_ok=True)

            #until = int(800 * (bench_params[key_this_run] + 1))
            until = int(bench_params[key_this_run][1] * (bench_params[key_this_run][0] + 1))
            params = {
                "records_path": records_path,
                "arrivals_number": int(until / 10),
                "run_number": j * parallel_runs + i,
                "service_seed": 120034 + i * 200202 + j * 20111,
                "traffic_tasks": bench_params[key_this_run][0],  # number of traffic tasks
                "until": until,  # 10M timesteps takes 1000 seconds, generates 900k samples
                "report_state": 0.05,  # report when 10%, 20%, etc progress reaches
                "num_queues": 10,
            }

            p = mp.Process(target=create_run_graph, args=(params,))
            p.start()
            processes.append(p)

        try:
            for p in processes:
                p.join()
        except KeyboardInterrupt:
            for p in processes:
                p.terminate()
                p.join()
                exit(0)
