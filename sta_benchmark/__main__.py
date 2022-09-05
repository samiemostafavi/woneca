import multiprocessing as mp
import os
import time
from pathlib import Path

import numpy as np
import polars as pl
from loguru import logger
from qsimpy.core import Model, Sink
from qsimpy.discrete import CapacityQueue, CapacitySource, Deterministic, Rayleigh
from qsimpy.polar import PolarSink
from qsimpy.simplequeue import SimpleQueue

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def create_run_graph(params):
    # Create the QSimPy environment
    # a class for keeping all of the entities and accessing their attributes
    model = Model(name=f"Gym Rayleigh benchmark #{params['run_number']}")

    # arrival process uniform
    arrival = Deterministic(
        seed=0,
        rate=11,
        initial_load=0,
        duration=None,
        dtype="float64",
    )
    # Create a source
    source = CapacitySource(
        name="start-node",
        arrival_rp=arrival,
        task_type="0",
    )
    model.add_entity(source)

    for queue_num in range(params["num_queues"]):
        queue_name = f"queue_{str(queue_num)}"

        if queue_num == 0:
            source.out = queue_name
        else:
            # old queue connect
            queue.out = queue_name

        # service process is Rayleigh channel capacity
        service = Rayleigh(
            seed=120034,
            snr=0,  # in db
            bandwidth=17e3,  # in hz
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

    last_queue = queue

    # create the sinks
    sink = PolarSink(
        name="gym-sink",
        batch_size=10000,
    )

    def user_fn(df):
        # df is pandas dataframe in batch_size
        df["end2end_delay"] = df["end_time"] - df["start_time"]
        return df

    sink._post_process_fn = user_fn
    model.add_entity(sink)

    # make the rest of the connections
    last_queue.out = sink.name
    last_queue.drop = sink.name

    # Setup task records
    model.set_task_records(
        {
            "timestamps": {
                source.name: {
                    "task_generation": "start_time",
                },
                last_queue.name: {
                    "service_time": "end_time",
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
        "{0}: Source generated {1} tasks".format(
            params["run_number"], source.get_attribute("tasks_generated")
        )
    )
    logger.info(
        "{0}: Queue completed {1}, dropped {2}".format(
            params["run_number"],
            queue.get_attribute("tasks_completed"),
            queue.get_attribute("tasks_dropped"),
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

    # print(df)

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
    project_path = str(p) + "/projects/sta_train/"

    # simulation parameters
    # bench_params = {str(n): n for n in range(15)}
    bench_params = {
        #'1':1,
        #'5':5,
        "10": 10,
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

            until = int(100000)
            params = {
                "records_path": records_path,
                "arrivals_number": int(until / 10),
                "run_number": j * parallel_runs + i,
                "service_seed": 120034 + i * 200202 + j * 20111,
                "num_queues": bench_params[key_this_run],  # number of queues
                "until": until,  # 10M timesteps takes 1000 seconds, generates 900k samples
                "report_state": 0.05,  # report when 10%, 20%, etc progress reaches
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
