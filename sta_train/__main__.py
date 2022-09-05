import itertools
import multiprocessing as mp
import os
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import polars as pl
from loguru import logger
from qsimpy.core import Model, Sink
from qsimpy.discrete import CapacityQueue, CapacitySource, Deterministic, Rayleigh
from qsimpy.polar import PolarSink
from qsimpy.simplequeue import SimpleQueue


def create_run_graph(params):
    # Create the QSimPy environment
    # a class for keeping all of the entities and accessing their attributes
    model = Model(name=f"Gym Rayleigh benchmark #{params['run_number']}")

    # arrival process uniform
    arrival = Deterministic(
        seed=0,
        rate=params["rho"],
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

    queue = None
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
            snr=params["snr"],  # in db
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
        df["snr"] = params["snr"]
        df["rho"] = params["rho"]
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
    print(df)
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
    project_path = str(p) + "/projects/sta_train_huge/"

    # simulation parameters
    # bench_params = {str(n): n for n in range(15)}
    hops_options = {
        "1": 1,
        "5": 5,
        "10": 10,
    }

    snr_params = OrderedDict()
    snr_params["snr_0"] = 0
    snr_params["snr_1p5"] = 1.5
    snr_params["snr_3"] = 3

    arrival_rate_params = OrderedDict()
    arrival_rate_params["rho_7"] = 7
    arrival_rate_params["rho_9"] = 9
    arrival_rate_params["rho_11"] = 11

    index_list = [[idx for idx in range(3)] for _ in range(2)]
    combinations_idx = list(itertools.product(*index_list))
    bench_params = {}
    for comb in combinations_idx:
        name_str = f"{list(snr_params.items())[comb[0]][0]}_"
        name_str += f"{list(arrival_rate_params.items())[comb[1]][0]}"
        bench_params = {
            **bench_params,
            name_str: {
                "snr": list(snr_params.items())[comb[0]][1],
                "rho": list(arrival_rate_params.items())[comb[1]][1],
            },
        }

    sequential_runs = 3  # 5
    parallel_runs = 18  # 18
    for j in range(sequential_runs):

        hops_keys = list(hops_options.keys())
        hops_this_run_key = hops_keys[j % len(hops_keys)]

        processes = []
        for i in range(parallel_runs):  # range(parallel_runs):

            # parameter figure out
            params_keys = list(bench_params.keys())
            params_this_run_key = params_keys[i % len(params_keys)]

            # create and prepare the results directory
            results_path = project_path + hops_this_run_key + "_results/"
            records_path = results_path + "records/"
            os.makedirs(records_path, exist_ok=True)

            until = int(1000000)
            params = {
                "records_path": records_path,
                "arrivals_number": int(until / 10),
                "run_number": j * parallel_runs + i,
                "service_seed": 120034 + i * 200202 + j * 20111,
                "num_queues": hops_options[hops_this_run_key],  # number of queues
                "until": until,  # 10M timesteps takes 1000 seconds, generates 900k samples
                "report_state": 0.05,  # report when 10%, 20%, etc progress reaches
                **bench_params[params_this_run_key],
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
