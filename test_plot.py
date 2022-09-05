import json
import math
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns
from loguru import logger

max_x = 60
hops_nums = [10]
fig, ax = plt.subplots()
project_path = "sta_benchmark/projects/sta_train/"
plot_wtb = False

for hops_num in hops_nums:

    results_path = project_path + f"{hops_num}_results/"
    records_path = results_path + "records/"

    json_path = results_path + f"{hops_num}_wtb_results.json"

    if plot_wtb:
        # JSON file
        with open(json_path, "r") as f:
            # Reading from file
            wtb_data = json.loads(f.read())

        wtb_delays = []
        wtb_probs = []
        for row in wtb_data["results"]:
            wtb_delays.append(row[0])
            wtb_probs.append(row[1])

    df = pl.read_parquet(records_path + "/*.parquet")
    logger.info(f"Project path {records_path} parquet files are loaded.")

    """
    # generate white noise
    noise_mu = 0
    noise_sigma = 0.25
    noise = np.random.normal(noise_mu, noise_sigma, len(df))
    df = df.with_column(pl.Series(name="gaussian_noise", values=noise)) 
    df = df.with_column(
        pl.sum(
            [
                pl.col('end2end_delay'),
                pl.col('gaussian_noise'),
            ]
        ).alias('noisy_end2end_delay')
    )
    """

    logger.info(f"Total number of samples in this empirical dataset: {len(df)}")
    delays = range(max_x)
    res = []
    for x in delays:
        # print(f"x: {x}")
        df_cond = df.filter(pl.col("end2end_delay") > x)
        # print(len(df_cond))
        res.append(len(df_cond) / len(df))
        # r = (df.select(pl.col('noisy_end2end_delay').quantile(quantile,'midpoint')))
        # res.append(r[0,0])

    ax.semilogy(
        delays,  # res,
        res,
        marker=".",
        label=f"simulation - {hops_num} hops",
    )
    if plot_wtb:
        ax.semilogy(
            wtb_delays,  # res,
            wtb_probs,
            marker=".",
            label=f"WTB - {hops_num} hops",
        )

ax.set_xlim(0, max_x)
ax.set_ylim(1e-6, 1e0)
ax.grid()
ax.legend()

fig.tight_layout()
fig.savefig(project_path + "sta_validation_tail.png")
