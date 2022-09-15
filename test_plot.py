import json
import math
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns
from loguru import logger

max_x = 60
hops_nums = [5,10]
project_path = "sta_train/projects/prefinal/"
x_label = "End-to-end delay (ms)"
y_label = "Violation probability"
plot_wtb = True

plt.rcParams["font.family"] = "Times New Roman"
font = {
    'size'   : 17
}
matplotlib.rc('font', **font)
fig, ax = plt.subplots(figsize=(8,5))

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

    drop_prc = 0.5
    s_num = int(len(df)*drop_prc)
    logger.info(f"Dropping first {drop_prc} fraction which is {s_num} samples.")
    pd_df = df.to_pandas()
    pd_df = pd_df.iloc[s_num:]
    df = pl.DataFrame(pd_df)
    logger.info(f"New df len {len(df)}")

    logger.info(f"Total number of samples in this empirical dataset: {len(df)}")
    delays = range(max_x)
    res = []
    #old = len(df)
    for x in delays:
        # print(f"x: {x}")
        df_cond = df.filter(pl.col("end2end_delay") > x)
        # print(len(df_cond))
        #if len(df_cond) > 20:
        #    diff = len(df_cond)/old
        res.append(len(df_cond) / len(df))
        #    old = len(df_cond)
        #else:
        #    res.append((old*diff) / len(df))
        #    old = old*diff
        # r = (df.select(pl.col('noisy_end2end_delay').quantile(quantile,'midpoint')))
        # res.append(r[0,0])
        

    sim_line, = ax.semilogy(
        delays,  # res,
        res,
        linestyle='solid',
        linewidth=3,
        marker=".",
        markersize=12,
        label=f"Simulation, {hops_num} hops",
    )
    if plot_wtb:
        ax.semilogy(
            wtb_delays,  # res,
            wtb_probs,
            linestyle='dotted',
            linewidth=3,
            color = sim_line.get_color(),
            marker=".",
            markersize=12,
            label=f"Bound, {hops_num} hops",
        )

ax.set_xlabel(x_label)
ax.set_ylabel(y_label)
ax.set_xlim(0, max_x)
ax.set_ylim(1e-5, 1e0)
ax.legend()
ax.grid()


fig.tight_layout()
fig.savefig(project_path + "sta_validation_tail_new.png")
