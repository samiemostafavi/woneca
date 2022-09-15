import json
import math
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns
from loguru import logger

max_x = 40
conditions = {
    "hops_num": 5,
    "snr": 3, 
    "rho": 20,
}

fig, ax = plt.subplots()


p = Path(__file__).parents[0]
project_path = str(p) + "/sta_benchmark/projects/sta_benchmark_finalplus/"
results_path = project_path + f"{conditions['hops_num']}_results/"
records_path = results_path + "records/"
plot_wtb = False
"""
json_path = results_path + f"{hops_num}_wtb_results.json"

# JSON file
with open(json_path, "r") as f:
    # Reading from file
    wtb_data = json.loads(f.read())

wtb_delays = []
wtb_probs = []
for row in wtb_data['results']:
    wtb_delays.append(row[0])
    wtb_probs.append(row[1])
"""

df = pl.read_parquet(records_path + "*.parquet")
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


'''
drop_prc = 0.5
s_num = int(len(df)*drop_prc)
logger.info(f"Dropping first {drop_prc} fraction which is {s_num} samples.")
pd_df = df.to_pandas()
pd_df = pd_df.iloc[s_num:]
df = pl.DataFrame(pd_df)
logger.info(f"New df len {len(df)}")
'''

# apply the condition:
df = df.filter(
    (pl.col("snr") == conditions["snr"]) & (pl.col("rho") == conditions["rho"])
)
logger.info(f"Number of conditioned samples: {len(df)}")

delays = range(max_x)
res = []
for x in delays:
    # print(f"x: {x}")
    df_cond = df.filter(pl.col("end2end_delay") > x)

    res.append(len(df_cond) / len(df))
    # r = (df.select(pl.col('noisy_end2end_delay').quantile(quantile,'midpoint')))
    # res.append(r[0,0])

ax.semilogy(
    delays,  # res,
    res,
    marker=".",
    label=f'simulation - {conditions["hops_num"]} hops',
)

"""
ax.semilogy(
    wtb_delays,  #res,
    wtb_probs,
    marker='.',
    label=f'WTB - {conditions["hops_num"]} hops',
)
"""

if plot_wtb:

    logger.info("Transient bound data openning from json")
    json_path = results_path + f"{conditions['hops_num']}_wtb_results.json"
    # JSON file
    with open(json_path, "r") as f:
        # Reading from file
        wtb_data = json.loads(f.read())

    wtb_delays = []
    wtb_probs = []
    for row in wtb_data["results"]:
        wtb_delays.append(row[0])
        wtb_probs.append(row[1])

    ax.semilogy(
        wtb_delays,  # res,
        wtb_probs,
        linestyle='dotted',
        marker=".",
        label="sim",
    )

ax.set_xlim(0, max_x)
ax.set_ylim(1e-6, 1e0)
ax.grid()
ax.legend()

fig.tight_layout()
fig.savefig(results_path + "validation_test.png")

fig, ax = plt.subplots()
arr = df.get_column("end2end_delay").to_numpy()
