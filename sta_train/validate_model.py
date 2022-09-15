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
from pr3d.de.cond_gamma_mevm import ConditionalGammaMixtureEVM

conditions = {
    "hops_num": 5,
    "snr": 5,
    "rho": 25,
}

fig, ax = plt.subplots()


p = Path(__file__).parents[0]
project_path = str(p) + "/projects/sta_train_finallast/"
results_path = project_path + f"{conditions['hops_num']}_results/"
records_path = results_path + "records/"
predictor_path = results_path + "predictors/" + "ls_gmevm_model_0.h5"

pr_model = (
    ConditionalGammaMixtureEVM(  # ConditionalGaussianMM, ConditionalGammaMixtureEVM
        h5_addr=predictor_path,
    )
)
logger.info(f"Predictor: {Path(predictor_path).stem} is loaded.")


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

# apply the condition:
df = df.filter(
    (pl.col("snr") == conditions["snr"]) & (pl.col("rho") == conditions["rho"])
)
logger.info(f"Number of conditioned samples: {len(df)}")

delays = range(60)
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

# calculate the prediction quantile values

x = [[conditions["snr"], conditions["rho"]] for _ in range(len(delays))]
x = np.array(x)
y = np.array(delays, dtype=np.float64)
y = y.clip(min=0.00)
prob, logprob, pred_cdf = pr_model.prob_batch(x, y)


ax.semilogy(
    delays,
    1.00 - np.array(pred_cdf),
    marker=".",
    label="prediction",
)


ax.set_xlim(0, 60)
ax.set_ylim(1e-6, 1e0)
ax.grid()
ax.legend()

fig.tight_layout()
fig.savefig(results_path + "sta_model_validation_tail2.png")
