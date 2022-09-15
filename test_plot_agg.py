import json
import math
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns
from pathlib import Path
from loguru import logger
from pr3d.de.cond_gamma_mevm import ConditionalGammaMixtureEVM


hops_nums = [5,10]
max_xs = [60, 60]

params = [
    (5,25),
    #(4,25),
    #(4,20),
    (3,20),
]

project_path = "sta_train/projects/prefinal_train_lowsample/"
x_label = "End-to-end delay (ms)"
y_label = "Violation probability"
plot_wtb = False
plot_dl = True

plt.rcParams["font.family"] = "Times New Roman"
font = {
    'size'   : 17
}
matplotlib.rc('font', **font)

for idx, hops_num in enumerate(hops_nums):

    fig, ax = plt.subplots(figsize=(8,5))
    max_x = max_xs[idx]

    results_path = project_path + f"{hops_num}_results/"
    records_path = results_path + "records/"
    predictor_path = results_path + "predictors/"

    # load parquet for empirical
    df = pl.read_parquet(records_path + "*.parquet")
    logger.info(f"Project path {records_path} parquet files are loaded.")
    logger.info(f"Total number of samples in this empirical dataset: {len(df)}")

    if plot_dl:
        # ConditionalGaussianMM, ConditionalGammaMixtureEVM
        pr_model = ConditionalGammaMixtureEVM(  
            h5_addr=predictor_path+"ls_gmevm_model_0.h5",
        )
        logger.info(f"Predictor: {Path(predictor_path).stem} is loaded.")

    if plot_wtb:
        logger.info("Transient bound data openning from json")
        json_path = results_path + f"{hops_num}_wtb_results.json"
        # JSON file
        with open(json_path, "r") as f:
            # Reading from file
            wtb_datas = json.loads(f.read())

    delays = range(max_x)
    for snr, rho in params:
        logger.info(f"Parameters SNR:{snr} and Rho:{rho}")

        # apply the condition:
        df_params = df.filter(
            (pl.col("snr") == snr) & (pl.col("rho") == rho)
        )
        logger.info(f"Number of conditioned samples: {len(df_params)}")
        
        res = []
        for x in delays:
            df_cond = df_params.filter(pl.col("end2end_delay") > x)
            res.append(len(df_cond) / len(df_params))

        logger.info(f"Plotting simulation curve")
        sim_line, = ax.semilogy(
            delays,  # res,
            res,
            linestyle='solid',
            linewidth=2,
            marker=".",
            markersize=10,
            label=r'Simulation $\bar{\gamma}=%s, \rho=%s$' %(hops_num,snr,rho),
        )

        if plot_dl:
            # calculate the prediction quantile values
            x = [[snr, rho] for _ in range(len(delays))]
            x = np.array(x)
            y = np.array(delays, dtype=np.float64)
            y = y.clip(min=0.00)
            prob, logprob, pred_cdf = pr_model.prob_batch(x, y)

            logger.info(f"Plotting data-driven curve")
            ax.semilogy(
                delays,  # res,
                1.00 - np.array(pred_cdf),
                linestyle='dotted',
                linewidth=2,
                color = sim_line.get_color(),
                marker="^",
                markersize=8,
                label=r'Data-driven $\bar{\gamma}=%s, \rho=%s$' %(snr,rho),
            )

        if plot_wtb:
            # find wtb dict
            for tmp in wtb_datas:
                if tmp['snr'] == snr and tmp['arrival']['rho'] == rho:
                    wtb_data = tmp

            wtb_delays = []
            wtb_probs = []
            for row in wtb_data["results"]:
                wtb_delays.append(row[0])
                wtb_probs.append(row[1])

            ax.semilogy(
                wtb_delays,  # res,
                wtb_probs,
                linestyle='dotted',
                linewidth=2,
                color = sim_line.get_color(),
                marker=".",
                markersize=10,
                label=r'Bound $\bar{\gamma}=%s, \rho=%s$' %(hops_num,snr,rho),
            )


    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_xlim(0, max_x)
    ax.set_ylim(1e-5, 1e0)
    ax.legend(prop={'size':10})
    ax.grid()


    fig.tight_layout()
    fig.savefig(results_path + "sta_benchmark_agg_new2.png")
