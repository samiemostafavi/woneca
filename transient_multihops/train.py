import numpy as np
import tensorflow as tf

tf.config.set_visible_devices([], "GPU")

import json
import os
import warnings
from os.path import abspath, dirname
from pathlib import Path

from loguru import logger
from petastorm import TransformSpec
from petastorm.spark import SparkDatasetConverter, make_spark_converter
from pr3d.de import (
    ConditionalGammaEVM,
    ConditionalGammaMixtureEVM,
    ConditionalGaussianMM,
)
from pyspark.sql import SparkSession
from pyspark.sql.functions import rand
from tensorflow import keras

warnings.filterwarnings("ignore")


def init_spark():

    spark = (
        SparkSession.builder.appName("Training")
        .config("spark.driver.memory", "70g")
        .config("spark.driver.maxResultSize", 0)
        .getOrCreate()
    )

    sc = spark.sparkContext
    return spark, sc


# init Spark
spark, sc = init_spark()

# Set a cache directory on DBFS FUSE for intermediate data.
file_path = dirname(abspath(__file__))
spark.conf.set(
    SparkDatasetConverter.PARENT_CACHE_DIR_URL_CONF,
    "file://" + file_path + "/sparkcache",
)
logger.info(f"Spark cache folder is set up at: {'file://' + file_path + '/sparkcache'}")

npdtype = np.float64
tfdtype = tf.float64
strdtype = "float64"

# open the dataset
# project folder setting
p = Path(__file__).parents[0]
project_folder = str(p) + "/projects/transient/"
project_paths = [
    project_folder + name
    for name in os.listdir(project_folder)
    if os.path.isdir(os.path.join(project_folder, name))
]

# limit
project_paths = [
    #'/home/wlab/woneca/transient_multihops/projects/transient/500_results',
    '/home/wlab/woneca/transient_multihops/projects/transient/1000_results',
    '/home/wlab/woneca/transient_multihops/projects/transient/1500_results',
]

logger.info(f"Opening projects paths: {project_paths}")
records_paths = [s + "/records" for s in project_paths]

# high sample training params

#training_params = {
#    'dataset_size': 60*1024,  # 60*1024, 60*1024*1024, 'all'
#    'batch_size': 1024,  # 1024, 1024*128
#    'rounds' : [
#        {'learning_rate': 1e-2, 'epochs':40}, # GMEVM: 15, GMM:
#        {'learning_rate': 1e-3, 'epochs':30}, # GMEVM: 10, GMM:
#    ],
#}

training_params = {
    'dataset_size': 'all',  # 60*1024, 60*1024*1024, 'all'
    'batch_size': 1024,  # 1024, 1024*128
    'rounds' : [
        {'learning_rate': 1e-2, 'epochs':30}, # GMEVM: 15, GMM:
        {'learning_rate': 1e-3, 'epochs':15}, # GMEVM: 15, GMM:
    ],
}

"""
# low sample training params
training_params = {
    "dataset_size": 'all' ,  # 60*1024, 60*1024*1024, 'all'
    "batch_size": 1024 * 512,  # 1024, 1024*128
    "rounds": [
        {"learning_rate": 1e-2, "epochs": 50},  # GMEVM: 15, GMM:
        #{"learning_rate": 1e-3, "epochs": 20},  # GMEVM: 15, GMM:
        #{"learning_rate": 1e-4, "epochs": 10},  # GMEVM: 10, GMM:
    ],
}
"""

#condition_labels = ["snr", "rho"]
condition_labels = ["queue_length"]
y_label = "end2end_delay"

models_conf = [
    #{
    #   'type': 'gmm',
    #   'bayesian': False,
    #   'ensembles': 1,
    #   'centers': 5,
    #   'hidden_sizes': (50, 80, 50),
    #   'condition_labels' : condition_labels,
    #   'y_label' : y_label,
    #   'training_params': training_params,
    #},
    {
        "type": "gmevm",
        "bayesian": False,
        "ensembles": 1,
        "centers": 4,
        "hidden_sizes": (30, 50, 30),
        "condition_labels": condition_labels,
        "y_label": y_label,
        "training_params": training_params,
    },
]

add_gaussian_noise = True
gaussian_scale = 0.25

# read all the files from the project

files = []
for idx, records_path in enumerate(records_paths):
    logger.info(f"Opening the path '{records_path}'")

    all_files = os.listdir(records_path)
    for f in all_files:
        if f.endswith(".parquet"):
            files.append(records_path + "/" + f)


predictors_path = project_folder + "/predictors/"
logger.info(f"Opening predictors directory '{predictors_path}'")
os.makedirs(predictors_path, exist_ok=True)

# read all files into Spark df
logger.info(f"Reading files: {files}")
main_df = spark.read.parquet(*files)

# Absolutely necessary for randomizing the rows (bug fix)
# first shuffle, then sample!
main_df = main_df.orderBy(rand())

for model_conf in models_conf:

    training_params = model_conf["training_params"]

    if training_params["dataset_size"] == "all":
        df_train = main_df.sample(
            withReplacement=False,
            fraction=1.00,
        )
    else:
        # take the desired number of records for learning
        df_train = main_df.sample(
            withReplacement=False,
            fraction=training_params["dataset_size"] / main_df.count(),
        )

    # dataset partitioning and making the converters
    # Make sure the number of partitions is at least the number of workers which is required for distributed training.
    df_train = df_train.repartition(1)
    converter_train = make_spark_converter(df_train)
    logger.info(f"Dataset loaded, train sampels: {len(converter_train)}")

    def transform_row(pd_batch):
        """
        The input and output of this function are pandas dataframes.
        """

        pd_batch = pd_batch[[y_label, *condition_labels]]
        pd_batch["y_input"] = pd_batch[y_label]
        pd_batch = pd_batch.drop(columns=[y_label])

        if add_gaussian_noise:
            pd_batch["y_input"] = pd_batch["y_input"] + np.random.normal(
                loc=np.zeros(len(pd_batch)),
                scale=np.ones(len(pd_batch)) * gaussian_scale,
            )
            # replace possible negative values by zero
            pd_batch[pd_batch["y_input"] < 0] = 0

        #print(pd_batch)
        # if input normalization
        pd_batch['queue_length'] = pd_batch['queue_length']/1500.0

        return pd_batch

    # Note that the output shape of the `TransformSpec` is not automatically known by petastorm,
    # so we need to specify the shape for new columns in `edit_fields` and specify the order of
    # the output columns in `selected_fields`.
    x_fields = [(cond, npdtype, (), False) for cond in condition_labels]
    transform_spec_fn = TransformSpec(
        transform_row,
        edit_fields=[
            *x_fields,
            ("y_input", npdtype, (), False),
        ],
        selected_fields=[*condition_labels, "y_input"],
    )

    model_type = model_conf["type"]
    ensembles = model_conf["ensembles"]
    condition_labels = model_conf["condition_labels"]
    training_rounds = training_params["rounds"]
    batch_size = training_params["batch_size"]
    for num_ensemble in range(ensembles):

        # initiate the non conditional predictor
        if model_type == "gmm":
            model = ConditionalGaussianMM(
                x_dim=condition_labels,
                centers=model_conf["centers"],
                hidden_sizes=model_conf["hidden_sizes"],
                dtype=strdtype,
                bayesian=model_conf["bayesian"],
                # batch_size = 1024,
            )
        elif model_type == "gevm":
            model = ConditionalGammaEVM(
                x_dim=condition_labels,
                hidden_sizes=model_conf["hidden_sizes"],
                dtype=strdtype,
                bayesian=model_conf["bayesian"],
                # batch_size = 1024,
            )
        elif model_type == "gmevm":
            model = ConditionalGammaMixtureEVM(
                x_dim=condition_labels,
                centers=model_conf["centers"],
                hidden_sizes=model_conf["hidden_sizes"],
                dtype=strdtype,
                bayesian=model_conf["bayesian"],
                # batch_size = 1024,
            )

        with converter_train.make_tf_dataset(
            transform_spec=transform_spec_fn,
            batch_size=batch_size,
        ) as train_dataset:

            # tf.keras only accept tuples, not namedtuples
            # map the dataset to the desired tf.keras input in _pl_training_model
            def map_fn(x):
                x_dict = {}
                for idx, cond in enumerate(condition_labels):
                    x_dict = {**x_dict, cond: x[idx]}
                return ({**x_dict, "y_input": x.y_input}, x.y_input)

            train_dataset = train_dataset.map(map_fn)

            steps_per_epoch = len(converter_train) // batch_size

            for idx, params in enumerate(training_rounds):
                logger.info(
                    f"Starting training session {idx}/{len(training_rounds)} with {params}"
                )

                model._pl_training_model.compile(
                    optimizer=keras.optimizers.Adam(
                        learning_rate=params["learning_rate"]
                    ),
                    loss=model.loss,
                )

                logger.info(f"steps_per_epoch: {steps_per_epoch}")

                model._pl_training_model.fit(
                    train_dataset,
                    steps_per_epoch=steps_per_epoch,
                    epochs=params["epochs"],
                    verbose=1,
                )

        model.save(predictors_path + f"ls_{model_type}_model_{num_ensemble}.h5")
        with open(
            predictors_path + f"ls_{model_type}_model_{num_ensemble}.json", "w"
        ) as write_file:
            json.dump(model_conf, write_file, indent=4)

        logger.info(
            f"A {model_type} {'bayesian' if model.bayesian else 'non-bayesian'} "
            + f"model got trained and saved, ensemble: {num_ensemble}."
        )
