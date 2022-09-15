[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)


# WoNeCa
To reproduce the simulations and results of the presentation at WoNeCa 2022

# Start

Create a Python 3.9 virtual environment using `virtualenv`

        $ python -m virtualenv --python=python3.9 ./venv
        $ source venv/bin/activate

Install dependencies

        $ pip install -Ur requirements.txt

# Usage

Comparison schemes figure 1:

        W=20khz, rho=25, snr=5db, 3.16 linear, N=10
        W=20khz, rho=25, snr=5db, 3.16 linear, N=5
        W=20khz, rho=25, snr=5db, 3.16 linear, N=1

Comparison schemes figure 2:

        W=20khz, rho=25, snr=5db, 3.16 linear, N=10
        W=20khz, rho=25, snr=5db, 3.16 linear, N=5

        W=20khz, rho=25, snr=4db, 2 linear, N=10
        W=20khz, rho=25, snr=4db, 2 linear, N=5

        W=20khz, rho=20, snr=4db, 2 linear, N=10
        W=20khz, rho=20, snr=4db, 2 linear, N=5

        W=20khz, rho=20, snr=3db, 2 linear, N=10
        W=20khz, rho=20, snr=3db, 2 linear, N=5

# Training specs: ideal

N=5
        "type": "gmevm",
        "centers": 4,
        "hidden_sizes": (30, 50, 30),
        "dataset_size": 'all'
        "batch_size": 1024 * 512,  # 1024, 1024*128
        "rounds": [
                {"learning_rate": 1e-2, "epochs": 40}
                {"learning_rate": 1e-3, "epochs": 10}
        ],
N=10
        "type": "gmevm",
        "centers": 4,
        "hidden_sizes": (30, 50, 30),
        "dataset_size": 'all'
        "batch_size": 1024 * 512,  # 1024, 1024*128
        "rounds": [
                {"learning_rate": 1e-2, "epochs": 40}
                {"learning_rate": 1e-3, "epochs": 10}
        ],


# Contributing

Use code checkers

        $ pre-commit autoupdate
        $ pre-commit install
        $ pre-commit run --all-files


