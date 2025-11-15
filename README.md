# Scalable Product Duplicate Detection using LSH

This repository contains the source code for "Scalable Product Duplicate Detection using LSH". The method filters out
dissimilar products using minhashing and LSH, after which only the candidate pairs are considered in Agglomerative
Clustering with single linkage.

## Getting Started

Clone this repository to your local machine, and open a terminal in this directory. This project uses `uv` as a package manager, [install uv](https://docs.astral.sh/uv/getting-started/installation/) or create your own virtual environment with the dependencies listed in `pyproject.toml`.

To create a virtual environment for this project using `uv`, run:

```
uv venv
```

Then, install dependencies listed in `pyproject.toml` using:

```
uv sync
```

Before running this script, download the [dataset](https://personal.eur.nl/frasincar/datasets/TVs-all-merged.zip) and open the zip-file into `data/TVs-all-merged.json`.

The `main.py` script is the main entry point of the program. Follow these instructions to run the script for various
models:

- MSMP-J: set the `fast` argument of the `apply_clustering` function to `False`
- MSMP-Lite: set the `fast` argument of the `apply_clustering` function to `True`

In order to use the same hash-function as the orginal MSMP method, pass `''` for the `separator` argument to the `lsh`
function.
