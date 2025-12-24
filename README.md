# Scalable Product Duplicate Detection using LSH

This repository contains the source code for "Scalable Product Duplicate Detection using LSH". The method filters out
dissimilar products using minhashing and LSH, after which only the candidate pairs are considered in Agglomerative
Clustering with single linkage.

## Getting Started

Clone this repository to your local machine, and open a terminal in this directory. This project uses `uv` as a package manager, [install uv](https://docs.astral.sh/uv/getting-started/installation/) or create your own virtual environment with the dependencies listed in `pyproject.toml`.

To install the dependencies in a virtual environment, run the following command:

```
uv sync
```

Before running this script, download the [dataset](https://personal.eur.nl/frasincar/datasets/TVs-all-merged.zip) and open the zip-file into `data/TVs-all-merged.json`.

The `src/pdd/main.py` script is the main entry point of the program, run `python src/pdd/main.py --help` for a list of options. Follow these instructions to run the script for various
models:

- MSMP-J: `python src/pdd/main.py --no-fast`
- MSMP-Lite: `python src/pdd/main.py --fast`

In order to use the same hash-function as the orginal MSMP method, pass `''` for the `separator` argument to the `lsh`
function.
