import argparse
import json
import math
import random
from dataclasses import dataclass
from typing import Dict, List, Set

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering
from tqdm import tqdm

from constants import BRANDS
from encoder import ModelWordsEncoder, ShingleEncoder
from lsh import lsh
from minhashing import minhash


@dataclass
class Product:
    title: str
    modelID: str
    shop: str
    url: str
    featuresMap: Dict[str, str]

    def get_shop(self):
        return self.shop


def diff_brand(a: Product, b: Product):
    a_str = a.title + " " + " ".join(a.featuresMap.values()).lower()
    b_str = b.title + " " + " ".join(b.featuresMap.values()).lower()

    for brand in BRANDS:
        a_has_brand = brand in a_str
        b_has_brand = brand in b_str

        if a_has_brand != b_has_brand:
            return True

    return False


def intersect(set1: Set, set2: Set) -> Set:
    smallest = set1 if len(set1) <= len(set2) else set2
    largest = set1 if len(set1) > len(set2) else set2

    return {i for i in smallest if i in largest}


def jaccard(value1: Set[str], value2: Set[str]):
    if len(value1) == 0 and len(value2) == 0:
        return 0

    n_intersect = len(intersect(value1, value2))
    return n_intersect / (len(value1) + len(value2) - n_intersect)


def cosine_similarity(value1: Set[str], value2: Set[str]):
    if len(value1) == 0 and len(value2) == 0:
        return 0

    n_intersect = len(intersect(value1, value2))
    return n_intersect / (math.sqrt(len(value1)) + math.sqrt(len(value2)))


def extract_model_words(attributes: Dict[str, str]) -> Set[str]:
    result: Set[str] = set()

    for value in attributes.values():
        for mw in ModelWordsEncoder.get_model_words(value):
            result.add(mw)

    return result


def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)


def apply_clustering(
    products: List[Product],
    candidate_pairs: NDArray,
    k: int,
    fast: bool,
    mu=0.650,
    gamma=0.775,
    distance_threshold=0.9,
):
    N = len(products)
    inf_distance = 1000
    distances = np.ones((N, N)) * inf_distance
    np.fill_diagonal(distances, 0)

    def shingle_product(p: Product):
        shingles = ShingleEncoder.shingle(p.title, k)

        for key, value in p.featuresMap.items():
            shingles.update(ShingleEncoder.shingle(key, k))
            shingles.update(ShingleEncoder.shingle(value, k))

        return shingles

    product_shingles = [shingle_product(p) for p in products] if fast else None

    def set_distance(i: int, j: int, value):
        distances[i, j] = value
        distances[j, i] = value

    for i, j in tqdm(candidate_pairs, desc="Clustering", leave=False):
        if i == j:
            continue

        product_i: Product = products[i]
        product_j: Product = products[j]

        if product_i.shop == product_j.shop or diff_brand(product_i, product_j):
            set_distance(i, j, inf_distance)
            continue
        elif fast:
            shingles_i = product_shingles[i]
            shingles_j = product_shingles[j]

            set_distance(i, j, 1 - jaccard(shingles_i, shingles_j))
            continue

        sim = 0
        m = 0
        w = 0

        nonmatching_i: Dict[str, str] = dict(product_i.featuresMap)
        nonmatching_j: Dict[str, str] = dict(product_j.featuresMap)

        for q_key, q_value in product_i.featuresMap.items():
            for r_key, r_value in product_j.featuresMap.items():
                key_similarity = jaccard(
                    ShingleEncoder.shingle(q_key, k), ShingleEncoder.shingle(r_key, k)
                )

                if key_similarity > gamma:
                    value_similarity = jaccard(
                        ShingleEncoder.shingle(q_value, k),
                        ShingleEncoder.shingle(r_value, k),
                    )
                    weight = key_similarity
                    sim += weight * value_similarity
                    m += 1
                    w += weight
                    nonmatching_i.pop(q_key)
                    nonmatching_j.pop(r_key)

        avg_sim = sim / w if w > 0 else 0
        mw_perc = jaccard(
            extract_model_words(nonmatching_i), extract_model_words(nonmatching_j)
        )
        title_sim = jaccard(
            ShingleEncoder.shingle(product_i.title.lower(), k),
            ShingleEncoder.shingle(product_j.title.lower(), k),
        )

        min_features = min(len(product_i.featuresMap), len(product_j.featuresMap))
        h_sim: float
        if title_sim < 0.5:
            theta1 = m / min_features
            theta2 = 1 - theta1
            h_sim = theta1 * avg_sim + theta2 * mw_perc
        else:
            theta1 = (1 - mu) * m / min_features
            theta2 = 1 - mu - theta1
            h_sim = theta1 * avg_sim + theta2 * mw_perc + mu * title_sim

        set_distance(i, j, 1 - h_sim)

    # perform clustering
    model = AgglomerativeClustering(
        distance_threshold=distance_threshold,
        n_clusters=None,
        linkage="single",
        metric="precomputed",
    )
    model.fit(distances)

    # sys.setrecursionlimit(10000)
    # plt.title("Hierarchical Clustering Dendrogram")
    # # plot the top three levels of the dendrogram
    # plot_dendrogram(model)
    # plt.xlabel("Duplicate products")
    # plt.show()

    comparisons_made = np.zeros((N, N))

    for i, j in np.argwhere((distances > 0) & (distances < inf_distance)):
        comparisons_made[min(i, j), max(i, j)] = 1

    num_comparisons_made = comparisons_made.sum()

    return model, num_comparisons_made


def performance_metrics(
    prefix: str,
    predicted_duplicates: NDArray,
    actual_duplicates: NDArray,
    num_comparisons_made: int,
):
    duplicates_found = np.sum(predicted_duplicates * actual_duplicates) / 2
    total_num_duplicates = np.sum(actual_duplicates) / 2

    pair_quality = duplicates_found / num_comparisons_made
    pair_completeness = duplicates_found / total_num_duplicates
    f1 = (2 * pair_quality * pair_completeness) / (pair_quality + pair_completeness)

    return {
        f"{prefix}f1": f1,
        f"{prefix}PQ": pair_quality,
        f"{prefix}PC": pair_completeness,
        f"{prefix}D_f": duplicates_found,
        f"{prefix}N_c": num_comparisons_made,
        f"{prefix}D_n": total_num_duplicates,
    }


def preprocess(products: List[Product]):
    mw_encoder = ModelWordsEncoder()
    mw = [mw_encoder.encode(product.title.lower()) for product in products]

    return mw_encoder, mw


def load_data():
    jsondata: Dict[str, List[Dict]]
    with open("data/TVs-all-merged.json") as f:
        jsondata = json.load(f)

    products: List[Product] = []

    for duplicates in jsondata.values():
        for product in duplicates:
            products.append(
                Product(
                    title=product["title"].lower(),
                    modelID=product["modelID"],
                    shop=product["shop"],
                    url=product["url"],
                    featuresMap=product["featuresMap"],
                )
            )

    N = len(products)
    duplicates_matrix = np.zeros((N, N)).astype(int)

    for i, p1 in enumerate(products):
        for j, p2 in enumerate(products):
            if i != j and p1.modelID == p2.modelID:
                duplicates_matrix[i, j] = 1

    print(f"\nN={N} (of which {len(jsondata)} unique)\n")

    return products, duplicates_matrix


def main(fast: bool):
    products, duplicates_matrix = load_data()
    N = len(products)

    replications = 10
    n = 1000
    k = 3

    index_range = range(len(products))
    experiment_results: List[Dict] = []

    for bootstrap in tqdm(range(replications), desc="Replications"):
        current_indices = random.sample(index_range, k=N)
        current_products = [products[i] for i in current_indices]
        current_duplicates = np.array(
            [
                [duplicates_matrix[i, j] for j in current_indices]
                for i in current_indices
            ]
        )
        current_num_duplicates = np.sum(current_duplicates) / 2

        mw_encoder, mw = preprocess(current_products)

        for r in tqdm(
            [r for r in range(2, n) if n % r == 0],
            desc="(r,b) combinations",
            leave=False,
        ):
            b = round(n / r)

            # create signature matrix
            m = minhash(mw, n=r * b, vocab_size=mw_encoder.vocabulary_size())

            if np.isinf(m).sum() > 0:
                raise ValueError(f"M still contains infinite values")

            # apply LSH
            candidate_pairs = lsh(m, b, r)
            lsh_predicted_duplicates = np.zeros_like(current_duplicates)

            for i, j in candidate_pairs:
                lsh_predicted_duplicates[i, j] = 1
                lsh_predicted_duplicates[j, i] = 1

            # perform clustering
            model, num_comparisons_made = apply_clustering(
                current_products, candidate_pairs, k=k, fast=fast
            )
            predicted_duplicates = np.array(
                [
                    [int(model.labels_[i] == model.labels_[j]) for j in range(N)]
                    for i in range(N)
                ]
            )

            experiment_results.append(
                {
                    "bootstrap": bootstrap,
                    "n": n,
                    "b": b,
                    "r": r,
                    "num_duplicates": current_num_duplicates,
                    **performance_metrics(
                        "lsh__",
                        lsh_predicted_duplicates,
                        current_duplicates,
                        candidate_pairs.shape[0],
                    ),
                    **performance_metrics(
                        "clu__",
                        predicted_duplicates,
                        current_duplicates,
                        num_comparisons_made,
                    ),
                }
            )

    df_results = pd.DataFrame(experiment_results)
    print(df_results)
    df_results.to_csv("results_all.csv")

    def get_best_for_bootstrap(i: int, key: str):
        filtered_df = df_results[df_results["bootstrap"] == i]
        f1 = filtered_df[key]
        return filtered_df[f1 == f1.max()]

    best_f1_star = pd.concat(
        [get_best_for_bootstrap(i, "lsh__f1") for i in range(replications)]
    )
    print("\n\nBest F1-star results per bootstrap:\n")
    print(
        best_f1_star[
            [
                "bootstrap",
                "n",
                "b",
                "r",
                "lsh__f1",
                "lsh__PQ",
                "lsh__PC",
                "lsh__D_f",
                "lsh__N_c",
                "lsh__D_n",
            ]
        ]
    )
    best_f1_star.to_csv("results_best_f1_star.csv")

    best_f1 = pd.concat(
        [get_best_for_bootstrap(i, "clu__f1") for i in range(replications)]
    )
    print("\n\nBest F1 results per bootstrap:\n")
    print(
        best_f1[
            [
                "bootstrap",
                "n",
                "b",
                "r",
                "clu__f1",
                "clu__PQ",
                "clu__PC",
                "clu__D_f",
                "clu__N_c",
                "clu__D_n",
            ]
        ]
    )
    best_f1.to_csv("results_best_f1.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--fast",
        action=argparse.BooleanOptionalAction,
        type=bool,
        default=True,
        help="When this is true, MSMP-Lite will be used. Otherwise, MSMP-J is used. (default: True)",
    )

    args = parser.parse_args()
    fast: bool = parser.fast

    main(fast)
