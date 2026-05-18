import os
import time
import hypernetx as hnx
import pandas as pd
from sklearn.metrics import rand_score
import random

from spectral import spectral_hypergraph_clustering
from qubo import qubo_clustering

ALL_RESULTS = []


def construct_hypergraph(data, id_column):
    hyperedge_dict = {}
    features = [col for col in data.columns if col != id_column]
    for col in features:
        unique_values = data[col].unique()
        for val in unique_values:
            subset = data[data[col] == val]
            nodes = subset[id_column].tolist()
            if nodes:
                hyperedge_dict[f"{col}_{val}"] = nodes
    return hnx.Hypergraph(hyperedge_dict)


def run(dataset_name, dataset_path, class_col, num_clusters, test_type, id_col=None, num_samples=-1, rnd=7, extra_info=""):
    df = pd.read_csv(dataset_path)

    if num_samples == -1 or num_samples > len(df):
        num_samples = len(df)
    else:
        df = df.sample(num_samples, random_state=rnd).reset_index(drop=True)

    if id_col is None:
        df["id"] = range(len(df))
        id_col = "id"

    names = sorted(df[id_col].values)
    ground_truth = df.set_index(id_col)[class_col].to_dict()
    labels_y = [ground_truth[n] for n in names]

    h = construct_hypergraph(df.drop(columns=[class_col]), id_col)

    s_time = time.time()
    spectral_res = spectral_hypergraph_clustering(h, num_clusters)
    e_time = time.time()
    labels_s = [spectral_res.get(n, -1) for n in names]
    sim_s = rand_score(labels_y, labels_s)
    time_s = e_time - s_time

    s_time = time.time()
    qubo_res = qubo_clustering(h, num_clusters)
    e_time = time.time()
    labels_q = [qubo_res.get(n, -1) for n in names]
    sim_q = rand_score(labels_y, labels_q)
    time_q = e_time - s_time

    info_str = f" | {extra_info}" if extra_info else ""
    print(f"[{test_type}] {dataset_name} (n={num_samples}, seed={rnd}){info_str}")
    print(
        f"  > Spectral: {sim_s:.4f} ({round(time_s, 3)}s) | QUBO: {sim_q:.4f} ({round(time_q, 3)}s)")

    ALL_RESULTS.append({
        "test_type": test_type,
        "dataset": dataset_name,
        "num_samples": num_samples,
        "num_clusters": num_clusters,
        "seed": rnd,
        "sim_spectral": sim_s,
        "time_spectral": time_s,
        "sim_qubo": sim_q,
        "time_qubo": time_q,
        "extra_info": extra_info
    })

    return sim_s, sim_q


def run_standard_benchmarks():
    seeds = [3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
    print("\n" + "="*25 + " 1. STANDARD BENCHMARKS " + "="*25)

    for s in seeds:
        random.seed(s)
        run("zoo", "./dataset/zoo.csv", "type", 7,
            "Standard", id_col="name", num_samples=101, rnd=s)
    print("-"*50)

    for n in [100, 250, 500]:
        for s in seeds:
            random.seed(s)
            run("mushroom", "./dataset/mushroom.csv",
                "cls", 2, "Standard", num_samples=n, rnd=s)
        print("-"*50)

    for n in [100, 250, 500]:
        for s in seeds:
            random.seed(s)
            run("letter", "./dataset/letter.csv", "lettr",
                26, "Standard", num_samples=n, rnd=s)
        print("-"*50)

    for n in [100, 250, 434]:
        for s in seeds:
            random.seed(s)
            run("votes", "./dataset/house-votes-84.csv", "class",
                2, "Standard", num_samples=n, rnd=s)
        print("-"*50)

    for n in [100, 250, 500]:
        for s in seeds:
            random.seed(s)
            run("nursey", "./dataset/nursery.csv", "class",
                5, "Standard", num_samples=n, rnd=s)
        print("-"*50)

    for n in [100, 250, -1]:
        for s in seeds:
            random.seed(s)
            run("soybean", "./dataset/soybean-large.csv", "class",
                19, "Standard", num_samples=n, rnd=s)
        print("-"*50)



def run_custom_subsets(target_subsets):
    dataset_path = "./dataset/zoo.csv"
    temp_path = "./dataset/zoo_temp_custom.csv"
    df_full = pd.read_csv(dataset_path)
    seeds = [3, 5, 7, 11, 13, 17, 19, 23, 29, 31]

    print("\n" + "="*25 + " 2. CUSTOM ZOO SUBSETS " + "="*25)
    for kept_classes in target_subsets:
        df_filtered = df_full[df_full["type"].isin(
            kept_classes)].reset_index(drop=True)
        if df_filtered.empty:
            continue
        df_filtered.to_csv(temp_path, index=False)

        for s in seeds:
            run("zoo_custom", temp_path, "type", len(kept_classes), "CustomSubset",
                id_col="name", rnd=s, extra_info=f"Classes: {kept_classes}")
        print("-"*50)

    if os.path.exists(temp_path):
        os.remove(temp_path)


def get_ablation_steps(df, class_col):
    class_counts = df[class_col].value_counts().sort_values(ascending=True)
    sorted_classes = class_counts.index.tolist()
    steps = []
    current_classes = sorted_classes.copy()
    while len(current_classes) >= 2:
        steps.append({
            "num_classes": len(current_classes),
            "classes_to_keep": current_classes.copy()
        })
        current_classes.pop(0)
    return steps


def run_systematic_ablation():
    dataset_path = "./dataset/zoo.csv"
    temp_path = "./dataset/zoo_temp_ablation.csv"
    df_full = pd.read_csv(dataset_path)
    ablation_steps = get_ablation_steps(df_full, "type")
    seeds = [3, 5, 7, 11, 13, 17, 19, 23, 29, 31]

    print("\n" + "="*25 + " 3. SYSTEMATIC ABLATION " + "="*25)
    for step in ablation_steps:
        df_filtered = df_full[df_full["type"].isin(
            step["classes_to_keep"])].reset_index(drop=True)
        df_filtered.to_csv(temp_path, index=False)

        for s in seeds:
            run("zoo_ablation", temp_path, "type", step["num_classes"], "Ablation",
                id_col="name", rnd=s, extra_info=f"K={step['num_classes']}")
        print("-"*50)

    if os.path.exists(temp_path):
        os.remove(temp_path)


if __name__ == "__main__":
    run_standard_benchmarks()
    # run_custom_subsets([[4, 7, 6], [4, 7, 6, 3, 5], [
    #                    2, 4, 7, 6, 3, 5], [2, 4, 7, 6]])
    # run_systematic_ablation()

    df_final = pd.DataFrame(ALL_RESULTS)
    df_final.to_csv("all_datasets_results.csv", index=False)

    group_cols = ["test_type", "dataset", "num_samples"]
    agg_cols = ["sim_spectral", "sim_qubo", "time_spectral", "time_qubo"]
    df = df_final.groupby(group_cols)[agg_cols].agg(["mean", "var"]).reset_index()
    df.columns = [f"{col}_{stat}" if stat else col for col, stat in df.columns]
    df = df.round(4)
    df.to_csv("aggregated.csv")
