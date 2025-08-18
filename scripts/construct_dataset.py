""" Construt benchmark dataset. """

import os
import argparse

import pandas as pd
import numpy as np

from tqdm import tqdm
from sklearn.model_selection import train_test_split

from src.data.chemistry import standardize_smiles
from src.data.fingerprints import calculate_fingerprints
from src.data.similarity import calculate_tanimoto_similarities
from src.data.properties import calculate_logp
from src.data.utils import get_smiles_dtype

DATASET_SIZE = 3000
TEST_SIZE = 0.2
CLASSIFICATION_PERCENTILE_THRESHOLD = 90
N_RANDOM_STATES = 5
OOD_THRESHOLD = 0.4


def _load(data_filepath):
    df = pd.read_csv(data_filepath)[["smiles"]]
    df["smiles_standardized"] = df.apply(lambda x: standardize_smiles(x["smiles"]), axis=1)
    df["logp"] = df.apply(lambda x: calculate_logp(x["smiles_standardized"]), axis=1)
    df = df[~(df["smiles_standardized"].isna() | df["logp"].isna())]
    df = df.rename(columns={"smiles_standardized": "sequence", "logp": "target"}, inplace=False)
    return df[["sequence", "target"]]


def _sample(df, num_samples, random_state):
    if num_samples < len(df):
        return df.sample(num_samples, random_state=random_state)
    else:
        return df


def _parse_args():

    args = argparse.ArgumentParser(description="Construct synthetic OOD dataset.")
    args.add_argument("--out_dir", type=str, required=True, help="Output directory for the dataset.")
    args.add_argument("--data_filepath", type=str, required=True, help="Path to the ZINC250K data file.")
    args.add_argument("--data_filepath_synthetic", type=str, required=True, help="Path to the synthetic data file.")
    return args


def main(out_dir: str, data_filepath: str, data_filepath_synthetic: str):

    os.makedirs(out_dir, exist_ok=True)

    _df = _load(data_filepath)
    _df_synthetic = _load(data_filepath_synthetic)

    target_min, target_max = _df_synthetic["target"].min(), _df_synthetic["target"].max()
    _df = _df[(_df["target"] >= target_min) & (_df["target"] <= target_max)]  # control for data imbalance
    assert len(_df) > DATASET_SIZE, "Dataset size is too small"

    for random_state in tqdm(range(N_RANDOM_STATES), desc="Constructing dataset"):

        _out_dir = os.path.join(out_dir, f"seed_{random_state}")
        os.makedirs(_out_dir, exist_ok=True)

        df = _sample(_df, num_samples=DATASET_SIZE, random_state=random_state)
        df_synthetic = _sample(_df_synthetic, num_samples=int(DATASET_SIZE * TEST_SIZE), random_state=random_state)

        classification_parcentile_threshold = np.percentile(df_synthetic["target"], CLASSIFICATION_PERCENTILE_THRESHOLD)
        df["target_classification"] = df["target"] >= classification_parcentile_threshold
        df_synthetic["target_classification"] = df_synthetic["target"] >= classification_parcentile_threshold

        df_train, df_test = train_test_split(df, test_size=TEST_SIZE, random_state=random_state)

        sequence_train = df_train["sequence"].tolist()
        sequence_test = df_test["sequence"].tolist()
        sequence_synthetic = df_synthetic["sequence"].tolist()

        X_train = calculate_fingerprints(sequence_train)
        X_test = calculate_fingerprints(sequence_test)
        X_synthetic = calculate_fingerprints(sequence_synthetic)

        sequence_train, sequence_test, sequence_synthetic = np.array(sequence_train), np.array(sequence_test), np.array(sequence_synthetic)

        y_train = df_train["target_classification"].to_numpy()
        y_test = df_test["target_classification"].to_numpy()
        y_synthetic = df_synthetic["target_classification"].to_numpy()

        similarities = calculate_tanimoto_similarities(X_test, X_train)
        
        sequence_iid = sequence_test[similarities >= OOD_THRESHOLD]
        sequence_ood = sequence_test[similarities < OOD_THRESHOLD]

        X_iid = X_test[similarities >= OOD_THRESHOLD]
        X_ood = X_test[similarities < OOD_THRESHOLD]

        y_iid = y_test[similarities >= OOD_THRESHOLD]
        y_ood = y_test[similarities < OOD_THRESHOLD]

        sequence_dtype = get_smiles_dtype(sequence_train.tolist() + sequence_iid.tolist() + sequence_ood.tolist() + sequence_synthetic.tolist())
    
        np.savez(os.path.join(_out_dir, f"train.npz"), X=X_train, y=y_train, sequence=sequence_train.astype(sequence_dtype), classification_threshold=classification_parcentile_threshold)
        np.savez(os.path.join(_out_dir, f"iid.npz"), X=X_iid, y=y_iid, sequence=sequence_iid.astype(sequence_dtype), classification_threshold=classification_parcentile_threshold)
        np.savez(os.path.join(_out_dir, f"ood.npz"), X=X_ood, y=y_ood, sequence=sequence_ood.astype(sequence_dtype), classification_threshold=classification_parcentile_threshold)
        np.savez(os.path.join(_out_dir, f"synthetic.npz"), X=X_synthetic, y=y_synthetic, sequence=sequence_synthetic.astype(sequence_dtype), classification_threshold=classification_parcentile_threshold)
        
        print(f"Dataset constructed for seed {random_state} and saved to {_out_dir}")
        print(f"Train size: {len(X_train)} with class balance: {np.mean(y_train)}")
        print(f"IID size: {len(X_iid)} with class balance: {np.mean(y_iid)}")
        print(f"OOD size: {len(X_ood)} with class balance: {np.mean(y_ood)}")
        print(f"Synthetic size: {len(X_synthetic)} with class balance: {np.mean(y_synthetic)}")
        print(f"Classification threshold: {classification_parcentile_threshold}")

if __name__ == "__main__":
    args = _parse_args()
    main(out_dir=args.out_dir, data_filepath=args.data_filepath, data_filepath_synthetic=args.data_filepath_synthetic)
