""" Run synthetic OOD benchmark. """

import os
import json
from typing import Optional

import numpy as np

from src.probes.knn import KNNProbe
from src.probes.rf import RFProbe

from src.probes.evaluate import evaluate_classification
from src.data.similarity import calculate_tanimoto_similarities


def _load_data(data_filepath):
    data = np.load(data_filepath)
    return data["X"], data["y"], data["sequence"], data["classification_threshold"]


def _save_results(results, out_dir, name):
    with open(os.path.join(out_dir, f"{name}.json"), "w") as f:
        json.dump(results, f, indent=4)


def main(data_dir: str, out_dir: str, probe_name: str, experiment_name: Optional[str] = None) -> dict:

    os.makedirs(out_dir, exist_ok=True)
    
    auprcs_iid = []
    similarities_iid = []
    
    auprcs_ood = []
    similarities_ood = []
    
    auprcs_synthetic = []
    similarities_synthetic = []

    for random_state in range(5):
        print("-"*100)
        print(f"Random state: {random_state}")
        print("-"*100)

        X_train, y_train, _, _ = _load_data(os.path.join(data_dir, f"seed_{random_state}/train.npz"))
        X_iid, y_iid, _, _ = _load_data(os.path.join(data_dir, f"seed_{random_state}/iid.npz"))
        X_ood, y_ood, _, _ = _load_data(os.path.join(data_dir, f"seed_{random_state}/ood.npz"))
        X_synthetic, y_synthetic, _, _ = _load_data(os.path.join(data_dir, f"seed_{random_state}/synthetic.npz"))

        print("Train size:", len(X_train), "with class balance:", np.mean(y_train))
        print("IID size:", len(X_iid), "with class balance:", np.mean(y_iid))
        print("OOD size:", len(X_ood), "with class balance:", np.mean(y_ood))
        print("Synthetic size:", len(X_synthetic), "with class balance:", np.mean(y_synthetic))

        if probe_name == "knn":
            probe = KNNProbe(k=[1, 5, 10, 20, 50, 100], metric='jaccard', weights=['uniform', 'distance', 'soft-distance'], random_state=random_state)
        elif probe_name == "rf":
            probe = RFProbe(cv_folds=5, random_state=random_state)
        else:
            raise ValueError(f"Unknown probe name: {probe_name}")

        probe.fit(X_train, y_train, selection_metric='auprc', n_jobs=8)

        print("--- Results (IID) ---")
        y_pred_iid = probe.predict(X_iid)
        results_iid = evaluate_classification(y_iid, y_pred_iid)
        print("AUPRC (IID):", results_iid["AUPRC"])
        auprcs_iid.append(results_iid["AUPRC"])
        similarities_iid = calculate_tanimoto_similarities(X_iid, X_train)
        print("Mean similarity to train (IID):", similarities_iid.mean())
        
        print("--- Results (OOD) ---")
        y_pred_ood = probe.predict(X_ood)
        results_ood = evaluate_classification(y_ood, y_pred_ood)
        print("AUPRC (OOD):", results_ood["AUPRC"])
        auprcs_ood.append(results_ood["AUPRC"])
        similarities_ood = calculate_tanimoto_similarities(X_ood, X_train)
        print("Mean similarity to train (OOD):", similarities_ood.mean())

        print("--- Results (Synthetic) ---")
        y_pred_synthetic = probe.predict(X_synthetic)
        results_synthetic = evaluate_classification(y_synthetic, y_pred_synthetic)
        print("AUPRC (Synthetic):", results_synthetic["AUPRC"])
        auprcs_synthetic.append(results_synthetic["AUPRC"])
        similarities_synthetic = calculate_tanimoto_similarities(X_synthetic, X_train)
        print("Mean similarity to train (Synthetic):", similarities_synthetic.mean())
    
    # aggregate
    auprcs_iid = np.array(auprcs_iid)
    auprcs_ood = np.array(auprcs_ood)
    auprcs_synthetic = np.array(auprcs_synthetic)
    similarities_iid = np.array(similarities_iid)
    similarities_ood = np.array(similarities_ood)
    similarities_synthetic = np.array(similarities_synthetic)
    
    # report
    print("Mean AUPRC (IID):", auprcs_iid.mean().round(3), "±", auprcs_iid.std().round(3))
    print("Mean similarity (IID) to train:", similarities_iid.mean().round(3), "±", similarities_iid.std().round(3))
    print("Mean AUPRC (OOD):", auprcs_ood.mean().round(3), "±", auprcs_ood.std().round(3))
    print("Mean similarity (OOD) to train:", similarities_ood.mean().round(3), "±", similarities_ood.std().round(3))
    print("Mean AUPRC (Synthetic):", auprcs_synthetic.mean().round(3), "±", auprcs_synthetic.std().round(3))
    print("Mean similarity (Synthetic) to train:", similarities_synthetic.mean().round(3), "±", similarities_synthetic.std().round(3))

    # save
    results = {
        "auprc_iid": auprcs_iid.tolist(),
        "auprc_ood": auprcs_ood.tolist(),
        "auprc_synthetic": auprcs_synthetic.tolist(),
        "similarities_iid": similarities_iid.tolist(),
        "similarities_ood": similarities_ood.tolist(),
        "similarities_synthetic": similarities_synthetic.tolist(),
    }

    _save_results(results, out_dir, f"{probe_name}_{experiment_name}" if experiment_name else probe_name)

    return results
