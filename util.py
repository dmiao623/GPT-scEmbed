from anndata import AnnData
import copy
import numpy as np
import pandas as pd
import json
import pickle
from pathlib import Path
import scanpy as sc
from typing import Dict, List

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold

with open("config.json") as f:
    config = json.load(f)
data_dir = Path(config["data_dir"])

def retrieve_gpt_gene_embeddings(path: Path = data_dir / "GenePT_gene_embedding_ada_text.pickle"):
    if not path.exists():
        raise FileNotFoundError(f"Error: File '{path}' not found.")
    with open(path, "rb") as fp:
        gene_embeddings = pickle.load(fp)
    N_genes = len(gene_embeddings)
    N_dims = len(gene_embeddings[next(iter(gene_embeddings))])
    return (N_genes, N_dims, gene_embeddings)

def generate_random_embeddings(N_genes: int, N_dims: int, gene_names: List[str]):
    assert N_genes == len(gene_names)
    return {gene_name: np.random.normal(size=N_dims) for gene_name in gene_names}

def retrieve_gene_info(path: Path = data_dir / "gene_info_table.csv"):
    return pd.read_csv(path)

def generate_w_gpt_embeddings(N_dims: int, 
                              sc_data: AnnData, 
                              gpt_embeddings: Dict[str, np.ndarray]) -> np.ndarray:
    sc_data = sc_data.copy()
    sc.pp.normalize_total(sc_data)
    sc.pp.log1p(sc_data)
    
    if isinstance(sc_data.X, np.ndarray):
        X_dense = sc_data.X
    else:
        X_dense = sc_data.X.toarray()
    X_prob = X_dense / X_dense.sum(axis=1, keepdims=True)

    Y = []
    for gene in sc_data.var_names:
        embedding = gpt_embeddings.get(gene, np.zeros(N_dims))
        Y.append(embedding)
    Y = np.array(Y)

    return X_prob @ Y

def logistic_random_forest_eval(X_array_tf, y_array_tf, display=True):
    cv = StratifiedKFold(n_splits=10)

    roc_auc_logistic = []
    roc_auc_rf = []

    # Lists to store ROC AUC scores for each fold
    roc_auc_logistic = []
    roc_auc_rf = []

    # Lists to store TPR and FPR for each fold
    tpr_logistic = []
    fpr_logistic = []
    tpr_rf = []
    fpr_rf = []

    for train_index, test_index in cv.split(X_array_tf, y_array_tf):
        X_train, X_test = X_array_tf[train_index], X_array_tf[test_index]
        y_train, y_test = y_array_tf[train_index], y_array_tf[test_index]

        # Logistic Regression
        logistic_model = LogisticRegression()
        logistic_model.fit(X_train, y_train)
        y_score_logistic = logistic_model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_score_logistic)
        roc_auc = auc(fpr, tpr)
        roc_auc_logistic.append(roc_auc)
        tpr_logistic.append(tpr)
        fpr_logistic.append(fpr)

        # Random Forest
        random_forest_model = RandomForestClassifier()
        random_forest_model.fit(X_train, y_train)
        y_score_rf = random_forest_model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_score_rf)
        roc_auc = auc(fpr, tpr)
        roc_auc_rf.append(roc_auc)
        tpr_rf.append(tpr)
        fpr_rf.append(fpr)

    if display:
        print(f"Logistic Regression ROC AUC: {np.mean(roc_auc_logistic):.3f} +/- {np.std(roc_auc_logistic):.3f}")
        print(f"Random Forest ROC AUC: {np.mean(roc_auc_rf):.3f} +/- {np.std(roc_auc_rf):.3f}")

    return ((roc_auc_logistic, tpr_logistic, fpr_logistic), 
            (roc_auc_rf, tpr_rf, fpr_rf))
