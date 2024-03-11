from pathlib import Path

import dvc.api
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from dvclive import Live
from helper import load_data,save_data
from mlem.api import save

def create_pipeline() -> Pipeline:
    return Pipeline([("xgb", XGBClassifier())])


def train_model(
    X_train,
    y_train,
):
    print('======================================================')
    print("Training Classifier : ", 'xgb')
    return XGBClassifier().fit(X_train, y_train)

def save_model(model, path: str, X_train: pd.DataFrame):
    """Save model to path"""
    Path(path).parent.mkdir(exist_ok=True)
    save(model, path, sample_data=X_train)

def train() -> None:
    """Train model and save it"""
    params = dvc.api.params_show()
    with Live(save_dvc_exp=True) as live:
        print('Loading data...')
        X_train = load_data(f"{params['data']['preprocessed']}/train.npy")
        y_train = load_data(f"{params['data']['preprocessed']}/train_labels.npy")
        #pipeline = create_pipeline()
        model=train_model(
            X_train,
            y_train
        )
        #live.log_params({"Best hyperparameters": grid_search.best_params_})
        #save_data(model, params["model"], 'model.pkl')
        save_model(model, params["model"], X_train)


if __name__ == "__main__":
    train()