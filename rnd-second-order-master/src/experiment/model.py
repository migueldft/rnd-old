from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import QuantileTransformer

from category_encoders import OneHotEncoder, TargetEncoder
from loguru import logger

from .dataset import Dataset, SecondOrderDataset
from .optimizer import Optimizer

TARGET_ENCODING_COLUMNS = ["state", "partner", "channel"]
ONE_HOT_ENCODING_COLUMNS = ["device", "gender"]
CONTINOUS_COLUMNS = ["age", "gmv", "days_since_last_bf"]
STRATIFY_BY = ["state"]


class MLModel(ABC):

    def __init__(self, dataset_class: Dataset = SecondOrderDataset):
        self.dataset = dataset_class()

    @property
    def artifact_filename(self) -> str:
        return 'model.joblib'

    @abstractmethod
    def model_id(self) -> str:
        pass

    def save(self, model_dir: Path) -> Path:
        model_path = model_dir / self.model_id / self.artifact_filename
        logger.info(f"Save model at {model_path}")
        dump(self.model, model_path)
        return model_path

    def load(self, model_dir: Path) -> None:
        model_path = model_dir / self.model_id / self.artifact_filename
        logger.info(f"Load model from {model_path}")
        try:
            self.model = load(model_path)
        except FileExistsError as e:
            logger.error(e)
        except Exception as e:
            raise e

    def __build_model(self, random_seed=42, **model_params) -> None:
        logger.info("__build_model")
        transformers = []
        transformers.append((
            'te_features',
            TargetEncoder(
                smoothing=10.0,
                min_samples_leaf=10,
                verbose=10,
                cols=TARGET_ENCODING_COLUMNS,
                drop_invariant=True),
            TARGET_ENCODING_COLUMNS
        ))
        transformers.append((
            'ohe_features',
            OneHotEncoder(
                verbose=0,
                drop_invariant=True,
                return_df=True,
                handle_missing="indicator",
                handle_unknown="indicator",
                cols=ONE_HOT_ENCODING_COLUMNS,
                use_cat_names=True,
            ),
            ONE_HOT_ENCODING_COLUMNS
        ))
        transformers.append((
            'qte_features',
            QuantileTransformer(random_state=random_seed),
            CONTINOUS_COLUMNS
        ))
        preprocessor = ColumnTransformer(transformers=transformers)
        self.model = Pipeline(
            steps=[
                ('preprocessor', preprocessor),
                ('classifier', self.estimator(**model_params))
            ]
         )

    def fit(
        self,
        train_set: Tuple[pd.DataFrame, np.ndarray],
        estimator: BaseEstimator,
        hyperparameters: Dict[str, Any],
    ) -> None:

        self.estimator = estimator
        hyperspec = hyperparameters.get("optimization_hyperparameters", {})
        if hyperspec:
            logger.info("Hyperparameters optimization will be applied to this training.")
            random_seed = hyperspec.pop('random_seed', 42)
            cv_steps = hyperspec.pop('cv_steps', 5)
            n_jobs = hyperspec.pop('n_jobs', 8)
            n_opt_steps = hyperspec.pop('n_opt_steps', 2)
            n_random_starts = hyperspec.pop('n_random_starts', 2)

            def loss(**model_params):
                def evaluate_params(X, y, seed, cv_steps, n_jobs, **model_params):
                    self.__build_model(**model_params)

                    def roc_auc_scorer(estimator, X, y):
                        s_pred = estimator.predict_proba(X)[:, 1]
                        return roc_auc_score(y, s_pred)

                    logger.info("cross_val_score")
                    return np.mean(
                        cross_val_score(
                            self.model, X, y, cv=cv_steps,
                            n_jobs=n_jobs,
                            scoring=roc_auc_scorer
                        )
                    )
                return -evaluate_params(
                    *train_set,
                    random_seed,
                    cv_steps,
                    n_jobs,
                    **model_params
                )
            optimal_params = Optimizer.minimize(
                loss,
                hyperspec,
                n_opt_steps,
                n_random_starts,
                random_seed,
            )
        else:
            optimal_params = {}

        self.__build_model(**optimal_params)
        X_train, y_train = train_set

        assert X_train.shape[0] == len(y_train), "X and y must have same instances count."
        logger.info("model.fit")
        self.model.fit(X_train, y_train)
        return optimal_params

    def test_predict(self, X_test: Tuple[pd.DataFrame, np.ndarray]):
        logger.info("model.predict_proba")
        return self.model.predict_proba(X_test)[:, 1]

    def predict(self, payload: str) -> Dict[str, Any]:
        self.dataset.load_payload(payload)
        ids = self.dataset.ids
        X = self.dataset.X

        idxpreds = self.model.predict(X)
        probs = list(self.model.predict_proba(X))
        print(idxpreds)
        output = {"predictions": []}
        idx_to_label = self.dataset.index_label

        for id, idxpred, prob in zip(ids, idxpreds, probs):
            idxpred = int(idxpred)
            pred = idx_to_label[str(idxpred)]
            output['predictions'].append(
                [
                    int(id),
                    f'{pred}',
                    '{:.3f}'.format(prob[idxpred])
                ]
            )

        return output


class SecondOrder15Model(MLModel):

    @property
    def model_id(self) -> str:
        return "15"


class SecondOrder30Model(MLModel):

    @property
    def model_id(self) -> str:
        return "30"


class SecondOrder90Model(MLModel):

    @property
    def model_id(self) -> str:
        return "90"


class SecondOrder180Model(MLModel):

    @property
    def model_id(self) -> str:
        return "180"
