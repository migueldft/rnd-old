import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
import pandas as pd
from loguru import logger
from matplotlib import pyplot as plt
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split

from xgboost import XGBClassifier

from .artifacts import ExperimentArtifacts


@dataclass(eq=False, order=False)
class Experiment(ABC):
    model: Any
    dataset: Any
    artifacts_handler: ExperimentArtifacts
    hyperparameters: Dict[str, Any]

    @abstractmethod
    def setup(self):
        pass

    @abstractmethod
    def run(self):
        pass


@dataclass(eq=False, order=False)
class SecondOrderExperiment(Experiment):
    run_tag: str

    def __post_init__(self):
        self.test_size = self.hyperparameters.pop('test_size', 0.2)
        model_id_str = self.model.model_id
        self.cutoff_period = int(model_id_str)

    def __split_data(self):
        X_train, X_test, y_train, y_test = train_test_split(
            self.dataset.X,
            self.targets,
            test_size=self.test_size,
            random_state=self.hyperparameters.get('random_seed', 42),
            stratify=self.targets,
        )
        self.train_set = (X_train, y_train)
        self.test_set = (X_test, y_test)

    def setup(self):
        logger.info(f"Setting up Experiment {self.run_tag} for model {self.model.model_id}.")

        def label_prep(raw_targets, cutoff_period):
            y = (raw_targets.waiting_time < cutoff_period).astype(int)
            return y
        self.targets = label_prep(self.dataset.y, self.cutoff_period)
        self.__split_data()

    def __plot_precision_recall_curve(self, pr_curve):
        fig = plt.figure(figsize=(12, 4))

        ax = fig.add_subplot(121)
        ax.set_title("Precision x Recall Curve")
        ax.plot(pr_curve.precision, pr_curve.recall)

        ax = fig.add_subplot(122)
        ax.set_title("Positive Rates")
        ax.plot(pr_curve.index, pr_curve.precision, label="Precision")
        ax.plot(pr_curve.index, pr_curve.recall, label="Recall")
        ax.plot(pr_curve.index, pr_curve.f1, label="F_1 Score")
        ax.legend()
        fig.tight_layout()
        return fig

    def __plot_auc_curve(self, score, target):
        fpr, tpr, roc_thres = roc_curve(target, score)
        fig = plt.figure(figsize=(12, 4))
        ax = fig.add_subplot(121)
        ax.set_title("ROC Curve - AUC")
        ax.plot(fpr, tpr)

        ax = fig.add_subplot(122)
        ax.set_title("Positive Rates")
        ax.plot(roc_thres, tpr, label="TPR")
        ax.plot(roc_thres, fpr, label="FPR")
        ax.set_xlim(0, 1)
        ax.legend()

        fig.tight_layout()
        return fig

    def __prec_recall_curve(self, score, target):
        n_points = min(100, np.unique(score).shape[0])
        thresholds = np.linspace(score.min(), score.max() * 0.999, n_points)

        metrics = pd.DataFrame(
            [
                precision_recall_fscore_support(target, (score >= threshold), average="binary")
                for threshold in thresholds
            ],
            columns=["precision", "recall", "f1", "support"],
            index=thresholds,
        )
        return metrics

    def fit(self):
        print(type(XGBClassifier), str(XGBClassifier))
        return self.model.fit(
            self.train_set,
            estimator=XGBClassifier,
            hyperparameters=self.hyperparameters,
        )

    def evaluate_model(self):
        X_test, y_test = self.test_set
        s_pred = self.model.test_predict(X_test)

        pr_curve = self.__prec_recall_curve(s_pred, y_test)
        optimal_threshold = pr_curve.f1.idxmax()

        return (
            {
                "test_roc_auc": roc_auc_score(y_test, s_pred),
                "precision_max_f1": pr_curve.loc[optimal_threshold, "precision"],
                "recall_max_f1": pr_curve.loc[optimal_threshold, "recall"],
                "f1_max_f1": pr_curve.loc[optimal_threshold, "f1"],
            },
            {
                "auc_curve": self.__plot_auc_curve(s_pred, y_test),
                "precision_recall_curve": self.__plot_precision_recall_curve(pr_curve),
            },
        )

    def run(self):
        logger.info(f"Begin Experiment {self.run_tag} for model {self.model.model_id}.")
        try:
            optimal_params = self.fit()
            metrics, figures = self.evaluate_model()
            logger.info(f"Train Finished: {metrics}")
        except Exception as e:
            # Write out an error file. This will be returned as the failureReason in the
            self.artifacts_handler.training_error(e)
            # A non-zero exit code causes the training job to be marked as Failed.
            sys.exit(255)

        self.artifacts_handler.save(
            {
                "model": self.model,
                "metrics": metrics,
                "figures": figures,
                "optimal_params": optimal_params,
            }
        )
