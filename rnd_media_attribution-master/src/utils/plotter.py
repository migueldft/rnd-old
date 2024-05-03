from typing import Any, Sequence, Tuple

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                             plot_confusion_matrix, confusion_matrix,
                             precision_recall_fscore_support, roc_curve)


class Plotter:

    def plot_tf_metrics(
        self,
        history
    ):
        METRICS = [s for s in list(history.history.keys()) if "val" not in s]
        plt.rcParams['figure.figsize'] = (12, 10)
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        fig_name = 'tf_history_metrics'
        fig = plt.figure(figsize=(12, 10))
        for n, metric in enumerate(METRICS):
            name = metric.replace("_", " ").capitalize()
            ax = fig.add_subplot(np.ceil(len(METRICS)/2).astype(int), 2, n+1)
            ax.plot(history.epoch, history.history[metric],
                    color=colors[0], label='Train')
            ax.plot(history.epoch, history.history['val_'+metric],
                    color=colors[0], linestyle="--", label='Val')
            ax.set_xlabel('Epoch')
            ax.set_ylabel(name)
            ax.legend()
        return fig_name, fig

    def plot_cm(
        self,
        labels,
        predictions,
        p=0.5
    ):
        if predictions.shape[1] > 1:
            predictions = ((predictions > p)[:,  1]).astype(int)
            cm = confusion_matrix(labels, predictions)
        else:
            cm = confusion_matrix(labels, predictions > p)
        _, ax = plt.subplots(figsize=(5, 5))
        sns.heatmap(cm, annot=True, fmt="d", ax=ax,
                    linewidths=.5, annot_kws={"size": 14}
                    )
        fig_name = 'confusion_matrix'
        plt.title('Confusion matrix @{:.2f}'.format(p))
        plt.ylabel('Actual label')
        plt.xlabel('Predicted label')
        return fig_name, ax.get_figure()

    def plot_comparison_attribution_methods(
        self,
        df: pd.DataFrame,
        df_pos: pd.DataFrame,
        fig_name,
        fig_name_pos,
        max_mediuns: np.int
    ):
        # fig_name = 'Attribution methods comparison'
        _, ax = plt.subplots(figsize=(15, 12))
        df = df.sort_values(
            by='norm_att', ascending=False
            ).iloc[0:max_mediuns]
        df[
            ['norm_att', 'first_touch', 'last_touch', 'position_based']
            ].plot.barh(ax=ax, width=0.70)
        plt.title(fig_name)
        ax.set_title('Attribution methods comparison', size=26)
        ax.legend(prop={'size': 22})
        ax.set_xlabel('Attribution score', size=22)
        ax.set_ylabel('Medium', size=22)
        ax.tick_params(axis='y', labelsize=16)
        ax.tick_params(axis='x', labelsize=16)
        fig = ax.get_figure()

        # fig_name_pos = 'DNAM Attribution score per touch point'
        _, ax = plt.subplots(figsize=(15, 12))
        df_pos = df_pos.sort_values(by='cnt', ascending=False)[:max_mediuns]
        sns.heatmap(
            df_pos[df_pos.columns[:-2]],
            linewidths=.5, ax=ax,
            annot=True, annot_kws={"size": 14}
        )
        ax.set_title(fig_name_pos, size=26)
        ax.set_xlabel('Touch point', size=22)
        ax.set_ylabel('Medium', size=22)
        ax.tick_params(axis='y', labelsize=16)
        ax.tick_params(axis='x', labelsize=16)
        fig_pos = ax.get_figure()
        return fig_name, fig, fig_name_pos, fig_pos

    def plot_weight_by_tp(
        self,
        df: pd.DataFrame,
        W: np.float,
        fig_name,
        path_len: np.int
    ):
        seqleng = df['utm_hash'].apply(len).values
        font_size = {
            'title': 30,
            'subtitle': 26,
            'legend': 18,
            'labels': 22,
            'ticklabels': 18
            }
        #  weight sequence length
        df_wsl = pd.DataFrame(index=np.arange(path_len)+1)
        for ii in range(path_len):
            a = ii+1
            df_wsl[str(a)] = np.nan
            arr = (
                (W[seqleng == a, :a, :a]).sum(axis=1) /
                ((W[seqleng == a, :a, :a]).sum(axis=1))
                .sum()).sum(axis=0)
            df_wsl[str(a)] = np.pad(
                arr, (0, path_len-a),
                mode='constant',
                constant_values=(np.nan,)
                )
        figax, ax = plt.subplots(1, 2, figsize=(18, 8))
        figax.suptitle(fig_name, size=font_size['title'])
        df_wsl.plot(ax=ax[0], lw=4)
        ax[0].set_ylim(0.05, 0.6)
        ax[0].set_xlim(0.5, 10.5)
        ax[0].set_xticks(np.arange(path_len)+1)
        ax[0].grid()
        ax[0].set_title(
            'Weight normalized on sum',
            size=font_size['subtitle']
            )
        ax[0].legend(prop={'size': font_size['legend']})
        ax[0].set_xlabel('Touchpoint', size=font_size['labels'])
        ax[0].set_ylabel('Weight', size=font_size['labels'])
        ax[0].tick_params(axis='y', labelsize=font_size['ticklabels'])
        ax[0].tick_params(axis='x', labelsize=font_size['ticklabels'])

        df_wsl.div(df_wsl.max(axis=0)).plot(ax=ax[1], lw=4)
        ax[1].set_xticks(np.arange(path_len)+1)
        ax[1].grid()
        ax[1].set_title(
            'Weight normalized by path length',
            size=font_size['subtitle']
            )
        ax[1].legend(prop={'size': font_size['legend']})
        ax[1].set_xlabel('Touchpoint', size=font_size['labels'])
        ax[1].set_ylabel('Weight', size=font_size['labels'])
        ax[1].tick_params(axis='y', labelsize=font_size['ticklabels'])
        ax[1].tick_params(axis='x', labelsize=font_size['ticklabels'])
        # fig = figax.get_figure()
        return fig_name, figax

    def plot_precision_recall_curve(
        self,
        pr_curve: pd.DataFrame
    ) -> plt.figure:
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

    def plot_auc_curve(self, score, target) -> plt.figure:
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

    def prec_recall_curve(self, score, target) -> pd.DataFrame:
        n_points = min(100, np.unique(score).shape[0])
        thresholds = np.linspace(score.min(), score.max() * 0.999, n_points)

        metrics = pd.DataFrame(
            [
                precision_recall_fscore_support(
                    target,
                    (score >= threshold),
                    average="binary"
                )
                for threshold in thresholds
            ],
            columns=["precision", "recall", "f1", "support"],
            index=thresholds,
        )

        return metrics

    def plot_confusion_matrix(
        self,
        model: Any,
        X: pd.DataFrame,
        y: np.ndarray
    ) -> plt.figure:
        fig = plt.figure(figsize=(12, 4))

        fig = plot_confusion_matrix(
            model,
            X,
            y,
            display_labels=['NAO IIA COMPRAR', 'IRA COMPRAR'],
            cmap=plt.cm.Blues,
            normalize='true'
        ).figure_

        return fig

    def plot_feature_importance(self, model: Any, columns_name) -> plt.figure:
        fig = plt.figure(figsize=(7, 15))

        df = pd.DataFrame(
            {
                'imp': model.feature_importances_,
                'col': columns_name
            }
        ).sort_values(by='imp', ascending=False)
        fig = df[0:20].plot.barh(
            x='col',
            y='imp',
            figsize=(7, 15),
            legend=None,
            color='black'
        ).get_figure()

        fig.tight_layout()

        return fig

    def _cumulative_true(
        self,
        y_true: Sequence[float],
        y_pred: Sequence[float]
    ) -> np.ndarray:
        df = pd.DataFrame({
            'y_true': y_true,
            'y_pred': y_pred,
        }).sort_values(by='y_pred', ascending=False)

        return (df['y_true'].cumsum() / df['y_true'].sum()).values

    def gain_and_gini(
        self,
        y_true: Sequence[float],
        y_pred: Sequence[float],
        y_base: Sequence[float]
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        gain = pd.DataFrame({
            'lorenz': self._cumulative_true(y_true, y_true),
            'baseline': self._cumulative_true(y_true, y_base),
            'model': self._cumulative_true(y_true, y_pred),
        })
        num_customers = np.float32(gain.shape[0])
        gain['cumulative_customer'] = (
            np.arange(num_customers) + 1.
        ) / num_customers

        df = gain[['lorenz', 'baseline', 'model']]
        raw = df.apply(lambda x: 2 * x.sum() / df.shape[0] - 1.)
        normalized = raw / raw[0]
        gini = pd.DataFrame({
            'raw': raw,
            'normalized': normalized
        })[['raw', 'normalized']]

        return gain, gini

    def _aggregate_fn(self, df: pd.DataFrame) -> pd.Series:
        return pd.Series({
            'label_mean': np.mean(df['y_true']),
            'pred_mean': np.mean(df['y_pred']),
            'normalized_rmse': (
                np.sqrt(
                    mean_squared_error(df['y_true'], df['y_pred'])
                ) / df['y_true'].mean()
            ),
            'normalized_mae': (
                mean_absolute_error(
                    df['y_true'], df['y_pred']
                ) / df['y_true'].mean()
            ),
        })

    def decile_stats(
        self,
        y_true: Sequence[float],
        y_pred: Sequence[float],
        q: int
    ) -> pd.DataFrame:
        decile = pd.qcut(
            y_pred, q=q, labels=['%d' % i for i in range(q)])

        df = pd.DataFrame({
            'y_true': y_true,
            'y_pred': y_pred,
            'decile': decile,
        }).groupby('decile').apply(self._aggregate_fn)

        df['decile_mape'] = (
            np.abs(df['pred_mean'] - df['label_mean']) / df['label_mean']
        )

        return df

    def plot_distribution(self, df: pd.DataFrame) -> plt.figure:
        fig = plt.figure(figsize=(12, 4))

        df = df[(df['y_true'] > 0) & (df['y_pred'] > 0)]
        fig = sns.jointplot(
            x=np.log(df['y_true']),
            y=np.log(df['y_pred']),
            kind='kde'
        )

        return fig

    def plot_gain_chart(self, gain: pd.DataFrame) -> plt.figure:
        fig = plt.figure(figsize=(12, 4))

        ax = gain[[
            'cumulative_customer',
            'lorenz',
            'baseline',
            'model',
        ]].plot(x='cumulative_customer', figsize=(8, 5), legend=True)

        ax.set_title('Gain Chart')
        ax.legend(['Groundtruth', 'Baseline', 'Model'], loc='lower right')

        ax.set_xlabel('Cumulative Fraction of Customers')
        ax.set_xticks(np.arange(0, 1.1, 0.1))
        ax.set_xlim((0, 1.))

        ax.set_ylabel('Cumulative Fraction of Total Lifetime Value')
        ax.set_yticks(np.arange(0, 1.1, 0.1))
        ax.set_ylim((0, 1.05))

        fig = ax.get_figure()
        fig.tight_layout()

        return fig

    def plot_decile_chart(self, df: pd.DataFrame) -> plt.figure:
        fig = plt.figure(figsize=(12, 4))

        ax = df[['label_mean', 'pred_mean']].plot.bar(rot=0)
        ax.set_title('Decile Chart')
        ax.set_xlabel('Prediction bucket')
        ax.set_ylabel('Average bucket value')
        ax.legend(['Label', 'Prediction'], loc='upper left')

        fig = ax.get_figure()
        fig.tight_layout()

        return fig
