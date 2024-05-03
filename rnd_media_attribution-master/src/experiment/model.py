from datetime import datetime
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from loguru import logger
from tensorflow.keras import Model, layers, metrics
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

from utils.files import PickleFile
from .dataset import Dataset, DatasetHandler


class MLModel(ABC):

    def __init__(self, dataset_class: Dataset = DatasetHandler):
        self.dataset = dataset_class()

    @property
    @abstractmethod
    def model_id(self) -> str:
        pass

    @abstractmethod
    def save(self, model_prefix: Path) -> Path:
        pass

    @abstractmethod
    def load(self, model_path: Path) -> None:
        pass

    @abstractmethod
    def predict(self, df: pd.DataFrame) -> Dict:
        pass

    @abstractmethod
    def fit(
        self,
        train_set: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        validation_set: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        **kwargs: Dict[str, Any]
    ) -> None:
        pass


class TokenAndPositionEmbedding(layers.Layer):

    def __init__(self, maxlen, vocab_size, embed_dim, **kwargs):
        super(TokenAndPositionEmbedding, self).__init__()
        self.maxlen = maxlen
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.token_emb = layers.Embedding(
            input_dim=vocab_size,
            output_dim=embed_dim
        )
        self.pos_emb = layers.Embedding(
            input_dim=maxlen,
            output_dim=embed_dim
        )

    def get_config(self):
        config = super(TokenAndPositionEmbedding, self).get_config()
        config.update({
            'maxlen': self.maxlen,
            'vocab_size': self.vocab_size,
            'embed_dim': self.embed_dim,
            'token_emb': layers.Embedding(
                input_dim=self.vocab_size,
                output_dim=self.embed_dim
            ),
            'pos_emb': layers.Embedding(
                input_dim=self.maxlen,
                output_dim=self.embed_dim
            )
        })
        return config

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions


class MultiHeadSelfAttention(layers.Layer):
    def __init__(self, embed_dim, num_heads=2, **kwargs):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embedding dimension = {embed_dim} \
                should be divisible by number of heads = {num_heads}"
            )
        self.projection_dim = embed_dim // num_heads
        self.query_dense = layers.Dense(embed_dim)
        self.key_dense = layers.Dense(embed_dim)
        self.value_dense = layers.Dense(embed_dim)
        self.combine_heads = layers.Dense(embed_dim)

    def get_config(self):
        config = super(MultiHeadSelfAttention, self).get_config()
        config.update({
            'embed_dim': self.embed_dim,
            'num_heads': self.num_heads,
            'projection_dim': self.embed_dim // self.num_heads,
            'query_dense': layers.Dense(self.embed_dim),
            'key_dense': layers.Dense(self.embed_dim),
            'value_dense': layers.Dense(self.embed_dim),
            'combine_heads': layers.Dense(self.embed_dim)
        })
        return config

    def attention(self, query, key, value):
        score = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_score = score / tf.math.sqrt(dim_key)
        weights = tf.nn.softmax(scaled_score, axis=-1)
        output = tf.matmul(weights, value)
        return output, weights

    def separate_heads(self, x, batch_size):
        x = tf.reshape(
            x,
            (batch_size, -1, self.num_heads, self.projection_dim)
        )
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs):
        # x.shape = [batch_size, seq_len, embedding_dim]
        batch_size = tf.shape(inputs)[0]
        query = self.query_dense(inputs)
        # (batch_size, seq_len, embed_dim)
        key = self.key_dense(inputs)
        # (batch_size, seq_len, embed_dim)
        value = self.value_dense(inputs)
        # (batch_size, seq_len, embed_dim)
        query = self.separate_heads(query, batch_size)
        # (batch_size, num_heads, seq_len, projection_dim)
        key = self.separate_heads(key, batch_size)
        # (batch_size, num_heads, seq_len, projection_dim)
        value = self.separate_heads(value, batch_size)
        # (batch_size, num_heads, seq_len, projection_dim)
        attention, weights = self.attention(query, key, value)
        attention = tf.transpose(attention, perm=[0, 2, 1, 3])
        # (batch_size, seq_len, num_heads, projection_dim)
        concat_attention = tf.reshape(
            attention,
            (batch_size, -1, self.embed_dim)
        )
        # (batch_size, seq_len, embed_dim)
        output = self.combine_heads(concat_attention)
        # (batch_size, seq_len, embed_dim)
        return output, weights


class AttentionModel(MLModel):

    @property
    def artifact_path(self) -> str:
        return Path(self.model_id) / 'model.h5'

    @property
    def model_id(self) -> str:
        return "attention_model"

    def save(self, model_dir: Path) -> Path:
        model_path = model_dir / self.artifact_path
        logger.info(f"Save model at {model_path}")
        self.model.save(str(model_path))
        PickleFile.write((str(model_path)+'tokenizer.pickle'), self.tokenizer)
        return model_path

    def __f1_metric(self, y_true, y_pred):
        K = tf.keras.backend
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        recall = true_positives / (possible_positives + K.epsilon())
        f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
        return f1_val

    def load(self, model_dir: Path) -> None:
        logger.info(f"Load model from {model_dir / self.artifact_path}")
        try:
            self.model = tf.keras.models.load_model(
                str(model_dir / self.artifact_path),
                custom_objects={
                    'TokenAndPositionEmbedding': TokenAndPositionEmbedding,
                    'MultiHeadSelfAttention': MultiHeadSelfAttention,
                    'f1_metric': self.__f1_metric,
                }
            )
        except FileExistsError as e:
            logger.error(e)
        except Exception as e:
            raise e

    def __build_model(
        self,
        hyperparameters: Dict[str, Any]
    ) -> None:
        logger.info("__build_model")

        METRICS = [
            metrics.BinaryAccuracy(name='accuracy'),
            metrics.Precision(name='precision'),
            metrics.Recall(name='recall'),
            metrics.AUC(name='auc'),
            self.__f1_metric,
        ]
#  UTIL LAYERS
        embedding_layer = TokenAndPositionEmbedding(
            hyperparameters['seq_maxlen'],
            hyperparameters['vocab_size'],
            hyperparameters['embed_dim']
        )
        attention_layer = MultiHeadSelfAttention(
            hyperparameters['embed_dim'],
            hyperparameters['num_heads']
        )
#     PATH INPUTS
        path_layer = layers.Input(
            shape=(hyperparameters['seq_maxlen'],),
            dtype='int32',
            name='input_path'
        )
        x_path = embedding_layer(path_layer)
        x_path, _ = attention_layer(x_path)
        x_path = layers.GlobalAveragePooling1D()(x_path)
        x_path = layers.Dropout(0.1)(x_path)
        x_path = layers.Dense(20, activation="relu", name='Dense_path')(x_path)
#     TIME-DECAY INPUTS
        td_layer = layers.Input(
            shape=(hyperparameters['seq_maxlen'],),
            dtype='float32',
            name='input_timedecay'
        )
        x_td = embedding_layer(td_layer)
        x_td, _ = attention_layer(x_td)
        x_td = layers.GlobalAveragePooling1D()(x_td)
        x_td = layers.Dropout(0.1)(x_td)
        x_td = layers.Dense(20, activation="relu", name='Dense_x_td')(x_td)
#     Concatenate path and time decay
        x_path_td = layers.Concatenate()([x_path, x_td])
        x_path_td = layers.Dropout(0.1)(x_path_td)
#     CUSTOMER-DEPENDENT INPUTS
        cd_input = layers.Input(
            shape=(hyperparameters['cus_dep_shape'],),
            dtype='int32',
            name='input_cd'
        )
        x_cd = layers.Dense(
            20,
            activation="relu",
            name='Dense_customer1'
        )(cd_input)
        x_cd = layers.Dropout(0.1)(x_cd)
        x_cd = layers.Dense(
            20,
            activation="relu",
            name='Dense_customer2'
        )(x_cd)
        x_cd = layers.Dropout(0.1)(x_cd)
#     OUPUTS
        x_path_td_cd = layers.Concatenate()([x_path_td, x_cd])
        outputs = layers.Dense(1, activation="sigmoid")(x_path_td_cd)

        model = Model(inputs=[path_layer, td_layer, cd_input], outputs=outputs)
        model.compile(
            optimizer="nadam",
            loss="binary_crossentropy",
            metrics=METRICS
        )
        self.model = model
        self.tokenizer = Tokenizer(
            num_words=hyperparameters['vocab_size'],
            split=','
        )

    def fit(
        self,
        df_dynamic: pd.DataFrame,
        X_control: np.ndarray,
        y: np.ndarray,
        tr_idx: np.ndarray,
        val_idx: np.ndarray,
        **kwargs: Dict[str, Any],
            ) -> None:
        assert "hyperparameters" in kwargs
        self.hyperparameters = kwargs["hyperparameters"]

        self.__build_model(
            self.hyperparameters,
        )

        corpus = df_dynamic['utm_hash'].to_list()
# tokenizer
        self.tokenizer.fit_on_texts(corpus)
        X_path = self.tokenizer.texts_to_sequences(corpus)
        seqleng = np.array([len(lspath) for lspath in X_path])
        zeroseq = (seqleng[seqleng == 0]).shape[0]
        logger.info(f'lost sequences: {(zeroseq)/len(X_path)}')
        self.X_path = pad_sequences(
            X_path,
            maxlen=self.hyperparameters['seq_maxlen'],
            padding='post'
        )

        self.X_TD = pad_sequences(
            df_dynamic['sess'].values,
            maxlen=self.hyperparameters['seq_maxlen'],
            dtype='float32',
            padding='post'
        )

        self.X_control = X_control
# cumulative token dataframe - to log the representativity of the words
        cum_tk_df = pd.DataFrame.from_dict(
            dict(
                self.tokenizer.word_counts
            ),
            orient='index',
            columns=['cnt']
        )
        cum_tk_df = cum_tk_df.sort_values(by='cnt', ascending=False)
        cum_tk_df['ratio'] = cum_tk_df / cum_tk_df.sum()
        cum_tk_df['cum_sum'] = cum_tk_df['ratio'].cumsum()
        cum_tk_df.reset_index(inplace=True)
        cum_tk_df = cum_tk_df.rename(columns={"index": "medium"})
        cum_tk_df['rank'] = cum_tk_df.index
        total_vocab_size = len(self.tokenizer.word_counts)+1
        embed_dim = self.hyperparameters['embed_dim']
        num_words = self.hyperparameters['vocab_size']
        logger.info(f'Hyperparameters vocab_size: {num_words}')
        logger.info(f'Embeding dim {embed_dim} \
         - sqrt used_vocab_size {np.ceil(np.sqrt(num_words))}')
        logger.info(f'Total vocabulary size: {total_vocab_size}')
        logger.info(f'Percentage of vocabulary \
            represented: {num_words / total_vocab_size}')
        last_wrd_cnt = cum_tk_df[
            cum_tk_df['rank'] < num_words
            ]['cnt'].tail(1).values
        logger.info(f'Last utm_hash count: {last_wrd_cnt}')
        pct_of_vocab_used = cum_tk_df[
            cum_tk_df['rank'] < num_words
            ]['cum_sum'].tail(1).values
        logger.info(f'Cumulative percentage of utm_hashs \
            represented: {pct_of_vocab_used}')
        del cum_tk_df

# Callbacks

        logdir = os.path.join(
                "/opt/ml/output/logs/",
                datetime.now().strftime("%Y%m%d-%H%M%S")
                )
        Path(logdir).mkdir(parents=True, exist_ok=True)
        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            logdir,
            histogram_freq=1
        )
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_recall',
            min_delta=0.02,
            patience=3,
            mode='max',
            restore_best_weights=False,
            verbose=1
        )
        logger.info(f"model_{self.model_id}.fit")
        historymh = self.model.fit(
            [self.X_path[tr_idx], self.X_TD[tr_idx], self.X_control[tr_idx]],
            y[tr_idx].astype(int),
            batch_size=self.hyperparameters['BATCH_SIZE'],
            epochs=self.hyperparameters['EPOCHS'],
            validation_data=(
                [
                    self.X_path[val_idx],
                    self.X_TD[val_idx],
                    self.X_control[val_idx]
                ],
                y[val_idx].astype(int)
            ),
            class_weight=self.hyperparameters['class_weight'],
            callbacks=[tensorboard_callback, early_stopping],
            verbose=2
        )
        return historymh, logdir

    def predict(self, test_idx: np.ndarray):
        attention_layer = self.model.layers[3]
        # or model.get_layer("attention”)
        attention_model = Model(
            inputs=self.model.inputs,
            outputs=self.model.outputs + [attention_layer.output]
        )
        # attention_model
        logger.info(f"model_{self.model_id}.predict")
        y_pred, _, att_weights = attention_model.predict(
            [
                # self.X_path[test_idx],
                # self.X_TD[test_idx],
                # self.X_control[test_idx]
                self.X_path,
                self.X_TD,
                self.X_control
            ],
            batch_size=self.hyperparameters['BATCH_SIZE']
        )
        # sum over heads [dim_corpus, dim_heads, dim_maxlen, dim_maxlen]
        att_weights = np.array(att_weights, dtype='float16')
        att_weights = ((att_weights.sum(axis=1)) / 2)
        return y_pred, att_weights
