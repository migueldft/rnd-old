from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import pandas as pd

from loguru import logger
import psutil
import os

from utils.files import BaseFile, CSVFile


class Dataset(ABC):

    @property
    @abstractmethod
    def raw_dtypes(self) -> Dict[str, str]:
        pass

    @property
    @abstractmethod
    def label_index(self):
        pass

    @property
    @abstractmethod
    def index_label(self):
        pass

    @property
    @abstractmethod
    def load(self, preprocess=False) -> None:
        pass

    @property
    @abstractmethod
    def load_payload(self, payload: str) -> None:
        pass

    @property
    @abstractmethod
    def preprocess(self, training: bool = False) -> None:
        pass


@dataclass
class DatasetHandler(Dataset):
    file_handler: BaseFile = CSVFile
    data_path: str = ""
    extra_features: List = field(default_factory=list)
    max_seq_len: int = 15
    max_memory_usage: int = 10000
    # maxlen_threshold: float = 0.98

    @property
    def raw_dtypes(self) -> Dict[str, str]:
        return {
            'gid': 'object',
            'gender': 'object',
            'birthday': 'float64',
            'cust_created_year': 'float64',
            'country_region': 'object',
            # agg_cv: how many conv_groups in this agg_sess
            'agg_cvg': 'object',
            'agg_sessions': 'object'
        }

    @property
    def label_index(self):
        return {
            'NAO COMPROU': 0,
            'COMPROU': 1,
        }

    @property
    def index_label(self):
        return {str(v): k for k, v in self.label_index.items()}

    def load(self, chunksize=100000) -> None:
        dtypes = self.raw_dtypes
        self.df = (
            self.file_handler.read(
                self.data_path, chunksize=chunksize, dtypes=dtypes
                )
            )
        with open(self.data_path, 'r') as f:
            total_lines = sum(1 for line in f) - 1
        self.num_chunks = int(np.ceil(total_lines/chunksize))
        logger.info(f'num_chunks = {int(np.ceil(total_lines/chunksize))}')
        self.preprocess()

    def load_payload(self, payload: str) -> None:
        df = pd.read_json(payload, orient='records')
        df.columns = list(self.raw_dtypes.keys())
        df.astype(self.raw_dtypes)
        self.df = df

    def __memory_footprint(self):
        mem = psutil.Process(os.getpid()).memory_info().rss
        return (mem / 1024 ** 2)

    # def __calc_seq_maxlen_threshold(
    #     self,
    #     df: pd.DataFrame,
    # ) -> int:
    #     cum_path = (df['utm_hash'].apply(len).value_counts().cumsum() /
    #                 df['utm_hash'].apply(len).value_counts().sum())
    #     maxlen = (cum_path[cum_path <= self.maxlen_threshold]
    #               .tail(1))
    #     return maxlen.index[0]

    def __time_decay(self, arr):
        #  in days
        return (np.nan_to_num(np.array([(max(arr) - elt)/60/60/24
                                        for elt in arr])))

    def preprocess(self) -> None:
        # CONTROL VARIABLES BY CLIENT
        # creating conversion groups based on num_orders
        logger.info("Preprocessing...")
        rows_croped, total_shape = 0, 0
        df_total = pd.DataFrame()
        for chunk_idx, chunk in enumerate(self.df):  # df is a iterator
            # CUSTOMER PATH
            # Exploding sessions
            chunk['agg_sessions'] = chunk['agg_sessions'].str.split(';')
            chunk = chunk.explode('agg_sessions')
            chunk['agg_sessions'] = chunk['agg_sessions'].str.split(':')
            # get parameters from agg_sessions
    # creating columns spliting session informations
            chunk['utm_hash'] = (
                chunk['agg_sessions']
                .apply(lambda x: x[0])
                .str.split(',')
            )
            chunk['trinomials'] = (
                chunk['agg_sessions']
                .apply(lambda x: x[2])
                .str.replace(' ', '')
                .str.replace('1', 'kwDft')
                .str.replace('0', 'nokwDft')
                .str.split(',')
            )
            chunk['sess'] = (
                chunk['agg_sessions']
                .apply(lambda x: np.fromstring(x[1], dtype=int, sep=(',')))
                .apply(lambda x: self.__time_decay(x))
            )
            chunk['is_conversion'] = (
                chunk['agg_sessions']
                .apply(
                    lambda x: np.fromstring(
                        x[3],
                        dtype=float,
                        sep=(',')
                        )
                    )
                )
            chunk = chunk.drop(['agg_cvg', 'agg_sessions'], axis=1)
            # cropping on 15 touchpoints
            # even before the real mask
            # calculated (based on % of representative)
            seq_maxlen = self.max_seq_len
            mask = chunk['utm_hash'].apply(len) <= seq_maxlen
            fc = chunk[chunk['utm_hash'].apply(len) > seq_maxlen].shape[0]
            rows_croped = rows_croped + fc
            total_shape = total_shape + chunk.shape[0]
            pct_crop_on_chunk = np.round((fc*100 / chunk.shape[0]), 2)
            chunk = chunk[mask]
            df_total = df_total.append(chunk)
            mem_df = np.round(
                df_total.memory_usage(
                    index=True,
                    deep=True
                    ).sum()/(1024**2), 2
            )
            logger.debug(f'on chunk {chunk_idx} / {self.num_chunks} \
            croped {fc} rows: \
            {pct_crop_on_chunk} % - df memory usage: {mem_df} MB')
            maximum_memory_usage = self.max_memory_usage  # MB
            if self.__memory_footprint() > maximum_memory_usage:
                logger.info("Maximum memory usage reached")
                break
        # seq_maxlen = self.__calc_seq_maxlen_threshold(df_total)
        # logger.debug(f'SEQUENCE MAXLEN {seq_maxlen}')
        # assert seq_maxlen > 1, 'Try a bigger dataset range.\
        #      Seq_maxlen shall be > 1'
        # mask = df_total['utm_hash'].apply(len) <= seq_maxlen
        # df_total = df_total[mask]
        logger.debug(f'Cropped {rows_croped} rows')
        pct_croped = np.round((rows_croped*100 / total_shape), 2)
        logger.debug(f'{pct_croped} % of dataframe')
        self.df_dynamic = df_total[
                                ['utm_hash',
                                 'trinomials',
                                 'sess',
                                 'is_conversion']]
        self.df_control = df_total[
                                ['gid',
                                 'country_region',
                                 'gender',
                                 'birthday',
                                 'cust_created_year']]
        self.Ccols = (pd.get_dummies(
            self.df_control.drop(
                'gid', axis=1
                ),
            columns=[
                'country_region',
                'gender',
                ]
                )
            .columns
            )
        self.X_control = (pd.get_dummies(
            self.df_control.drop(
                'gid', axis=1
                ),
            columns=[
                'country_region',
                'gender',
                ]
                )
            .values
            )
        self.y = (df_total['is_conversion'] > 0).values.astype(int)
        self.seq_maxlen = seq_maxlen
        num_cust_without_hashid = df_total[
            (df_total['gid'].str.contains(':')) &
            (df_total['is_conversion'] > 0)
            ].shape[0]
        logger.info(f"Number of chunks processed: {chunk_idx + 1}")
        logger.info(f'Total df shape: {df_total.shape}')
        logger.debug(
            f'Total df memory usage: {mem_df} MB')
        logger.opt(colors=True).warning(
            f"<red>Number of customer with order and without \
                 any custom_dimensions8: {num_cust_without_hashid}</red>"
        )
        logger.info("Preprocessing Done!")
