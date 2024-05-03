from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta
from math import ceil
from typing import Dict, List

import dask.dataframe as dd
import pandas as pd
from loguru import logger
from pandas import to_datetime
from unidecode import unidecode

from utils.files import BaseFile, ParquetFile

BLACK_FRIDAY = {
    2014: to_datetime("2014-11-28"),
    2015: to_datetime("2015-11-27"),
    2016: to_datetime("2016-11-25"),
    2017: to_datetime("2017-11-24"),
    2018: to_datetime("2018-11-23"),
    2019: to_datetime("2019-11-29"),
}

SECONDS_PER_YEAR = 86400.0


class Dataset(ABC):

    @property
    @abstractmethod
    def raw_dtypes(self) -> Dict[str, str]:
        pass

    @property
    @abstractmethod
    def id_column(self) -> List:
        pass

    @property
    @abstractmethod
    def feature_columns(self) -> List:
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
    def save(self, output_path: str) -> None:
        pass

    @property
    @abstractmethod
    def preprocess(self, training: bool = False) -> None:
        pass


@dataclass
class SecondOrderDataset(Dataset):
    file_handler: BaseFile = ParquetFile
    data_path: str = ""

    @property
    def raw_dtypes(self) -> Dict[str, str]:
        return {
            'fk_customer': 'object',
            'channel': 'object',
            'partner': 'object',
            'device': 'object',
            'first_sale_number': 'int',
            'age': 'int',
            'gender': 'object',
            'state': 'object',
            'expected_delivery_date': 'object',
            'delivered_date': 'object',
            'first_sale_date': 'object',
            'second_sale_date': 'object',
            'has_marketplace': 'int',
            'has_crossdocking': 'int',
            'has_private_label': 'int',
            'has_brands': 'int',
            'gmv': 'float',
        }

    @property
    def __training_dtypes(self) -> Dict[str, str]:
        return {
            'fk_customer': 'int',
            'channel': 'object',
            'partner': 'object',
            'device': 'object',
            'age': 'int',
            'gender': 'object',
            'state': 'object',
            'has_marketplace': 'bool',
            'has_crossdocking': 'bool',
            'has_private_label': 'bool',
            'has_brands': 'bool',
            'gmv': 'float',
            'days_since_last_bf': 'int',
            'waiting_time': 'int',
            'has_second_purchase': 'bool',
        }

    @property
    def __dates_columns(self) -> List:
        return [
            'first_sale_date',
            'second_sale_date',
            'delivered_date',
            'expected_delivery_date',
        ]

    @property
    def label_column(self) -> List:
        return ['waiting_time', 'has_second_purchase']

    @property
    def id_column(self) -> List:
        return 'fk_customer'

    @property
    def feature_columns(self) -> List:
        return list(self.__training_dtypes.keys())[1:-2]

    @property
    def label_index(self):
        return {
            'does_not_has_second_purchase': 0,
            'has_second_purchase': 1,
        }

    @property
    def index_label(self):
        return {str(v): k for k, v in self.label_index.items()}

    def __parse_dates(self) -> None:
        for col in self.__dates_columns:
            if col in self.df.columns:
                self.df[col] = to_datetime(
                    self.df[col],
                    format='%Y%m%d',
                    errors='coerce'
                )

    def load(self, preprocess: bool = True) -> None:
        dtypes = self.raw_dtypes
        df = self.file_handler.read(self.data_path, dtypes)
        self.df = df.compute()
        self.df.fillna(value=pd.np.nan, inplace=True)
        self.preprocess(training=True)

    def load_payload(self, payload: str) -> None:
        df = pd.read_json(payload, orient='instances')
        df = df.astype(self.raw_dtypes)
        self.df = df
        self.preprocess()

    def save(self, output_path: str) -> None:
        npartitions = ceil(self.df.memory_usage().sum() / 100_000_000)
        df = dd.from_pandas(self.df, npartitions=npartitions)
        self.file_handler.write(output_path, df)

    def preprocess(self, training: bool = False) -> None:
        logger.debug("Begin preprocessing...")
        self.ids = self.df[self.id_column].tolist()

        n_rows_original = self.df.shape[0]
        date_next_year = (datetime.today() + timedelta(days=365)).strftime("%Y%m%d")
        self.df.second_sale_date = self.df.second_sale_date.fillna(date_next_year)
        self.df = self.df.dropna()

        self.__parse_dates()

        logger.debug(f"Dropped {n_rows_original - self.df.shape[0]} rows with nan values.")

        def fix_state_information(state_series):
            return (
                state_series.apply(unidecode)
                .str.replace("b'", "")
                .str.replace("'", "")
                .str.replace(" ", "_")
            )

        def get_last_black_friday_date(date):
            current_year_bf = BLACK_FRIDAY[date.year]
            last_year_bf = BLACK_FRIDAY[date.year - 1]
            return last_year_bf if date <= current_year_bf else current_year_bf

        def get_days_since_black_friday(date_series):
            last_bf_date = date_series.apply(get_last_black_friday_date)
            return (date_series - last_bf_date).dt.total_seconds().div(SECONDS_PER_YEAR)

        self.X = (
            self.df.assign(state=lambda df: fix_state_information(df.state))
            .assign(days_since_last_bf=lambda df: get_days_since_black_friday(df.first_sale_date))
            .loc[:, self.feature_columns]
        )

        if training:
            waiting_time = (
                (self.df.second_sale_date - self.df.first_sale_date)
                .dt.total_seconds()
                .div(SECONDS_PER_YEAR)
            )
            has_second_purchase = ~waiting_time.isnull()
            self.y = pd.DataFrame({"waiting_time": waiting_time, "has_second_purchase": has_second_purchase})
        else:
            self.y = None
        logger.debug("Preprocessing done!")
