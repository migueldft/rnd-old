import argparse
import configparser
import datetime
import json
import multiprocessing
import os
import pathlib
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict

from .files import JsonFile


class ArgParser(ABC):

    def __init__(self):
        self.environment = "Sagemaker" \
            if os.environ.get("SM_MODEL_DIR", False) \
            else os.environ.get("ENVIRON", "Default")

        config = configparser.ConfigParser()
        config.read(self.configuration_file_path)
        if self.environment == "Local":
            config[self.environment]["home_dir"] = str(pathlib.Path(__file__).parent.absolute())
        if self.environment != 'Sagemaker':
            config[self.environment]["SM_NUM_CPUS"] = str(multiprocessing.cpu_count())

        for key, value in config[self.environment].items():
            os.environ[key.upper()] = value

        self.run_tag = datetime.datetime \
            .fromtimestamp(time.time()) \
            .strftime('%Y-%m-%d-%H-%M-%S')

    @property
    def configuration_file_path(self) -> str:
        return 'config.ini'

    @property
    def hyperparameters_file_name(self) -> str:
        return "hyperparameters.json"

    @abstractmethod
    def get_arguments(self) -> Dict[str, Any]:
        pass


class APIArgParser(ArgParser):

    def get_arguments(self) -> Dict[str, Any]:
        parser = argparse.ArgumentParser()
        parser.add_argument(
            '--model_dir',
            type=Path,
            default=os.environ['SM_MODEL_DIR'],
        )
        parser.add_argument(
            '--model_name',
            default='second-order',
            type=str,
            help="Project name",
        )
        parser.add_argument(
            '--num_cpus',
            type=int,
            default=os.environ['SM_NUM_CPUS'],
        )
        parser.add_argument(
            '--model_server_timeout',
            default=60,
            type=Path,
            help="Number of model server workers (default: 60)",
        )
        parser.add_argument(
            '--run_tag',
            default=self.run_tag,
            type=str,
            help=f"Run ID (default: '{self.run_tag}')",
        )
        args = parser.parse_args()

        return args
