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

from .files import YAMLFile


class ArgParser(ABC):

    def __init__(self):
        self.environment = "Docker" \
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

    @abstractmethod
    def get_arguments(self) -> Dict[str, Any]:
        pass


class TrainArgParser(ArgParser):

    def get_hyperparameters(self, hyperparameters_dir: Path, hyperparameters_file_name: str) -> Dict[str, Any]:
        hyperparam_file_path = hyperparameters_dir / hyperparameters_file_name
        if hyperparam_file_path.exists():
            hyperparameters = YAMLFile.read(hyperparam_file_path)
            return hyperparameters
        else:
            return {}

    def get_arguments(self) -> Dict[str, Any]:
        parser = argparse.ArgumentParser()

        parser.add_argument(
            '--model_dir',
            type=Path,
            default=os.environ['SM_MODEL_DIR'],
        )
        parser.add_argument(
            '--channel_names',
            default=json.loads(os.environ['SM_CHANNELS']),
        )
        parser.add_argument(
            '--num_gpus',
            type=int,
            default=os.environ['SM_NUM_GPUS'],
        )
        parser.add_argument(
            '--num_cpus',
            type=int,
            default=os.environ['SM_NUM_CPUS'],
        )
        parser.add_argument(
            '--user_args',
            default=json.loads(os.environ['SM_USER_ARGS']),
        )
        parser.add_argument(
            '--input_dir',
            type=Path,
            default=os.environ['SM_INPUT_DIR'],
        )
        parser.add_argument(
            '--input_config_dir',
            type=Path,
            default=os.environ['SM_INPUT_CONFIG_DIR'],
        )
        parser.add_argument(
            '--output_dir',
            type=Path,
            default=os.environ['SM_OUTPUT_DIR'],
        )

        # Extra arguments
        parser.add_argument(
            '--run_tag',
            default=self.run_tag,
            type=str,
            help="Run tag (default: 'datetime.fromtimestamp')",
        )
        parser.add_argument(
            '--hyperparameters_file_name',
            type=str,
            default="optimizer_hyperparameters.yaml",
            help="Hyperparameters file name (default: 'optimizer_hyperparameters.yaml')",
        )

        args = parser.parse_args()
        d = vars(args)
        d["hyperparameters"] = \
            self.get_hyperparameters(args.input_config_dir, d["hyperparameters_file_name"])

        return args
