import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

from loguru import logger

from utils.files import JsonFile, YAMLFile


@dataclass(eq=False, order=False)
class ExperimentArtifacts(ABC):
    output_dir: Path
    model_dir: Path

    @abstractmethod
    def training_error(self, error) -> None:
        pass

    @abstractmethod
    def save(self, artifacts: Dict[str, Any]) -> None:
        pass


class SecondOrderArtifacts(ExperimentArtifacts):

    def __create_if_not_exist(self):
        Path(self.__model_prefix).mkdir(parents=True, exist_ok=True)

    @property
    def __model_prefix(self):
        return self.model_dir / self.model_id

    def training_error(self, error) -> None:
        # Write out an error file. This will be returned as the failureReason in the
        # DescribeTrainingJob result.
        trc = traceback.format_exc()
        with open(self.output_dir / 'failure', 'w') as s:
            s.write('Exception during training: ' + str(error) + '\n' + trc)
        # Printing this causes the exception to be in the training job logs, as well.
        logger.info('Exception during training: ' + str(error) + '\n' + trc)

    def save(self, artifacts: Dict[str, Any]) -> None:
        assert "model" in artifacts
        self.model_id = artifacts["model"].model_id
        self.__create_if_not_exist()
        artifacts["model"].save(self.model_dir)

        metrics = artifacts.get("metrics")
        if metrics:
            metrics_path = self.__model_prefix / 'metrics.json'
            JsonFile.write(metrics_path, metrics)
        optimal_params = artifacts.get("optimal_params")
        if optimal_params:
            optimal_params_path = self.__model_prefix / 'optimal_params.yml'
            YAMLFile.write(optimal_params_path, optimal_params)

        figures = artifacts.get("figures")
        if figures:
            for (name, figure) in figures.items():
                file_name = self.__model_prefix / f"{name}.png"
                logger.debug(f"Saving {file_name}")
                figure.savefig(file_name)
