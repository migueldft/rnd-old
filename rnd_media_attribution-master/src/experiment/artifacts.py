import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

from loguru import logger

from utils.files import PickleFile, CSVFile, JsonFile


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


class ArtifactsHandler(ExperimentArtifacts):

    def __create_if_not_exist(self):
        Path(self.__model_prefix).mkdir(parents=True, exist_ok=True)

    @property
    def __model_prefix(self):
        return self.model_dir / self.model_id

    def training_error(self, error) -> None:
        # Write out an error file. This will be returned as
        # the failureReason in the DescribeTrainingJob result.
        trc = traceback.format_exc()
        with open(self.output_dir / 'failure', 'w') as s:
            s.write('Exception during training: ' + str(error) + '\n' + trc)
        # Printing this causes the exception to be
        # in the training job logs, as well.
        logger.info('Exception during training: ' + str(error) + '\n' + trc)

    def save(self, artifacts: Dict[str, Any]) -> None:
        self.model_id = artifacts["model"].model_id
        self.__create_if_not_exist()
        artifacts["model"].save(self.model_dir)

        metrics_path = str(self.__model_prefix / 'metrics.pkl')
        PickleFile.write(metrics_path, artifacts["metrics_clf"])

        hyperparameters_path = \
            str(self.__model_prefix / 'hyperparameters.json')
        JsonFile.write(hyperparameters_path, artifacts["hyperparameters"])

        df_tri_path = str(self.__model_prefix / 'df_tri.csv')
        CSVFile.write(df_tri_path, artifacts["df_tri"])
        df_tripos_path = str(self.__model_prefix / 'df_tripos.csv')
        CSVFile.write(df_tripos_path, artifacts["df_tripos"])

        df_utm_path = str(self.__model_prefix / 'df_utm.csv')
        CSVFile.write(df_utm_path, artifacts["df_utm"])
        df_utmpos_path = str(self.__model_prefix / 'df_utmpos.csv')
        CSVFile.write(df_utmpos_path, artifacts["df_utmpos"])

        rm_eff_path = str(self.__model_prefix / 'rm_effect.csv')
        CSVFile.write(rm_eff_path, artifacts["rm_effect"])

        rm_eff_path = str(self.__model_prefix / 'df_tri_re.csv')
        CSVFile.write(rm_eff_path, artifacts["df_tri_re"])

        for (name, figure) in artifacts["figures_clf"].items():
            file_name = self.__model_prefix / f"{name}.png"
            logger.debug(f"Saving {file_name}")
            figure.savefig(file_name, bbox_inches='tight')
