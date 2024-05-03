import csv
import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import falcon
from loguru import logger

from experiment.model import MLModel


@dataclass
class PredictionService:
    """
        A singleton for holding the model. This simply loads
        the model and holds it.
        It has a predict function that does a prediction based
        on the model and the input data.
    """
    def __init__(self, models_classes: Dict[str, MLModel], model_dir: Path) -> None:
        self.model_dict = {}
        for model_name, model_class in models_classes.items():
            logger.info(f"Loading model: '{model_name}'")
            model = model_class()
            model.load(model_dir)
            self.model_dict[model.model_id] = model

    def on_post(self, request, response):

        if request.content_type == "application/json":
            payload = request.stream.read().decode("utf-8")
        elif request.content_type == "text/csv":
            decoded_payload = request.stream.read().decode("utf-8")
            reader = csv.DictReader(io.StringIO(decoded_payload))
            payload = json.dumps(list(reader))
        else:
            raise falcon.HTTPUnsupportedMediaType('json please!')

        preds = {}
        for model_name, model in self.model_dict.items():
            logger.info(f'Generated predictions for model "{model_name}".')
            preds[model_name] = model.predict(payload)

        response.status = falcon.HTTP_OK
        response.body = json.dumps(preds)
