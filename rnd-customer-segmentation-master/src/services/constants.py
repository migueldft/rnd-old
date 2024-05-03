from pathlib import Path
from os import environ


# These are the paths to where SageMaker mounts interesting things in your container.
ML_PREFIX = Path(environ["ML_PREFIX"])

INPUT_PATH = ML_PREFIX / 'input/data'
OUTPUT_PATH = ML_PREFIX / 'output'
MODEL_PATH = ML_PREFIX / 'model'
PARAM_PATH = ML_PREFIX / 'input/config/hyperparameters.json'
SECRETS_PATH = ML_PREFIX / 'input/config/secrets.json'
URL_DATA_PATH = INPUT_PATH / 'urls.gz'
PROJ_PREFIX = Path(environ["PROJ_PREFIX"])

SERVICES_PATH = PROJ_PREFIX / 'services'

# TODO: specify variables and functions for use-case

# LABEL_INDEX = {'NÃO REATIVARÁ' : 0, 'REATIVARÁ' : 1}
