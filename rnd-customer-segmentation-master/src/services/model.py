from io import StringIO
import codecs, json
import sys
import traceback

from joblib import dump, load
import lightgbm as lgb
import pandas as pd
from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

import constants as constants
import utils import preproc


logger = set_up_logging(__name__)


# This algorithm has a single channel of input data called 'training'. Since we run in
# File mode, the input files are copied to the directory specified here.
CHANNEL_NAME = 'training'
TRAINING_PATH = constants.INPUT_PATH / CHANNEL_NAME


def get_label_id(category):
    return constants.LABEL_INDEX[category]


def get_class_label(prediction):
    for key, value in constants.LABEL_INDEX.items():
        if value == prediction:
            return key


def save_model(classifier):
    weights_file = constants.MODEL_PATH / 'model.joblib'
    dump(classifier, weights_file)
    logger.info(f'Saved model artifact at: {weights_file}')


def load_model(weights_file):
    return load(weights_file)


def predict(payload, clf):
    """For the input, do the predictions and return them.
    Args:
        input a list of instances:
        The data on which to do the predictions. """
    df = pd.read_csv(StringIO(payload), sep=',', header=None)

    df = preproc(df)

    ids   = list(df[constants.ID_COLUMN[0]])
    preds = list(clf.predict(df[constants.NUMERIC_FEATURES]))
    probs = list(clf.predict_proba(df))

    output = {"predictions": []}
    for id, pred, prob in zip(ids, preds, probs):
        output['predictions'].append(
            [
                id,
                float('{:.0f}'.format(pred)),
                float('{:.3f}'.format(prob[int(pred)]))
            ]
        )
    return output


def get_model_pipe(clf, **params):
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())])

    transformers = [('num', numeric_transformer, constants.NUMERIC_FEATURES)]

    input_prep = ColumnTransformer(transformers=transformers)

    steps = []
    steps.append(('input_prep', input_prep))
    steps.append(('classifier', clf(**params)))
    return Pipeline(steps=steps)


def train():
    """
    A sample training component that trains a simple scikit-learn model.
    Receives no arguments.
    expects 1 file: hyperparameters (json) @ constants.PARAM_PATH
    expects 2 outputs:
        model artifact (joblib) @ constants.MODEL_PATH / 'model.joblib'
        metrics file (json) @ constants.OUTPUT_PATH / 'metrics.json'
    """
    logger.info('Launching training.')
    try:
        # Read in any hyperparameters that the user passed with the training job
        with open(constants.PARAM_PATH, 'r') as tc:
            trainingParams = json.load(tc)

        class_weight = trainingParams.get("class_weight", None)
        if class_weight is not None:
            trainingParams["class_weight"] = {}
            for k, w in class_weight.items():
                trainingParams["class_weight"][int(k)] = w

        # Take the set of files and read them all into a single pandas dataframe
        input_files = [ TRAINING_PATH / file for file in TRAINING_PATH.iterdir() ]
        if len(input_files) == 0:
            raise ValueError(f'There are no files in {TRAINING_PATH}.\n' +
                              f'This usually indicates that the channel ({CHANNEL_NAME}) was incorrectly specified,\n' +
                              'the data specification in S3 was incorrectly specified or the role specified\n' +
                              'does not have permission to access the data.')
        raw_data = [ pd.read_csv(file, sep=',', header=None) for file in input_files ]
        raw_data = pd.concat(raw_data)
        train_data = preproc(raw_data, training=True)

        # labels are in the last column
        train_y = train_data.iloc[:,-1]
        # Ids are in the first column
        train_X = train_data.iloc[:,1:-1]

        # Now use scikit-learn's pipeline to train the model.
        clf = lgb.LGBMClassifier
        model = get_model_pipe(clf, **trainingParams)
        model.fit(train_X, train_y)

        # save metrics
        results = {"AUC":
            metrics.roc_auc_score(
                train_y, model.predict(train_X))}
        file_path = constants.OUTPUT_PATH / 'metrics.json'
        json.dump(
            results,
            codecs.open(file_path, 'w', encoding='utf-8'),
            separators=(',', ':'), sort_keys=True)

        # save the model
        save_model(model)       
        with open(constants.OUTPUT_PATH / 'success', 'w') as s:
            s.write('Done!')
    except Exception as e:
        # Write out an error file. This will be returned as the failureReason in the
        # DescribeTrainingJob result.
        trc = traceback.format_exc()
        with open(constants.OUTPUT_PATH / 'failure', 'w') as s:
            s.write('Exception during training: ' + str(e) + '\n' + trc)
        # Printing this causes the exception to be in the training job logs, as well.
        logger.info('Exception during training: ' + str(e) + '\n' + trc)
        # A non-zero exit code causes the training job to be marked as Failed.
        sys.exit(255)
