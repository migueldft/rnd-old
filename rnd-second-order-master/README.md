# Marketing Second Order Prediction

[![CircleCI](https://circleci.com/gh/dafiti-group/rnd-second-order/tree/master.svg?circle-token=0c2ba7701ce97658039506678e9563e8b377a9a6&style=shield)](https://circleci.com/gh/dafiti-group/rnd-second-order/tree/master)

This project is related to [RAD-114](https://dafiti.jira.com/browse/RAD-114).


## How to install this module

This repo is based on python 3.7.4, makefile, and docker.

To install docker please follow the steps here: https://docs.docker.com/install/linux/docker-ce/ubuntu/

### Training and predict

There are 2 ways of using this resource: through an AWS client, that will consume a sagemaker service or running a local docker container.

The first option will consume a cloud based service, thus you will need to setup an AWS client on your operational system:

### Running by consuming AWS service

Create a file `secrets/secrets.yml` with the following structure:

```
dbconn:
  host: "gfg-dwh-prod.dafiti.io"
  dbname: "dftdwh"
  port: "5439"
  user: "<your-user>"
  password: "<your-password>"
dump-conn:
  aws_access_key_id: "<your-access-key-to-dafiti-live>"
  aws_secret_access_key: "<your-secret-access-key-to-dafiti-live>"
  aws_session_token: "<your-session-token-to-dafiti-live>" # Not required
sagemaker-conn:
  aws_access_key_id: "<your-access-key-to-aws-account>"
  aws_secret_access_key: "<your-secret-access-key-to-aws-account>"
  aws_default_region: "us-east-2"
  aws_default_output: "json"
  aws_session_token: "<your-session-token-to-aws-account>" # Not required
```

Replace `<your-user>` and `<your-password>` with your respectively RedShift user and password. Replace all `<>` below `dump-conn` with access information for `Dafiti Live` AWS account. Also replace all `<>` below `sagemaker-conn` with access information for the AWS account, probably it's the `Research and Development` AWS account.

#### Training:

Execute `make raise-training-job` or execute `make raise-optmize`. Second option applys a Bayesian Optmization for hyper parameter tunning, although it takes around 5 days to finish.

#### Serving API and Predicting:

This funcionality uses a R&D docker registry at GCP, you will need to login on it before using it. Please contact R&D for this.
After obtaining a `<sevice-key-file>` with R&D team, you need to run the following command for login in:

```
cat <service-key-file> | docker login -u _json_key --password-stdin https://gcr.io
```


The container used are defined in the repositories below:
1. [redshift-adapter](https://github.com/dafiti-group/rnd-redshift-adapter)

Execute `make sage-predict`

This will generate a file `scripts/aws_user/ml/output/predictions.parquet` following the structure below:

```
        ids  90days_predictions  90days_confidence  ...  180days_confidence  30days_predictions  30days_confidence
0  32152984                   0              0.866  ...               0.800                   0              0.938
1  32219292                   0              0.900  ...               0.868                   0              0.900
2  28412464                   0              0.906  ...               0.881                   0              0.934
3  30232704                   0              0.889  ...               0.836                   0              0.937
4  27379452                   0              0.770  ...               0.719                   0              0.823
```


### Running locally:


Grant executable mode to the script entrypoints by running the commands below:

```
chmod +x second_order/serve
```
```
chmod +x second_order/train
```

#### Training:

Execute `make test-train`

#### Serving API:

Execute `make test-serve`

You should see somehting like this:
```
2019-10-08 21:24:58,096 - __main__ - INFO - Starting the inference server with 8 workers.
[2019-10-08 21:24:58 +0000] [18] [INFO] Starting gunicorn 19.9.0
[2019-10-08 21:24:58 +0000] [18] [INFO] Listening at: unix:/tmp/gunicorn.sock (18)
[2019-10-08 21:24:58 +0000] [18] [INFO] Using worker: gevent
[2019-10-08 21:24:58 +0000] [22] [INFO] Booting worker with pid: 22
...
2019-10-08 21:25:02,464 - api.app - INFO - Starting server.
...
2019-10-08 21:25:02,476 - utils.io - INFO - Loading models from /opt/ml/model
...
2019-10-08 21:25:02,478 - api.resources.prediction - INFO - Begin server with artifacts: ['180days', '30days', '90days', '15days']
```

To check if its really up, go to your browser and try: `http://localhost:8080/ping`

How to predict, indeed? There is a CSV file called `payload.csv` at `ml/input/data`. Use it to make a `POST` at:

```
curl --header "Content-Type: text/csv" --request POST \
--data-binary @payload.csv http://localhost:8080/invocations
```

Response must be a JSON like this:
```
{
	"predictions": {
		"ids" : [3492072, ..., 35758985],
		"30days_predictions" : [0.0, ..., 0.0],
		"30days_confidence" : [0.90, ..., 0.89],
		...
		"180days_predictions" : [0.0, ..., 1.0],
		"180days_confidence" : [0.917, ..., 0.908],
	}
}
```
This is standard input for `pandas.DataFrames.from_dict`.

#### Predicting:

Execute `make test-predict`


## Credits, doubts and known issues


## Maturity Model Evaluation

[![Infrastructure](https://img.shields.io/static/v1.svg?label=Infrastructure&message=Level%200&color=red)](https://github.com/dafiti-group/rnd-second-order/blob/master/MATURITY_MODEL.md)
[![Architecture](https://img.shields.io/static/v1.svg?label=Architecture&message=Level%200&color=red)](https://github.com/dafiti-group/rnd-second-order/blob/master/MATURITY_MODEL.md)
[![Security](https://img.shields.io/static/v1.svg?label=Security&message=Level%200&color=red)](https://github.com/dafiti-group/rnd-second-order/blob/master/MATURITY_MODEL.md)
[![Quality](https://img.shields.io/static/v1.svg?label=Quality&message=Level%200&color=red)](https://github.com/dafiti-group/rnd-second-order/blob/master/MATURITY_MODEL.md)
