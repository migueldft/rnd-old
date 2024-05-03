#!/bin/bash

bucket=719003640801-customer-reactivation
query_file=data_etl/query.sql
secrets_file=secrets/secrets.yml

CURRENT_UID=$(eval id -u)
query_string=$(cat ${query_file})
secrets=`cat ${secrets_file}`
TIME_STAMP=$(date +%Y-%m-%d --date='-1 year')

query=${query_string/date_replacement/$TIME_STAMP}

docker run --rm \
    -u ${CURRENT_UID}:${CURRENT_UID} \
    -v ${PWD}/ml/input/data:/opt/data \
    --network="host" \
    719003640801.dkr.ecr.us-east-2.amazonaws.com/redshift-adapter:latest \
        --query="${query}" \
        --secrets="${secrets}" \
        --remote_storage=s3://${bucket}/input/data/raw
