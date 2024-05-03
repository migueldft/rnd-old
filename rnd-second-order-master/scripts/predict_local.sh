#!/bin/bash

payload=$1
content=${2}

#curl --header "Content-Type: ${content}" --request POST --data-binary @${payload} https://mkt-reactivation.priv.qa.dafiti.local/invocations -k
curl \
    --header "Content-Type: ${content}" \
    --request POST \
    --data-binary @${payload} \
    http://localhost:8080/invocations
