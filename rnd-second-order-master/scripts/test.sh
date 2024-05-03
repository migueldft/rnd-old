#!/usr/bin/env bash


OPTIND=1

BASE="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT=$(dirname ${BASE})
PROJECT=second_order

usage()
{
    echo "usage: <command> options:<-m>"
}

monitor=0

while getopts ":m" opt; do
    case "$opt" in
    m)
        monitor=1
        ;;
    *)
        usage
        exit 0
        ;;
    esac
done

shift $((OPTIND-1))

[ "$1" = "--" ] && shift

TEST_CMD="pytest '${ROOT}/${PROJECT}'"

for i in "$@"; do
    case "$i" in
        *\'*)
            i=`printf "%s" "$i" | sed "s/'/'\"'\"'/g"`
            ;;
        *) : ;;
    esac
    if [[ ${i} == -* ]];
    then
        TEST_CMD="$TEST_CMD $i"
    else
        TEST_CMD="$TEST_CMD \"$i\""
    fi
done

if [ ${monitor} -eq 1 ]
then
    TEST_CMD="watchmedo shell-command --patterns='*.py' --recursive --command='${TEST_CMD}'"
else
    TEST_CMD="${TEST_CMD}"
fi

if [ ${monitor} -eq 1 ]
then
    eval "${TEST_CMD} || exit 1"
    test_ret_code=$?
else
    eval "${TEST_CMD}"
    test_ret_code=$?
fi
cd ${BASE}

exit $test_ret_code