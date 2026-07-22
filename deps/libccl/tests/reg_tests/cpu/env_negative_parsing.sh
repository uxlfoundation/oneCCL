#!/bin/bash

SCRIPT_DIR=$(cd $(dirname "${BASH_SOURCE}") && pwd -P)
ROOT_DIR="$(dirname "${SCRIPT_DIR}")"
BASENAME="$(basename $0 .sh)"
TEST_LOG="${BASENAME}.log"

source ${ROOT_DIR}/utils.sh

make_common_actions ${SCRIPT_DIR} ${TEST_LOG}

test="benchmark"

# command used to generate env_values_bool:
# echo $(egrep "^    bool " src/common/env/env.hpp | grep -v "was_printed" | awk '{print "CCL_"toupper($2)}' | sed "s/;//g" | head -n 6)

# currently, only sample API variables are tested
env_values_bool="CCL_ABORT_ON_THROW CCL_QUEUE_DUMP CCL_SCHED_DUMP CCL_SCHED_PROFILE CCL_WORKER_OFFLOAD CCL_WORKER_WAIT"
env_values_bool_incorrect="-1 false FALSE true TRUE 2 3 a A"
env_values_unknown="CCL_UNKNOWN_VARIABLE=1 CCL_VARIABLE_THAT_IS_NOT_KNOWN=6"

for env_value_bool in ${env_values_bool}
do
    for env_value_bool_incorrect in ${env_values_bool_incorrect}
    do

        cmd="${env_value_bool}=${env_value_bool_incorrect}"
        cmd+=" mpiexec -l -n 1"
        cmd+=" ${SCRIPT_DIR}/${test} > ${TEST_LOG} 2>&1"
        run_negative_cmd "${cmd}"
        expected_string="${env_value_bool}: unexpected value: ${env_value_bool_incorrect}, expected values: 0, 1"
        if [[ ! $(grep "${expected_string}" ${TEST_LOG}) ]]
        then
            echo "ERROR: the error string [${expected_string}] not found in output of a command that should cause the error" 1>&2 | tee -a ${TEST_LOG}
            exit 1
        fi
    done
done

for env_value_unknown in ${env_values_unknown}
do
    cmd="CCL_LOG_LEVEL=debug "
    cmd+="${env_value_unknown}"
    cmd+=" mpiexec -l -n 1"
    cmd+=" ${SCRIPT_DIR}/${test} > ${TEST_LOG} 2>&1"
    run_cmd "${cmd}"
    expected_string="${env_value_unknown} is unknown to and unused by oneCCL code but is present in the environment, check if it is not mistyped."
    if [[ ! $(grep "${expected_string}" ${TEST_LOG}) ]]
    then
        echo "ERROR: the error string [${expected_string}] not found in output of a command that should cause the error" 1>&2 | tee -a ${TEST_LOG}
        exit 1
    fi
done


for test in ${tests}
do
    rm ${test}
done
rm ${TEST_LOG}

echo "Pass"
