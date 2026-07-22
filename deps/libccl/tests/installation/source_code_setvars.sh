#!/bin/bash

# remove duplicate "/"
simplify_path() {
    echo $1 | sed "s/\/\+/\//g"
}


INSTALL_DIRECTORY=$1
SOURCE_VARS="source $( simplify_path ${INSTALL_DIRECTORY}/env/vars.sh ) "
SOURCE_SETVARS="source $( simplify_path  ${INSTALL_DIRECTORY}/env/setvars.sh ) "

BUNDLED_MPI_YES="--ccl-bundled-mpi=yes"
BUNDLED_MPI_NO="--ccl-bundled-mpi=no"
BUNDLED_MPI_INVALID="--ccl-bundled-mpi=yas"

BUNDLED_MPI_YES_SINGLE="-ccl-bundled-mpi=yes"
BUNDLED_MPI_NO_SINGLE="-ccl-bundled-mpi=no"
BUNDLED_MPI_INVALID_SINGLE="-ccl-bundled-mpi=yas"

CHECK_MPI="which mpirun"
MPI_DEFAULT=$(which mpirun)
MPI_BUNDLED="$( simplify_path ${INSTALL_DIRECTORY}/opt/mpi/bin/mpirun )"

if [[ "${MPI_DEFAULT}" == "${MPI_BUNDLED}" ]] ; then
    echo "default mpi and sourced mpi are the same, should be different"
    echo "mpi tests should be run without sourcing the tested script"
    exit 1
fi

ERROR_MESSAGE_SETVARS_FLAG="ccl_bundled_mpi is only supported by direct call vars.sh, ignoring"

# execute command supplied in arguments in a subshell
# so that env variables in the current shell are not modified
execute_in_subshell_and_check() {
    to_execute=$1
    expected_substring=$2
    unexpected_substring=$3

    command="${to_execute} && ${CHECK_MPI}"
    output=$( bash -c "${command}" )
    if [[ ! "${output}" == *"${expected_substring}"* ]] ; then
        echo "expected part NOT found: ${expected_substring} in ${output}"
        exit 1
    fi
    if [[ "${output}" == *"${unexpected_substring}"* ]] ; then
        echo "unexpected part found: ${unexpected_substring} in ${output}"
        exit 1
    fi
}

# tests for bundled
execute_in_subshell_and_check "${SOURCE_SETVARS}" "${MPI_BUNDLED}" "${MPI_DEFAULT}" # setvars
execute_in_subshell_and_check "${SOURCE_VARS}" "${MPI_BUNDLED}" "${MPI_DEFAULT}"	# default
execute_in_subshell_and_check "${SOURCE_VARS} ${BUNDLED_MPI_YES}" "${MPI_BUNDLED}" "${MPI_DEFAULT}"	# explicit yes

# test for not bundled
execute_in_subshell_and_check "${SOURCE_VARS} ${BUNDLED_MPI_NO}" "${MPI_DEFAULT}" "${MPI_BUNDLED}" # explicit no

# execution errors - flag supplied to setvars
# check loaded mpi
execute_in_subshell_and_check "${SOURCE_SETVARS} ${BUNDLED_MPI_YES}" "${MPI_BUNDLED}" "${MPI_DEFAULT}"
execute_in_subshell_and_check "${SOURCE_SETVARS} ${BUNDLED_MPI_NO}" "${MPI_BUNDLED}" "${MPI_DEFAULT}"
execute_in_subshell_and_check "${SOURCE_SETVARS} ${BUNDLED_MPI_INVALID}" "${MPI_BUNDLED}" "${MPI_DEFAULT}"
# check error message
execute_in_subshell_and_check "${SOURCE_SETVARS} ${BUNDLED_MPI_YES}" "${ERROR_MESSAGE_SETVARS_FLAG}" "${MPI_DEFAULT}"
execute_in_subshell_and_check "${SOURCE_SETVARS} ${BUNDLED_MPI_NO}" "${ERROR_MESSAGE_SETVARS_FLAG}" "${MPI_DEFAULT}"
execute_in_subshell_and_check "${SOURCE_SETVARS} ${BUNDLED_MPI_INVALID}" "${ERROR_MESSAGE_SETVARS_FLAG}" "${MPI_DEFAULT}"

# execution errors - invalid flag
execute_in_subshell_and_check "${SOURCE_VARS} ${BUNDLED_MPI_INVALID}" "${MPI_BUNDLED}" "${MPI_DEFAULT}"
execute_in_subshell_and_check "${SOURCE_VARS} ${BUNDLED_MPI_INVALID}" "unrecognized" "${MPI_DEFAULT}"

# repeat tests with single dash

# tests for bundled
execute_in_subshell_and_check "${SOURCE_SETVARS}" "${MPI_BUNDLED}" "${MPI_DEFAULT}" # setvars
execute_in_subshell_and_check "${SOURCE_VARS}" "${MPI_BUNDLED}" "${MPI_DEFAULT}"	# default
execute_in_subshell_and_check "${SOURCE_VARS} ${BUNDLED_MPI_YES_SINGLE}" "${MPI_BUNDLED}" "${MPI_DEFAULT}"	# explicit yes

# test for not bundled
execute_in_subshell_and_check "${SOURCE_VARS} ${BUNDLED_MPI_NO_SINGLE}" "${MPI_DEFAULT}" "${MPI_BUNDLED}" # explicit no

# execution errors - flag supplied to setvars
# check loaded mpi
execute_in_subshell_and_check "${SOURCE_SETVARS} ${BUNDLED_MPI_YES_SINGLE}" "${MPI_BUNDLED}" "${MPI_DEFAULT}"
execute_in_subshell_and_check "${SOURCE_SETVARS} ${BUNDLED_MPI_NO_SINGLE}" "${MPI_BUNDLED}" "${MPI_DEFAULT}"
execute_in_subshell_and_check "${SOURCE_SETVARS} ${BUNDLED_MPI_INVALID_SINGLE}" "${MPI_BUNDLED}" "${MPI_DEFAULT}"
# check error message
execute_in_subshell_and_check "${SOURCE_SETVARS} ${BUNDLED_MPI_YES_SINGLE}" "${ERROR_MESSAGE_SETVARS_FLAG}" "${MPI_DEFAULT}"
execute_in_subshell_and_check "${SOURCE_SETVARS} ${BUNDLED_MPI_NO_SINGLE}" "${ERROR_MESSAGE_SETVARS_FLAG}" "${MPI_DEFAULT}"
execute_in_subshell_and_check "${SOURCE_SETVARS} ${BUNDLED_MPI_INVALID_SINGLE}" "${ERROR_MESSAGE_SETVARS_FLAG}" "${MPI_DEFAULT}"

# execution errors - invalid flag
execute_in_subshell_and_check "${SOURCE_VARS} ${BUNDLED_MPI_INVALID_SINGLE}" "${MPI_BUNDLED}" "${MPI_DEFAULT}"
execute_in_subshell_and_check "${SOURCE_VARS} ${BUNDLED_MPI_INVALID_SINGLE}" "unrecognized" "${MPI_DEFAULT}"

echo "Pass"
