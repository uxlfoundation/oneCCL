#!/bin/bash

BASENAME=`basename $0 .sh`

cmd_timeout=600

echo_log()
{
    echo -e "$*"
}

print_help()
{
    echo_log ""
    echo_log "This example demonstrates launching of processes and creation of CCL communicator"
    echo_log "without regular mpirun launcher. KVS address is exchanged between processes using file system."
    echo_log "During execution KVS and communicator will be created and destroyed multiple times."
    echo_log ""
    echo_log "Usage:"
    echo_log "    ./${BASENAME}.sh [options]"
    echo_log ""
    echo_log "<options>:"
    echo_log "    -v  Path to oneCCL variables script"
    echo_log "    -h  Path to hostfile, one host per line"
    echo_log "    -s  Total number of ranks"
    echo_log ""
    echo_log "Example:"
    echo_log "    ./${BASENAME}.sh -v <vars_dir>/setvars.sh -h <hostfile_dir>/hostfile -s 4"
    echo_log ""
}

parse_arguments()
{
    if [ $# -ne 6 ];
    then
        print_help
        exit 1
    fi

    read_count=0

    while [ $# -ne 0 ]
    do
        case $1 in
            "-v"|"--vars")
                VARS=$2
                read_count=$((read_count+1))
                ;;
            "-h"|"--hostfile")
                HOSTFILE=$2
                read_count=$((read_count+1))
                ;;
            "-s"|"--size")
                SIZE=$2
                read_count=$((read_count+1))
                ;;
            *)
                echo_log "ERROR: unknown option ($1)"
                print_help
                exit 1
                ;;
        esac

        shift
        shift
    done

    if [ ${read_count} -ne 3 ];
    then
        echo_log "ERROR: unexpected number of read options ($read_count), expected 3"
        print_help
        exit 1
    fi

    if [ ! -f $VARS ];
    then
        echo_log "ERROR: can not find vars file: ${VARS}"
        exit 1
    fi

    if [ ! -f $HOSTFILE ];
    then
        echo_log "ERROR: can not find hostfile: ${HOSTFILE}"
        exit 1
    fi

    if [[ $VARS == *"setvars.sh"* ]];
    then
        echo "Use standalone CCL variables script"
    elif [[ $VARS == *"vars.sh"* ]];
    then
        echo "Use oneAPI CCL variables script"
        if [ -z "${I_MPI_ROOT}" ];
        then
            echo_log "ERROR: I_MPI_ROOT was not set"
            if [ -z "${IMPI_PATH}" ];
            then
                echo_log "ERROR: IMPI_PATH was not set"
                exit 1
            fi
        fi
    else
        echo_log "ERROR: unknown CCL variables script"
        exit 1
    fi

    unique_host_count=( `cat $HOSTFILE | grep -v ^$ | uniq | wc -l` )
    host_count=( `cat $HOSTFILE | grep -v ^$ | wc -l` )

    if [ "${unique_host_count}" != "${host_count}" ];
    then
        echo_log "ERROR: hostfile should contain unique hostnames"
        exit 1
    fi

    if [ "${host_count}" -eq "0" ];
    then
        echo_log "ERROR: hostfile should contain at least one row"
        exit 1
    fi

    echo_log "-----------------------------------------------------------"
    echo_log "PARAMETERS"
    echo_log "-----------------------------------------------------------"
    echo_log "VARS     = ${VARS}"
    echo_log "HOSTFILE = ${HOSTFILE}"
    echo_log "SIZE     = ${SIZE}"
    echo_log "-----------------------------------------------------------"
}

run_cmd()
{
    host="$1"
    cmd="$2"
    timeout_prefix="$3"

    if [[ "${host}" == "localhost" ]]
    then
        eval ${timeout_prefix} $cmd&
    else
        ${timeout_prefix} ssh ${host} $cmd&
    fi
}

cleanup_hosts()
{
    hostlist=$1

    echo "clean up hosts"
    for host in "${hostlist[@]}"
    do
        echo "host ${host}"
        cmd="killall -9 external_launcher run_binary.sh"
        run_cmd ${host} "${cmd}"
    done
}

run_binary()
{
    kvs_mode="$1"

    dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

    mapfile -t hostlist < $HOSTFILE
    host_count=( `cat $HOSTFILE | grep -v ^$ | wc -l` )

    local_size=$((SIZE / host_count))
    echo "local_size: ${local_size}"

    declare -a log_files=()

    cleanup_hosts $hostlist

    if [ "$kvs_mode" == "store" ]
    then
        kvs_param="$dir/store"
        if [ -f $kvs_param ]
        then
            rm $kvs_param
        fi
    elif [ "$kvs_mode" == "ip_port" ]
    then
        cmd="hostname -I | sed -e 's/\s.*$//'"
        kvs_param=`run_cmd ${hostlist[0]} "${cmd}"`
    fi

    host_idx=0

    for host in "${hostlist[@]}"
    do
        echo "start ranks on host: $host"
        for ((i = 0; i < $local_size ; i++ ));
        do
            rank=$((host_idx * local_size + i))
            log_file="${dir}/${host}_${rank}_${1}.out"
            log_files=("${log_files[@]}" "${log_file}")

            cmd="$dir/run_binary.sh -s ${SIZE} -r ${rank} -ls ${local_size} -lr ${i}"
            cmd="${cmd} -cclv ${VARS} -lf ${log_file} -km ${kvs_mode} -kp ${kvs_param}"
            if [[ -z ${I_MPI_ROOT} ]]
            then
                cmd="${cmd} -mv ${IMPI_PATH}/env/vars.sh"
            else
                cmd="${cmd} -mv ${I_MPI_ROOT}/env/vars.sh"
            fi
            if [[ ! -z "${SYCL_BUNDLE_ROOT}" && "${ENABLE_CODECOV}" = "yes" ]];
            then
                cmd="${cmd} -cv ${SYCL_BUNDLE_ROOT}/env/vars.sh"
            fi
            timeout_prefix="timeout -k $((cmd_timeout))s $((cmd_timeout))s"
            run_cmd ${host} "${cmd}" "${timeout_prefix}"
        done
        host_idx=$((host_idx + 1))
    done

    echo "wait completion"
    wait

    echo "check results"
    for file in "${log_files[@]}"
    do
        echo "check: $file"
        proc_count=`fuser $file | wc -w`
        while [ "${proc_count}" != "0" ]
        do
            sleep 1
            proc_count=`fuser $file | wc -w`
        done
        pass_count=`cat $file | grep "PASSED" | wc -l`
        if [ "${pass_count}" != "1" ]
        then
            echo -e "${RED}FAIL: expected 1 pass, got ${pass_count}${NC}"
        else
            echo -e "${GRN}PASSED${NC}"
        fi
    done

    cleanup_hosts $hostlist

    if [ "$kvs_mode" == "store" ]
    then
        if [ -f $kvs_param ]
        then
            rm $kvs_param
        fi
    fi
}

run()
{
    RED='\033[0;31m'
    GRN='\033[0;32m'
    PUR='\033[0;35m'
    NC='\033[0m'

    # kvs_modes="ip_port store"
    kvs_modes="ip_port"

    for mode in $kvs_modes
    do
        echo -e "${PUR}START EXAMPLE${NC}"
        exec_time=`date +%s`

        run_binary $mode

        exec_time="$((`date +%s`-$exec_time))"
        if [ "$exec_time" -ge "$cmd_timeout" ];
        then
             echo -e "${RED}FAILED: Timeout ($exec_time > $cmd_timeout)${NC}"
             exit 1
        fi
    done
}

parse_arguments $@
run
