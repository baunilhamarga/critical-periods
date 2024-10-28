#!/bin/bash

# Define the log_and_run function with an argument for the script name
log_and_run() {
    local script_name=$1
    shift
    echo -e "\n\n********************************************************************************************************************\n" >> log.log
    echo "$ nohup /usr/bin/time -v python $script_name >> log.log 2>&1 &" >> log.log
    nohup /usr/bin/time -v python $script_name >> log.log 2>&1 &
}

log_and_run "$@"

# To stop run
# $ ps aux | grep 'python main.py'
# Process will be shown with 'user PID...'
# $ kill PID