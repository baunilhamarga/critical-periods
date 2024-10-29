#!/bin/bash

# Define the log_and_run function with an argument for the script name
log_and_run() {
    local script_name=$1
    local log_file="${script_name%.*}.log"  # Use the script name without the extension as the log file name
    shift  # Shift to remove the script name from the list of arguments
    echo -e "\n\n********************************************************************************************************************\n" >> "$log_file"
    echo "$ nohup /usr/bin/time -v python $script_name $@ >> $log_file 2>&1 &" >> "$log_file"
    nohup /usr/bin/time -v python "$script_name" "$@" >> "$log_file" 2>&1 &
}

log_and_run "$@"

# To stop run
# $ ps aux | grep 'python main.py'
# Process will be shown with 'user PID...'
# $ kill PID
