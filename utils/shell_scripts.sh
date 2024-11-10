#!/bin/bash

# Define the log_and_run function with an argument for the script name
log_and_run() {
    local script_name=$1
    local log_dir="./logs"
    local base_log_file="${log_dir}/${script_name%.*}"
    local i=1

    # Create logs directory if it doesn't exist
    mkdir -p "$log_dir"

    # Find the next available log file name
    while [[ -e "${base_log_file}_${i}.log" ]]; do
        ((i++))
    done

    local log_file="${base_log_file}_${i}.log"
    shift  # Remove the script name from the list of arguments

    # Print the current date and time at the start of the log file
    echo "Log started at: $(date)" > "$log_file"
    echo -e "\n********************************************************************************************************************\n" >> "$log_file"
    echo "$ nohup /usr/bin/time -v python $script_name $@ >> $log_file 2>&1 &" >> "$log_file"

    # Run the command with nohup and append output to the log
    nohup /usr/bin/time -v python "$script_name" "$@" >> "$log_file" 2>&1 &
}

log_and_run "$@"

# To stop run
# $ ps aux | grep 'python main.py'
# Process will be shown with 'user PID...'
# $ kill PID
