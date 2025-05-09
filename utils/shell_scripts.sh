#!/bin/bash

log_and_run() {
    local log_dir="./logs"
    mkdir -p "$log_dir"

    # Search for the .py file in the arguments
    for arg in "$@"; do
        if [[ "$arg" == *.py ]]; then
            script_name="$arg"
            break
        fi
    done

    # If no .py file is found, exit with error
    if [[ -z "$script_name" ]]; then
        echo "Error: No .py script found in arguments."
        exit 1
    fi

    # Extract base name (remove path and extension)
    local base_script_name=$(basename "$script_name" .py)
    local base_log_file="${log_dir}/${base_script_name}"
    local i=1

    # Find the next available log file name
    while [[ -e "${base_log_file}_${i}.log" ]]; do
        ((i++))
    done

    local log_file="${base_log_file}_${i}.log"

    echo "Log started at: $(date)" > "$log_file"
    echo -e "\n********************************************************************************************************************\n" >> "$log_file"
    echo "$ nohup /usr/bin/time -v $@ >> $log_file 2>&1 &" >> "$log_file"

    # Run the command
    nohup /usr/bin/time -v "$@" >> "$log_file" 2>&1 &
}

log_and_run "$@"
