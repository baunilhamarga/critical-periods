#!/bin/bash

# Define the log_and_run function
log_and_run() {
    echo -e "\n\n********************************************************************************************************************\n" >> log.log
    echo "$ nohup /usr/bin/time -v python main.py >> log.log 2>&1 &" >> log.log
    nohup /usr/bin/time -v python main.py >> log.log 2>&1 &
}

log_and_run

# To stop run
# $ ps aux | grep 'python main.py'
# Process will be shown with 'user PID...'
# $ kill PID