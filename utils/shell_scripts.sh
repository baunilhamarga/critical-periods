#!/bin/bash

# Define the log_and_run function
log_and_run() {
    echo -e "\n\n********************************************************************************************************************\n" >> log.log
    echo "$ nohup python main.py >> log.log 2>&1 &\n" >> log.log
    nohup python main.py >> log.log 2>&1 &
}

log_and_run
