#!/bin/bash

# Define a função log_and_run com o nome do script e quaisquer argumentos extras
log_and_run() {  
    local script_name=$1  
    shift  # Remove o nome do script dos argumentos  
    echo -e "\n\n********************************************************************************************************************\n" >> log.log  
    echo "$ nohup /usr/bin/time -v python $script_name \"$@\" >> log.log 2>&1" >> log.log  
    nohup /usr/bin/time -v python "$script_name" "$@" >> log.log 2>&1  # Removed '&' to ensure sequential execution  
}  

# Chama a função log_and_run com todos os argumentos passados para o script
log_and_run "$@"

# Para parar a execução:
# $ ps aux | grep 'python main.py'
# O processo será exibido com 'user PID...'
# $ kill PID