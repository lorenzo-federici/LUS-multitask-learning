#!/bin/bash

# Path to your files
file_python="./src/setting_exp.py"
file_param="./config/params.json"

# Experiments folder
exp_folder="./exp"
grid_folder="./grid_result"

# Default value for idx_exp
default_idx_exp=0

# Check if at least one parameter is provided
if [ "$#" -ge 1 ]; then
    case "$1" in
        "run")
            # Check if idx_exp is specified, otherwise use the default value
            if [ "$#" -eq 1 ]; then
                idx_exp=$default_idx_exp
            else
                idx_exp="$2"
            fi

            # Execute the Python file with the specified or default parameters
            if [ -e "$file_python" ]; then
                python3 "$file_python" --exps_json "$file_param" --idx_exp "$idx_exp"
            else
                echo "ERROR: The Python file does not exist - $file_python"
            fi
            ;;
        "rm")
            # Check for additional options after "rm"
            case "$2" in
                "exp")
                    # Remove the experiments folder
                    if [ -d "$exp_folder" ]; then
                        rm -r "$exp_folder"
                        echo "Experiments folder removed successfully."
                    else
                        echo "The experiments folder does not exist."
                    fi
                    ;;
                "grid")
                    # Remove the grid_search folder
                    if [ -d "$grid_folder" ]; then
                        rm -r "$grid_folder"
                        echo "Grid search folder removed successfully."
                    else
                        echo "The grid search folder does not exist."
                        echo "$grid_folder"
                    fi
                    ;;
                *)
                    echo "ERROR: Invalid option after 'rm'. Use 'exp' or 'grid'."
                    ;;
            esac
            ;;
        "init")
            # Execute pip install -r requirements.txt
            pip install -r requirements.txt
            ;;
        *)
            echo "ERROR: Invalid option. Use 'run', 'rm', or 'init'."
            ;;
    esac
else
    echo "ERROR: Provide at least one option ('run', 'rm', or 'init')."
fi

