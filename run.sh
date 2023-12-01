#!/bin/bash

# Path to your files
file_python="./src/setting_exp.py"
file_param="./config/params.json"

# Experiments folder
exp_folder="./exp"
grid_folder="./grid_result"

# Default value for idx_exp
default_idx_exp=0

# Display help message
display_help() {
    echo "Usage: $0 <option>"
    echo "Options:"
    echo "  --run  : Launch the project. Requires two arguments:"
    echo "              1) Experiment index to run"
    echo "              2) Boolean indicating whether the experiment should be run in training or evaluation mode (default is training)"
    echo "  --clr  : Remove result folders. Requires one argument:"
    echo "              1) exp : Remove the folder containing all experiment results"
    echo "              2) grid: Remove the folder containing grid search results"
    echo "  --init : Launch the installation of dependencies from requirements.txt"
}

# Check if at least one parameter is provided
if [ "$#" -ge 1 ]; then
    case "$1" in
        "--run")
            # Check if idx_exp is specified, otherwise use the default value
            if [ "$#" -eq 1 ]; then
                idx_exp=$default_idx_exp
            else
                idx_exp="$2"
            fi

            # Execute the Python file with the specified or default parameters
            if [ -e "$file_python" ]; then
                if [ "$#" -ge 3 ] && [ "$3" == "isEval" ]; then
                    python3 "$file_python" --exps_json "$file_param" --idx_exp "$idx_exp" --isEval
                fi 
                python3 "$file_python" --exps_json "$file_param" --idx_exp "$idx_exp"
            else
                echo "ERROR: The Python file does not exist - $file_python"
            fi
            ;;
        "--clr")
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
                    echo "ERROR: Invalid option after '--clr'. Use 'exp' or 'grid'."
                    ;;
            esac
            ;;
        "--init")
            # Execute pip install -r requirements.txt
            pip install -r requirements.txt
            ;;
        "--help")
            display_help
            ;;
        *)
            echo "ERROR: Invalid option. Use '--run', '--clr', '--init', or '--help'."
            ;;
    esac
else
    echo "ERROR: Provide at least one option ('--run', '--clr', '--init', or '--help')."
fi

