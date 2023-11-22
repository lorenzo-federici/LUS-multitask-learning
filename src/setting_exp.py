
import os
import sys
import json
import time
import argparse
from pathlib import Path
import tensorflow as tf
from keras import backend as K

BASE_PATH = str(Path(__file__).resolve().parent.parent)
if BASE_PATH not in sys.path:
    sys.path.append(BASE_PATH)

# import custom lib
from utils.dataset import *
from experiment import Experiment

class SettingExp:
    """ 
        Set of experiments that can be executed sequentially 
        Output mode: (Bool, Bool) -> Displaying, Saving
    """
    def __init__(self, base_path, json_path, idx_exp, output_mode=(False,True)):
        self.json_path   = json_path
        self.base_path   = base_path
        self.idx_exp     = idx_exp
        self.output_mode = output_mode

    def _generate_experiments(self):
        # Load JSON file
        with open(self.json_path) as f:
            configs = json.load(f)

        configs_general = configs['SETTING']
        configs_exps    = configs['EXPS'][self.idx_exp]

        experiment = Experiment(self.base_path, output_mode=self.output_mode)

        for config in configs_exps:
            experiment.build_experiment(configs_general, config)
            experiment.split_dataset()
            experiment.generate_split_charts('pierclass')
            experiment.generate_split_charts('ldistr')
            yield experiment
    
    def run(self):
        """ Execute the experiments """
        # Generate info for esperiment
        for experiment in self._generate_experiments():
            t_start = time.time()
            model   = experiment.build_model()
            model   = experiment.compile_model(model)
            history = experiment.train_model(model)
            experiment.nn_train_graphs(history)
            
            _ = experiment.evaluation_model(model)

            t_end = time.time()
            print('\n')
            print(f'Train time: {(t_end - t_start)/60}')
            print("-"*90)
            print("\t\t\t\t>> END EXPERIMENT <<")
            print("-"*90)

            K.clear_session()

def _config_accelleration():
    # Set the device to GPU 
    gpu = len(tf.config.list_physical_devices('GPU'))>0
    if gpu:
        tf.config.set_visible_devices([tf.config.list_physical_devices('GPU')[0],tf.config.list_physical_devices('CPU')[0]])
        tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True)
        print("\n---> GPU is available <---")

def clear_terminal():
    """Cleaning cross-platform Terminal"""
    os.system('cls' if os.name == 'nt' else 'clear')

def main():
    # clear_terminal()
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

    parser = argparse.ArgumentParser(description="LUS Ordinal Classification")
    parser.add_argument("--exps_json", type=str, required=True, help="json file containing the experiments to be performed")
    parser.add_argument("--idx_exp", type=int, required=True, help="experiment index on json")
    args = parser.parse_args()

    _config_accelleration()

    setting_exp = SettingExp(BASE_PATH, args.exps_json, args.idx_exp, output_mode=(False,True))
    setting_exp.run()

if __name__ == "__main__":
    main()