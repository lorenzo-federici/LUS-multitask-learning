
import os
import sys
import json
import time
import argparse
import pandas as pd
from pathlib import Path
import tensorflow as tf
from keras import backend as K
from datetime import datetime, date
from keras import mixed_precision

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
        self.eval_path   = os.path.join( base_path, 'models/weight')
        self.exp_path    = None

    def _generate_experiments(self):
        # Load JSON file
        with open(self.json_path) as f:
            configs = json.load(f)

        configs_general = configs['SETTING']
        configs_exps    = configs['EXPS'][self.idx_exp]

        # Create main experiment directory
        today     = date.today().strftime("%d/%m/%Y").split('/')
        time      = datetime.now().strftime("%d/%m/%Y %H:%M:%S")[-8:].split(":")
        exp_name  = f"{today[0]}{today[1]}_{time[0]}{time[1]}"
        self.exp_path = os.path.join(self.base_path, 'exp', exp_name)
        
        if not os.path.exists(self.exp_path):
            os.makedirs(self.exp_path)

        experiment = Experiment(self.base_path, self.exp_path, output_mode=self.output_mode)

        for config in configs_exps:
            experiment.build(configs_general, config)
            experiment.split_dataset()
            experiment.generate_split_charts()
            yield experiment
    
    def run(self):
        """ Execute the experiments """
        # Generate info for esperiment
        dict_eval = {}
        eval_exp  = []

        for experiment in self._generate_experiments():
            t_start = time.time()
            model   = experiment.build_model()
            model   = experiment.compile_model(model)
            history = experiment.train_model(model)
            experiment.get_train_graphs(history)
            
            t_end  = time.time()
            t_time = "{:.2f}".format((t_end - t_start)/60)
            
            eval = experiment.evaluation_model(model)

            print('\n')
            print(f'Experiment time: {t_time}m')
            print("-"*90)
            print("\t\t\t\t>> END EXPERIMENT <<")
            print("-"*90)

            eval['time'] = t_time
            eval_exp.append(eval)

            K.clear_session()

        for key in eval_exp[0].keys():
            valori = [d[key] for d in eval_exp]
            dict_eval[key] = valori
        
        df = pd.DataFrame(dict_eval)
        df.to_csv(f'{self.exp_path}/eval_results.csv', index=False)

def _config_accelleration():
    # Set the device to GPU 
    cpu_device = tf.config.list_physical_devices('CPU')[0]
    gpu_devices = tf.config.list_physical_devices('GPU')

    if gpu_devices:
        device = gpu_devices[0]
        details = tf.config.experimental.get_device_details(device)
        gpu_name = details.get('device_name', 'Unknown')
        gpu_capability = float(".".join(map(str, details.get('compute_capability', '0'))))

        tf.config.set_visible_devices([device, cpu_device])
        tf.config.experimental.set_memory_growth(device, True)

        print(f'>> GPU {gpu_name} available (compute capability: {gpu_capability})')
        print(f'\t• Name: {gpu_name}')
        print(f'\t• Compute capability: {gpu_capability}')

def main():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
    K.clear_session()

    parser = argparse.ArgumentParser(description="LUS Ordinal Classification")
    parser.add_argument("--exps_json", type=str, required=True, help="json file containing the experiments to be performed")
    parser.add_argument("--idx_exp", type=int, required=True, help="experiment index on json")
    parser.add_argument("--isEval", action='store_true', help="Evaluation only or Training")
    
    args = parser.parse_args()

    print('\n')
    print("~"*50)
    _config_accelleration()
    print("~"*50)

    setting_exp = SettingExp(BASE_PATH, args.exps_json, args.idx_exp, output_mode=(False,True))

    setting_exp.run()

if __name__ == "__main__":
    main()