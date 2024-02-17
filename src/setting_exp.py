
import os
import gc
import json
import time
import shutil
from pathlib import Path
from keras import backend as K
from datetime import datetime, date

# import custom lib
from utils.dataset import *
from experiment import Experiment
from grid_search import GridExperiment

class SettingExp:
    """ 
        Set of experiments that can be executed sequentially 
        Output mode: (Bool, Bool) -> Displaying, Saving
    """
    def __init__(self, base_path, json_path, idx_exp, isGrid = False, output_mode=(False,True)):
        self.json_path   = json_path
        self.base_path   = base_path
        self.idx_exp     = idx_exp
        self.output_mode = output_mode
        self.eval_path   = os.path.join( base_path, 'models/weight')
        self.exp_path    = None
        self.isGrid      = isGrid     # tipologia degli esperimenti, se esperimenti normali oppure grid search

    def _generate_experiments(self):
        # Load JSON file
        with open(self.json_path) as f:
            configs = json.load(f)

        configs_general = configs['SETTING']
        
        if self.isGrid:
            json_key = 'GRID'
            experiment = GridExperiment(self.base_path, output_mode=self.output_mode)
        else: 
            json_key = 'EXPS'
            # Create main experiment directory
            today     = date.today().strftime("%d/%m/%Y").split('/')
            time      = datetime.now().strftime("%d/%m/%Y %H:%M:%S")[-8:].split(":")
            exp_name  = f"{today[0]}{today[1]}_{time[0]}{time[1]}"
            self.exp_path = os.path.join(self.base_path, 'exp', exp_name)
            eval_result_path = os.path.join(self.exp_path, 'eval_result')
            os.makedirs(self.exp_path, exist_ok=True)
            os.makedirs(eval_result_path, exist_ok=True)

            experiment = Experiment(self.base_path, exp_name, output_mode=self.output_mode)

        configs_exps = configs[json_key][self.idx_exp]

        for i, config in enumerate(configs_exps):
            experiment.build(configs_general, config, i+1)
            experiment.split_dataset()
            experiment.generate_split_charts()
            yield experiment
    
    def run(self):
        for experiment in self._generate_experiments():
            if self.isGrid:
                tuner = experiment.run_tuner()
                del tuner
            else:
                t_start = time.time()
                model   = experiment.build_model()
                model   = experiment.compile_model(model)
                history = experiment.train_model(model)
                experiment.get_train_graphs(history)

                _ = experiment.evaluation_model(model)
                
                t_end  = time.time()
                t_time = "{:.2f}".format((t_end - t_start)/60)

                print('\n')
                print(f'Experiment time: {t_time}m')
                del model, history
                
            print("-"*90)
            print("\t\t\t\t>> END EXPERIMENT <<")
            print("-"*90)  

            K.clear_session()
            gc.collect()  # garbage collector