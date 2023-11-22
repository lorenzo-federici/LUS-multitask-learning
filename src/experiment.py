
import os
import sys
import keras
import numpy as np
import pandas as pd
from keras import backend as K
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

from utils.loss import *
from utils.metrics import *
from utils.dataset import *
from utils.callbacks import CustomCallbacks

from models.network import Network
from utils.utilities import *

from datetime import datetime, date
from keras.optimizers.legacy import Adam, SGD
from tensorflow.keras.saving import load_model

class Experiment:
    def __init__(self, base_path, output_mode = (False, True)):
        self.name = None
        self.output_mode = output_mode # (verbose,save)
        self.base_path = base_path

        self.seed, self.n_class, self.img_shape = (
            None,
            None,
            None,
        )
        self.exp_config = None
        self.exp_path   = None
        self.dataset, self.ds_infos, self.train_ds, self.val_ds, self.test_ds, self.class_weight_dicts = (
            None,
            None,
            None,
            None,
            None,
            None
        )
        self.y_train_ds, self.y_val_ds, self.y_test_ds = (
            None,
            None,
            None
        )
        self.train_subset_idxs, self.val_subset_idxs, self.test_subset_idxs = (
            None,
            None,
            None
        )

    def build_experiment(self, setting, exp_config):
        self.seed, self.n_class, self.img_shape = (
            setting['SEED'],
            setting['N_CLASS'],
            setting['IMG_SHAPE'],
        )
        self.exp_config = exp_config

        self.get_exp_name()
        self.exp_path = self.base_path + '/exp/' + self.name
        if not os.path.exists(self.exp_path):
            os.makedirs(self.exp_path)
            os.makedirs(self.exp_path + '/fig')
        self.load_dataset()

    def get_exp_name(self):
        if 'name' in self.exp_config:
            name = self.exp_config['name']
        else:
            name = "exp_{}_{}_BS{}_EP{}_OPT{}_LR{}_AUG{}".format(
                self.exp_config['task'],
                self.exp_config['backbone'],
                self.exp_config['batch_size'],
                self.exp_config['epoch'],
                self.exp_config['optimizer'],
                self.exp_config['lr'],
                self.exp_config['augmentation'],
                )
        today     = date.today().strftime("%d/%m/%Y").split('/')
        time      = datetime.now().strftime("%d/%m/%Y %H:%M:%S")[-8:].split(":")
        date_name = f"{today[0]}{today[1]}_{time[0]}{time[1]}"

        self.name = f"{name}__DT{date_name}"
        print('\n')
        print("-"*90)
        print(f">> EXPERIMENT: {self.name} <<")
        print("-"*90)
        
    def load_dataset(self):
        if self.dataset is None:
            dataset_h5 = os.path.join(self.base_path, 'data/iclus', 'dataset.h5')
            pkl_framesmap = os.path.join(self.base_path, 'data/iclus', 'hdf5_frame_index_map.pkl')
            self.dataset = RichHDF5Dataset(dataset_h5, pkl_framesmap)

    def calculate_class_weights(self):
        ds_labels = self.ds_infos['labels']

        # Extract the labels for the current subset
        self.y_train_ds = np.array(ds_labels)[self.train_subset_idxs]
        self.y_val_ds   = np.array(ds_labels)[self.val_subset_idxs]
        self.y_test_ds  = np.array(ds_labels)[self.test_subset_idxs]

        # Calculate class balance using 'compute_class_weight'
        class_weights_train = compute_class_weight('balanced', classes=np.unique(self.y_train_ds), y=self.y_train_ds)
        class_weights_val   = compute_class_weight('balanced', classes=np.unique(self.y_val_ds), y=self.y_val_ds)
        class_weights_test  = compute_class_weight('balanced', classes=np.unique(self.y_test_ds), y=self.y_test_ds)

        # Create a dictionary that maps classes to their weights
        class_weight_dict_train = dict(enumerate(class_weights_train))
        class_weight_dict_val   = dict(enumerate(class_weights_val))
        class_weight_dict_test  = dict(enumerate(class_weights_test))

        train_idxs_p = round((len(self.train_subset_idxs) / len(self.dataset)) * 100)
        val_idxs_p = round((len(self.val_subset_idxs) / len(self.dataset)) * 100)
        test_idxs_p = 100 - (train_idxs_p + val_idxs_p)

        print('\n>> SPLITTING:')
        print(f">>> Dataset Split: Train={len(self.train_subset_idxs)}({train_idxs_p}%), Val={len(self.val_subset_idxs)}({val_idxs_p}%), Test={len(self.test_subset_idxs)}({test_idxs_p}%)")
        print(f">>> Train Class Weights: {class_weight_dict_train}")
        print(f">>> Val Class Weights: {class_weight_dict_val}")
        print(f">>> Test Class Weights: {class_weight_dict_test}")

        self.class_weight_dicts = [class_weight_dict_train, class_weight_dict_val, class_weight_dict_test]

    def split_dataset(self):
        if self.dataset is None:
            self.load_dataset()

        split_ratios = self.exp_config['split_ratio']
        batch_size, task, augmentation = (
            self.exp_config['batch_size'],
            self.exp_config['task'],
            self.exp_config['augmentation'],
        )

        pkl_centersdict_path = os.path.join(
            self.base_path, 'data/iclus', 'hospitals-patients-dict.pkl'
        )

        _data_split_strategy = split_strategy(
            self.dataset,
            ratios=split_ratios,
            pkl_file=pkl_centersdict_path,
            rseed=self.seed,
        )
        
        if self.train_subset_idxs == None:
            self.train_subset_idxs, self.val_subset_idxs, self.test_subset_idxs, self.ds_infos = _data_split_strategy
        
        self.train_ds = TFDataset(self.dataset, self.train_subset_idxs, batch_size, task, is_train=augmentation).as_iterator()
        self.val_ds   = TFDataset(self.dataset, self.val_subset_idxs, batch_size, task, is_train=False).as_iterator()
        self.test_ds  = TFDataset(self.dataset, self.test_subset_idxs, batch_size, task, is_train=False).as_iterator()

        self.calculate_class_weights()

    def build_model(self):
        model_obj = Network(self.exp_config, num_classes_cls = self.n_class)
        model = model_obj.build_model()
        return model
    
    def compile_model(self, model):
        def get_metric_loss():
            task = self.exp_config['task']
            
            task_mapping = {
                'multitask': {
                    'lossFunc': {'cls_label': weighted_categorical_crossentropy(list(self.class_weight_dicts[0].values())), 'seg_mask': dice_coef_loss},
                    'lossWeights': {'cls_label': 0.5, 'seg_mask': 1},
                    'metrics': {'cls_label': 'accuracy', 'seg_mask': [dice_coef, iou, tversky]}
                },
                'segmentation': {
                    'lossFunc': dice_coef_loss,
                    'lossWeights': 1,
                    'metrics': [dice_coef, iou, tversky]
                },
                'classification': {
                    # 'lossFunc': weighted_categorical_crossentropy(list(self.class_weight_dicts[0].values())),
                    'lossFunc': categorical_crossentropy,
                    'lossWeights': 1,
                    'metrics': 'accuracy'
                }
            }

            if task in task_mapping:
                task_settings = task_mapping[task]
                lossFunc = task_settings.get('lossFunc', None)
                lossWeights = task_settings.get('lossWeights', None)
                metrics = task_settings.get('metrics', None)
            else:
                raise ValueError(f"[ERROR] Unknown task: {task}")
            
            return [lossFunc, lossWeights, metrics]

        lr  = self.exp_config['lr']
        opt = self.exp_config['optimizer']
        
        optimizer_map = {
            'Adam': Adam(learning_rate=lr),
            'SGD': SGD(learning_rate=lr),
        }
        
        if opt in optimizer_map:
            optimizer = optimizer_map[opt]
        else:
            raise ValueError(f"[ERROR] Unknown optimizer: {opt}")

        loss_f, loss_w, metrics = get_metric_loss()

        model.compile(optimizer    = optimizer,
                      loss         = loss_f,
                      loss_weights = loss_w,
                      metrics      = metrics)
        
        return model
    
    def train_model(self, model):
        if self.train_ds is None:
            print('>> ERROR Dataset not uploaded')
            return None

        task = self.exp_config['task']

        callbacks = CustomCallbacks(self.exp_path, self.exp_config['backbone'], task).get_list_callbacks()

        print('\n')
        print("-"*60)
        print('\t\t   ~~~~~~ TRAINING ~~~~~')
        print("-"*60)

        # steps
        batch_size = 16
        train_samples = len(self.train_subset_idxs)
        train_steps = train_samples // batch_size
        if train_samples % batch_size != 0:
            train_steps += 1

        val_samples = len(self.val_subset_idxs)
        val_steps = val_samples // batch_size
        if val_samples % batch_size != 0:
            val_steps += 1

        if task == 'classification':
            history = model.fit(
                self.train_ds,
                epochs           = self.exp_config['epoch'],
                steps_per_epoch  = train_steps,
                validation_data  = self.val_ds,
                class_weight     = self.class_weight_dicts[0],
                validation_steps = val_steps,
                callbacks        = callbacks,
                workers = 8
            )
        else:
            history = model.fit(
                self.train_ds,
                shuffle         = True,
                epochs          = self.exp_config['epoch'],
                validation_data = self.val_ds,
                callbacks       = callbacks,
            )

        return history
    
    def evaluation_model(self, model):
        model_path = f"{self.exp_path}/checkpoint/"

        if os.path.exists(model_path):
            # model = load_model(model_path, compile=False)
            print("Loading best weights")
            model.load_weights(model_path)
        
        model = self.compile_model(model)

        batch_size = self.exp_config['batch_size']
        test_samples = len(self.test_subset_idxs)
        test_steps = test_samples // batch_size
        if test_samples % batch_size != 0:
            test_steps += 1

        print('\n')
        print("-"*60)
        print('\t\t   ~~~~~ EVALUATION ~~~~')
        print("-"*60)

        # EVALUATION
        evaluate = model.evaluate(
            self.test_ds,
            batch_size=batch_size,
            # sample_weight=self.class_weight_dicts[2],
            steps=test_steps,
            workers=1,
            use_multiprocessing=False,
            return_dict=True
        )

        df = pd.DataFrame([evaluate])
        df.to_csv(f'{self.exp_path}/eval_results.csv', index=False)
        
        print('\n')
        print("-"*60)
        print('\t\t   ~~~~~ PREDICTION ~~~~')
        print("-"*60)

        # PREDICT & CONFUSION_MTRX
        test_prediction = model.predict(
            self.test_ds,
            batch_size=self.exp_config['batch_size'],
            steps=test_steps,
            callbacks=None,
            workers=1,
            use_multiprocessing=False
        )

        score_test = self.y_test_ds
        pred_score = tf.argmax(test_prediction, axis=-1)
        cf_matrix_test = confusion_matrix(score_test, pred_score, normalize='true', labels=[list(range(self.n_class))])

        ax = sns.heatmap(cf_matrix_test, linewidths=1, annot=True, fmt='.2f', cmap="BuPu")
        ax.set_ylabel('Actual')
        ax.set_xlabel('Predicted')
        ax.set_title('Test Set Confusion Matrix')

        display, save = self.output_mode

        if save:
            save_path = self.exp_path
            train_graphs_path = os.path.join(save_path, f"confusion_matrix.png")
            plt.savefig(train_graphs_path)

        # Show the figure
        if display:
            plt.show()
        
        plt.clf()
        plt.close()

        return evaluate
    
    # Plot function -----------------------------------------------------------------------------
    def generate_split_charts(self, charts=None):
        if self.ds_infos is not None:
            if charts is None:
                charts = ["fdistr", "pdistr", "ldistr"]
            
            # get the output mode
            display_mode = self.output_mode

            # choose the right save path (global of per-experiment)
            save_path = self.exp_path + "/fig/"
            print(save_path)
   
            if "pierclass" in charts:
                chart_file_path = os.path.join(save_path, "split_pie_per_class.png")
                print(chart_file_path)
                plot_fdistr_per_class_pie(self.y_train_ds, self.y_val_ds, self.y_test_ds, chart_file_path,output_mode=display_mode)
             
            if "splitinfo" in charts:
                print_split_ds_info(self.ds_infos)
            
            if "fdistr" in charts:
                chart_file_path = os.path.join(save_path, "split_per_frames.png")
                plot_frames_split(self.ds_infos, chart_file_path, log_scale=True, output_mode=display_mode)

            if "pdistr" in charts:
                chart_file_path = os.path.join(save_path, "split_per_patients.png")
                plot_patients_split(self.ds_infos, chart_file_path, output_mode=display_mode)

            if "ldistr" in charts:
                chart_file_path = os.path.join(save_path, "distr_labels_per_set.png")
                plot_labels_distr(self.y_train_ds, self.y_val_ds, self.y_test_ds, chart_file_path, output_mode=display_mode)
        else:
            raise Exception('dataset not yet splitted.')

    def nn_train_graphs(self, history):
        def _plot_histo(history, keys):
            nkey = len(keys)
            if nkey == 1:
                fig, ax = plt.subplots(1, nkey, figsize=(6, 6))
                ax.plot(history.history[keys[0]], label=keys[0])
                if not keys[0] == 'lr':
                    val_key = ('val_' + keys[0])
                    ax.plot(history.history[val_key], label=val_key, linestyle='--')

                ax.legend()
                ax.set_xlabel('epoch')
                ax.set_title(f'{keys[0]}')
                ax.grid()
                fig.suptitle(self.name)
            else:
                fig, ax = plt.subplots(1, nkey, figsize=(12, 4))
                for i in range(nkey):
                    ax[i].plot(history.history[keys[i]], label=keys[i])
                    val_key = ('val_' + keys[i])
                    ax[i].plot(history.history[val_key], label=val_key, linestyle='--')

                    ax[i].legend()
                    ax[i].set_xlabel('epoch')
                    ax[i].set_title(f'{keys[i]}')
                    ax[i].grid()
                fig.suptitle(self.name)
            
            display, save = self.output_mode

            if save:
                save_path = self.exp_path + "/fig/"
                train_graphs_path = os.path.join(save_path, f"train_graphs{keys[0]}.png")
                plt.savefig(train_graphs_path)

            # Show the figure
            if display:
                plt.show()
            
            plt.clf()
            plt.close()

        # Get the loss and metrics from history
        history_keys = history.history.keys()

        loss_key = [k for k in history_keys if 'loss' in k and 'val' not in k]
        _plot_histo(history, loss_key)

        if self.exp_config['task'] == 'multitask':
            metrics_keys_seg = [key for key in history_keys if((key not in loss_key) and ('mask' in key) and ('val' not in key) and ('lr' not in key))]
            metrics_keys_cls = [key for key in history_keys if((key not in loss_key) and ('cls' in key) and ('val' not in key) and ('lr' not in key))]
            if len(metrics_keys_seg) > 0:
                _plot_histo(history, metrics_keys_seg)
            if len(metrics_keys_cls) > 0:
                _plot_histo(history, metrics_keys_cls)
        elif self.exp_config['task'] == 'classification':
            metrics_keys_seg = [key for key in history_keys if((key not in loss_key) and ('val' not in key) and ('lr' not in key))]
            _plot_histo(history, metrics_keys_seg)
        elif self.exp_config['task'] == 'segmentation':
            _plot_histo(history, ['accuracy'])

        if 'lr' in history_keys:
            _plot_histo(history, ['lr'])
