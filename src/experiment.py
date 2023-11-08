
import os
import keras
import numpy as np
import matplotlib.pyplot as plt
from keras import backend as K
from sklearn.utils.class_weight import compute_class_weight
from keras.optimizers.legacy import Adam, SGD

from src.utils.loss import *
from src.utils.metrics import *
from src.utils.dataset import *
from src.utils.callbacks import CustomCallbacks

from models.network import Network


class Experiment:
    def __init__(self, setting, exp_config):
        self.name = None
        self.seed, self.n_class, self.img_shape, self.base_path = (
            setting['SEED'],
            setting['N_CLASS'],
            setting['IMG_SHAPE'],
            setting['BASE_PATH'],
        )
        self.exp_config = exp_config
        self.exp_path   = None
        self.dataset, self.ds_info, self.train_ds, self.val_ds, self.test_ds, self.class_weight_dicts = (
            None,
            None,
            None,
            None,
            None,
            None
        )

    def build_experiment(self):
        self.get_exp_name()
        self.exp_path = self.base_path + '/models/data/' + self.name
        if not os.path.exists(self.exp_path):
            os.makedirs(self.exp_path)
        self.load_dataset()

    def get_exp_name(self):
        if 'name' in self.exp_config:
            self.name = self.exp_config['name']
        else:
            self.name = "exp_{}_{}_BS{}_EP{}_OPT{}_LR{}_AUG{}".format(
                self.exp_config['task'],
                self.exp_config['backbone'],
                self.exp_config['batch_size'],
                self.exp_config['epoch'],
                self.exp_config['optimizer'],
                self.exp_config['lr'],
                self.exp_config['augmentation'],
                )
        print("-"*90)
        print(f"\t>> EXPERIMENT: {self.name} <<")
        print("-"*90)
        
    def load_dataset(self):
        if self.dataset is None:
            dataset_h5 = os.path.join(self.base_path, 'data', 'dataset.h5')
            pkl_framesmap = os.path.join(self.base_path, 'data', 'hdf5_frame_index_map.pkl')
            self.dataset = RichHDF5Dataset(dataset_h5, pkl_framesmap)

    def calculate_class_weights(self, subsets):
        class_weight_dicts = {}
        ds_labels = self.ds_info['labels']
        for i, subset in enumerate(subsets):
            # Extract the labels for the current subset
            y_subset_labels = np.array(ds_labels)[subset]

            # Calculate class balance using 'compute_class_weight'
            class_weights = compute_class_weight('balanced', classes=np.unique(y_subset_labels), y=y_subset_labels)

            # Create a dictionary that maps classes to their weights
            class_weight_dict = dict(enumerate(class_weights))

            class_weight_dicts[i] = class_weight_dict
            print(f">> Class Weights for Subset {i + 1}: {class_weight_dict}")
        
        self.class_weight_dicts = class_weight_dicts

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
            self.base_path, 'data', 'hospitals-patients-dict.pkl'
        )

        train_subset_idxs, val_subset_idxs, test_subset_idxs, self.ds_info = split_strategy(
            self.dataset,
            ratios=split_ratios,
            pkl_file=pkl_centersdict_path,
            rseed=self.seed,
        )

        self.train_ds = HDF5Dataset(self.dataset, train_subset_idxs, batch_size, task, augmentation)
        self.val_ds  = HDF5Dataset(self.dataset, val_subset_idxs, batch_size, task)
        self.test_ds = HDF5Dataset(self.dataset, test_subset_idxs, batch_size, task)

        self.calculate_class_weights([train_subset_idxs, val_subset_idxs, test_subset_idxs])

    def build_model(self):
        task = self.exp_config['task']
        backbone  = self.exp_config['backbone']
        model_obj = Network(task, backbone, num_classes_cls=self.n_class)
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
                    'lossFunc': weighted_categorical_crossentropy(list(self.class_weight_dicts[0].values())),
                    #'lossFunc': categorical_crossentropy,
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

        print("-"*60)
        print('\t\t   ~~~~~~ RUNNING ~~~~~~')
        print("-"*60)
        history = model.fit(
            self.train_ds,
            shuffle         = True,
            epochs          = self.exp_config['epoch'],
            validation_data = self.val_ds,
            callbacks       = callbacks,
        )

        return history

    def nn_train_graphs(self, history, show=True, save=False):
        def _plot_histo(history, keys):
            nkey = len(keys)
            if nkey == 1:
                _, ax = plt.subplots(1, nkey, figsize=(6, 6))
                ax.plot(history.history[keys[0]], label=keys[0])
                val_key = ('val_' + keys[0])
                ax.plot(history.history[val_key], label=val_key, linestyle='--')

                ax.legend()
                ax.set_xlabel('epoch')
                # ax.set_ylabel(f'{self.settings["loss"]}')
                ax.set_title(f'{keys[0]} - ' + self.name)
                ax.grid()
            else:
                _, ax = plt.subplots(1, nkey, figsize=(12, 4))
                for i in range(nkey):
                    ax[i].plot(history.history[keys[i]], label=keys[i])
                    val_key = ('val_' + keys[i])
                    ax[i].plot(history.history[val_key], label=val_key, linestyle='--')

                    ax[i].legend()
                    ax[i].set_xlabel('epoch')
                    # ax[i].set_ylabel(f'{self.settings["loss"]}')
                    ax[i].set_title(f'{keys[i]} - ' + self.name)
                    ax[i].grid()
        # Get the loss and metrics from history
        history_keys = history.history.keys()

        loss_key         = [k for k in history_keys if 'loss' in k and 'val' not in k] 
        metrics_keys_seg = [key for key in history_keys if((key not in loss_key) and ('mask' in key) and ('val' not in key))]
        metrics_keys_cls = [key for key in history_keys if((key not in loss_key) and ('cls' in key) and ('val' not in key))]

        _plot_histo(history, loss_key)
        _plot_histo(history, metrics_keys_seg)
        _plot_histo(history, metrics_keys_cls)
        
        # Show the figure
        if show:
            plt.show()

        if save:
            train_graphs_path = os.path.join('./', "train_graphs.png")
            plt.savefig(train_graphs_path)
        
        plt.close()
