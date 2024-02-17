import os
import csv
import numpy as np
import pandas as pd

from sklearn.utils.class_weight import compute_class_weight
from keras.callbacks import *

from utils.loss import *
from utils.metrics import *

from models.resnet import ResNet
from models.unet import UNet
from utils.dataview import *

from keras.optimizers import Adam, SGD
# from keras.optimizers.legacy import Adam, SGD

from utils.dataset import DatasetHandler

import keras_tuner
from keras_tuner import BayesianOptimization, RandomSearch, Hyperband

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate, BatchNormalization, Activation, Add, GlobalAveragePooling2D, Dense, Flatten, Dropout
from models.resUnet import ResUnet


class GridExperiment:
    def __init__(self, base_path, output_mode = (False, True)):
        self.name      = None           # Nome del singolo esperimento
        self.i_exp     = None
        self.grid_path = None

        self.output_mode = output_mode  # modalità dell'esperimento, se si vuole visualizzare o salvare (verbose,save)

        self.base_path   = base_path    # Path del progetto
        self.dataset_dir = None         # Path del dataset

        self.seed, self.n_class, self.img_shape = (
            None,
            None,
            None,
        )
        self.exp_config = None
        self.task = None

        self.dataset = None
        self.x_train, self.x_val, self.x_test, = (
            None,
            None,
            None
        )
        self.y_train, self.y_val, self.y_test, = (
            None,
            None,
            None
        )
        self.train_class_weights = None

        self.workers   = 1          # fit parameter
        self.max_qsize = 100

    def get_exp_name(self):
        if 'name' in self.exp_config:
            self.name = self.exp_config['name']
        else:
            weights  = self.exp_config.get('weights', None)
            model_name = "{}_{}".format(
                    self.task,
                    self.exp_config['backbone']
                )
            if weights is not None:
                model_name = model_name + "_" + weights
                
            self.name = "{}_{}_{}_BS{}_OPT{}".format(
                self.i_exp,
                model_name,
                self.exp_config['tuner'],
                self.exp_config['batch_size'],
                self.exp_config['optimizer']
                )

    # def build(self, setting, exp_config):
    def build(self, setting, exp_config, i_exp):
        self.seed, self.n_class, self.img_shape = (
            setting['SEED'],
            setting['N_CLASS'],
            setting['IMG_SHAPE'],
        )
        self.exp_config = exp_config
        self.task = exp_config['task']
        self.i_exp = i_exp

        self.get_exp_name()

        print('\n')
        print("-"*90)
        print(f">> BETA: GRID SEARCH {self.name} <<")
        print("-"*90)

        self.dataset_dir = os.path.join(self.base_path, 'dataset/TFRecords')
        self.grid_path = os.path.join(self.base_path, "grid_result", self.name)

        os.makedirs(self.grid_path, exist_ok=True)
        
        if self.dataset is None:
            self.load_dataset()

    def load_dataset(self):
        # shuffle_bsize = 2154
        shuffle_bsize = 100

        self.dataset = DatasetHandler(self.dataset_dir,
                                      shuffle_bsize = shuffle_bsize,
                                      random_state = self.seed,
                                      task = self.task)
        self.dataset.build()
        print(f'>> Dataset for {self.task} Loaded')

    def split_dataset(self, set_view = None):
        # gather the needed settings and data
        split_ratio, batch_size, augmentation = (
                self.exp_config['split_ratio'],
                self.exp_config['batch_size'],
                self.exp_config['augmentation'],
            )
        
        # split dataset
        self.dataset.split_dataset(split_ratio)

        # prepare sets
        self.x_train, self.y_train = self.dataset.prepare_tfrset('train')
        self.x_val, self.y_val = self.dataset.prepare_tfrset('val')
        self.x_test, self.y_test = self.dataset.prepare_tfrset('test')
        
        # generate sets
        # create the train, (val) and test sets to feed the neural networks
        self.x_train = self.dataset.generate_tfrset(self.x_train,
                                                    batch_size=batch_size,
                                                    shuffle=True,
                                                    augment=augmentation)
        self.x_val  = self.dataset.generate_tfrset(self.x_val, batch_size=batch_size) 
        self.x_test = self.dataset.generate_tfrset(self.x_test, batch_size=batch_size)

        print('>> Dataset Splitted')

        self.compute_class_weight()

        if set_view is not None:
            print('>> Print Batches:')
            plot_set_batches(self, set=set_view, num_batches=10)

    def generate_split_charts(self, charts=None):
        # default graphs
        charts = charts or ["pdistr", "lsdistr_pie"]
        
        # get the output mode
        display, save = self.output_mode
        
        plot_charts(self, charts, display, save, self.grid_path)

        print('>> Split Charts Generated')

    def compute_class_weight(self):
        # calculate class balance using 'compute_class_weight'
        train_class_weights = compute_class_weight('balanced', classes=np.unique(self.y_train), y=self.y_train)
        train_class_weights = np.round(train_class_weights, 4)

        self.train_class_weights = dict(enumerate(train_class_weights))

        print('>> Class Weights Computed')
    
    def _build_grid_model(self, hp):
        def get_metric_loss():
            task_mapping = {
                'segmentation': {
                    'lossFunc': dice_coef_loss,
                    'lossWeights': 1,
                    'metrics': ['accuracy', dice_coef]
                },
                'classification': {
                    'lossFunc': categorical_crossentropy,
                    'lossWeights': 1,
                    'metrics': 'accuracy'
                }
            }

            if self.task in task_mapping:
                task_settings = task_mapping[self.task]
                lossFunc = task_settings.get('lossFunc', None)
                lossWeights = task_settings.get('lossWeights', None)
                metrics = task_settings.get('metrics', None)
            else:
                raise ValueError(f"[ERROR] Unknown task: {self.task}")
            
            return [lossFunc, lossWeights, metrics]
        weights = self.exp_config.get('weights', None)
        model = self.exp_config['backbone']

        if weights == 'imagenet':
            ltf = hp.Choice('ltf' , values=['none', 'enc_2', 'all'])
        else:
            ltf = hp.Choice('ltf' , values=['none'])

        if self.task == 'segmentation':
            model = UNet(
                input_shape     = (224,224,3),
                model           = model,
                weights         = weights,
                layer_to_freeze = ltf,
                num_class       = 1,
                dropout         = hp.Float('dropout', 0, 0.5, step=0.1),
                dil_rate        = hp.Int('dil_rate', min_value = 1, max_value = 3)
            )()
        elif self.task == 'classification':
            model = ResNet(
                input_shape     = (224,224,3),
                model           = model,
                weights         = weights,
                layer_to_freeze = ltf,
                num_class       = 4,
                dropout         = hp.Float('dropout', 0, 0.5, step=0.1),
                dil_rate        = hp.Int('dil_rate', min_value = 1, max_value = 3)
            )()

        opt = self.exp_config['optimizer']
        if opt == 'Adam':
            lr = hp.Choice("learning_rate", values=[1e-3, 1e-4, 1e-5])   # ADAM
            opt = Adam(learning_rate=lr)
        elif opt == 'SGD':
            lr = hp.Choice("learning_rate", values=[1e-1, 1e-2, 1e-3])   # SGD
            opt = SGD(learning_rate=lr)

        loss_f, loss_w, metrics = get_metric_loss()
        model.compile(optimizer    = opt,
                      loss         = loss_f,
                      loss_weights = loss_w,
                      metrics      = metrics)

        return model
    
    def _build_grid_multitask_model(self, hp):        
        weights = self.exp_config.get('weights', None)
        backbone = self.exp_config['backbone']

        # if weights == 'imagenet':
        #     ltf = hp.Choice('ltf' , values=['none', 'enc_2', 'all'])
        # else:
        #     ltf = hp.Choice('ltf' , values=['none'])
        ltf = 'none'
        dropout         = hp.Float('dropout_seg', 0.0, 0.5, step=0.1)
        dropout = 0.0
        dil_rate        = hp.Int('dil_rate', min_value = 1, max_value = 3)

        seg_model = ResUnet(
            input_shape     = (224,224,3),
            model           = backbone,
            weights         = weights,
            layer_to_freeze = ltf,
            num_class       = 1, 
            dropout         = dropout,
            dil_rate        = dil_rate)()
        
        dropout_cls = hp.Float('dropout_cls', 0, 0.5, step=0.1)
        cls_model = self.classification_model(seg_model, dropout_cls)

        output = [cls_model.output, seg_model.output]  # Combine outputs.
        model  = Model(inputs=[seg_model.input], outputs=output, name=f'multitask_{backbone}')

        opt = self.exp_config['optimizer']
        if opt == 'Adam':
            lr = hp.Choice("learning_rate", values=[1e-3, 1e-4, 1e-5])   # ADAM
            opt = Adam(learning_rate=lr)
        elif opt == 'SGD':
            lr = hp.Choice("learning_rate", values=[1e-1, 1e-2, 1e-3])   # SGD
            opt = SGD(learning_rate=lr)

    
        gamma  = hp.Float('gamma', 0, 1, step=0.2)

        loss_f = {'cls_label': categorical_crossentropy, 'seg_mask': dice_coef_loss}
        loss_w = {'cls_label': gamma, 'seg_mask': (1-gamma)}
        metrics = {'cls_label': 'accuracy', 'seg_mask': ['accuracy', dice_coef]}
        
        model.compile(optimizer    = opt,
                      loss         = loss_f,
                      loss_weights = loss_w,
                      metrics      = metrics)

        return model
    
    def classification_model(self, base_model, dropout):
        '''Classificator model'''
        conv4_layer       = base_model.get_layer("enc_4")
        bottleneck_layer  = base_model.get_layer("bottleneck")
        dec4_layer        = base_model.get_layer("dec_4")
        
        conv4      = conv4_layer.output
        bottleneck = bottleneck_layer.output
        dec4       = dec4_layer.output
        
        xcon = GlobalAveragePooling2D(keepdims = False)(conv4)
        xbot = GlobalAveragePooling2D(keepdims = False)(bottleneck)
        xdec = GlobalAveragePooling2D(keepdims = False)(dec4)

        x = tf.concat([xcon, xbot, xdec], axis=-1)

        x = Dense(256, activation='relu')(x)
        if dropout > .0:
            x = Dropout(rate=dropout)(x)
        x = Dense(128, activation='relu')(x)
        if dropout > .0:
            x = Dropout(rate=dropout)(x)
            
        output = Dense(1, activation='softmax', name='cls_label')(x)

        return Model(inputs=base_model.input, outputs=output, name='classificator')
    
    def get_tuner(self):
        # obj = 'val_accuracy' if self.task == 'classification' else keras_tuner.Objective("val_dice_coef", direction="max")
        obj = keras_tuner.Objective("val_accuracy_cls", direction="max")
        
        grid_model = self._build_grid_multitask_model if self.task == 'multitask' else self._build_grid_model
        if self.exp_config['tuner'] == 'bayesian':
            tuner = BayesianOptimization(
                grid_model,
                objective = obj,
                max_trials = 10,
                seed = 42,
                directory = os.path.join(self.base_path, 'grid_result'),
                project_name = self.name
            )
        elif self.exp_config['tuner'] == 'randomsearch':
            tuner = RandomSearch(
                self._build_grid_model,
                objective = obj,
                max_trail = 5,
                execution_per_trail = 1,
                seed = 42,
                directory = os.path.join(self.base_path, 'grid_result'),
                project_name = self.name
            )
        elif self.exp_config['tuner'] == 'hyperband':
            tuner = Hyperband(
                self._build_grid_model,
                objective = obj,
                hyperband_iterations=1,
                seed = 42,
                directory = os.path.join(self.base_path, 'grid_result'),
                project_name = self.name
            )

        return tuner
    
    def run_tuner(self):
        tuner = self.get_tuner()

        batch_size, epochs = (
            self.exp_config['batch_size'],
            self.exp_config['epoch'],
        )
        train_steps = self.dataset.frame_counts['train'] // batch_size
        val_steps   = self.dataset.frame_counts['val'] // batch_size

        tuner.search(
            self.x_train,
            epochs              = epochs,
            steps_per_epoch     = train_steps,
            validation_data     = self.x_val,
            validation_steps    = val_steps,
            max_queue_size      = self.max_qsize,
            workers             = self.workers,
            use_multiprocessing = False,
            verbose = 5
        )

        # Ottieni i primi tre migliori set di iperparametri
        best_hps_list = tuner.get_best_hyperparameters(num_trials=3)
        best_trials   = tuner.oracle.get_best_trials(num_trials=3)
        best_scores = [trial.score for trial in best_trials]

        # Creazione di un elenco di dizionari, ciascuno contenente iperparametri di un modello
        hp_dicts = []
        for best_hps in best_hps_list:
            hp_dict = {
                #'ltf': best_hps.get('ltf'),
                # 'dropout_seg': best_hps.get('dropout_seg'),
                'dropout_cls': best_hps.get('dropout_cls'),
                'dil_rate': best_hps.get('dil_rate'),
                'learning_rate': best_hps.get('learning_rate'),
                'gamma': best_hps.get('gamma')
                # Aggiungi qui altri iperparametri se necessario
            }
            hp_dicts.append(hp_dict)

        # Crea un DataFrame pandas dai dizionari
        hp_df = pd.DataFrame(hp_dicts)
        hp_df['best_score'] = best_scores

        csv_path = os.path.join(self.grid_path, 'best_hyperparameters.csv')

        # Salva il DataFrame in un file CSV
        hp_df.to_csv(csv_path, index=False, sep=';', decimal='.')

        return tuner
