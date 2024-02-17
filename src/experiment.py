import os
import csv
import numpy as np
import pandas as pd

from sklearn.utils.class_weight import compute_class_weight
from keras.callbacks import *

from utils.loss import *
from utils.metrics import *


from models.network import Network
from utils.dataview import *

from utils.DisplayCallback import DisplayCallback

from keras.optimizers import Adam, SGD
# from keras.optimizers.legacy import Adam, SGD

from utils.dataset import DatasetHandler

class Experiment:
    def __init__(self, base_path, exps_name, output_mode = (False, True)):
        self.name      = None           # Nome del singolo esperimento
        self.exps_name = exps_name      # Nome della cartella del gruppo degli esperimenti
        self.i_exp     = None

        self.output_mode = output_mode  # modalità dell'esperimento, se si vuole visualizzare o salvare (verbose,save)

        self.base_path   = base_path    # Path del progetto
        self.exp_path    = None         # Path del singolo esperimento
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
                
            self.name = "exp_{}_BS{}_EP{}_OPT{}_LR{}_AUG{}_DO{}_DIL{}".format(
                model_name,
                self.exp_config['batch_size'],
                self.exp_config['epoch'],
                self.exp_config['optimizer'],
                self.exp_config['lr'],
                self.exp_config['augmentation'],
                self.exp_config.get('dropout', .0),
                self.exp_config.get('dil_rate', 1)
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
        print(f">> EXPERIMENT {i_exp}: {self.name} <<")
        print("-"*90)

        self.exp_path    = os.path.join(self.base_path, 'exp', self.exps_name, f"{i_exp}_{self.name}")
        self.dataset_dir = os.path.join(self.base_path, 'dataset/TFRecords')
        if self.exp_config.get('out_class_seg', 1) == 4:
            self.dataset_dir = self.dataset_dir + '-multiclass'

        os.makedirs(self.exp_path + '/fig', exist_ok=True)

        with open(self.exp_path + '/model_config.csv', mode='w', newline='') as file_csv:
            writer = csv.writer(file_csv)
            writer.writerow(self.exp_config.keys())
            writer.writerow(self.exp_config.values())
        
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

    def compute_class_weight(self):
        # calculate class balance using 'compute_class_weight'
        train_class_weights = compute_class_weight('balanced', classes=np.unique(self.y_train), y=self.y_train)
        train_class_weights = np.round(train_class_weights, 4)

        self.train_class_weights = dict(enumerate(train_class_weights))

        print('>> Class Weights Computed')
    
    def generate_split_charts(self, charts=None):
        # default graphs
        charts = charts or ["pdistr", "lsdistr_pie"]
        
        # get the output mode
        display, save = self.output_mode
        
        save_path = os.path.join(self.base_path, "exp/")
        plot_charts(self, charts, display, save, save_path)

        print('>> Split Charts Generated')

    def build_model(self):
        model_obj = Network(self.exp_config, num_classes_cls = self.n_class)
        model = model_obj.build_model()
        return model

    def compile_model(self, model):
        def get_metric_loss():
            gamma = 0.5
            task_mapping = {
                'multitask': {
                    'lossFunc': {'cls_label': categorical_focal_loss_with_fixed_weights(list(self.train_class_weights.values()), gamma=2.0), 'seg_mask': dice_coef_loss},
                    #'lossFunc': {'cls_label': categorical_crossentropy, 'seg_mask': dice_coef_loss},
                    #'lossWeights': {'cls_label': gamma, 'seg_mask': 1-gamma},
                    'lossWeights': {'cls_label': 1, 'seg_mask': 1},
                    'metrics': {'cls_label': 'accuracy', 'seg_mask': ['accuracy', dice_coef]}
                },
                'segmentation': {
                    'lossFunc': dice_coef_loss,
                    'lossWeights': 1,
                    'metrics': ['accuracy', dice_coef]
                },
                'classification': {
                    #'lossFunc': weighted_categorical_crossentropy(list(self.train_class_weights.values())),
                    #'lossFunc': categorical_crossentropy,categorical_focal_loss
                    #'lossFunc': categorical_focal_loss(gamma=2., alpha=.25),
                    'lossFunc': categorical_focal_loss_with_fixed_weights(list(self.train_class_weights.values()), gamma=2.0),
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

        if self.task == 'multitask':
            if self.i_exp == 1 or self.i_exp == 6 or self.i_exp == 11 or self.i_exp == 16:
                gamma = 0
            elif self.i_exp == 2 or self.i_exp == 7 or self.i_exp == 12 or self.i_exp == 17:
                gamma = 0.25
            elif self.i_exp == 3 or self.i_exp == 8 or self.i_exp == 13 or self.i_exp == 18:
                gamma = 0.5
            elif self.i_exp == 4 or self.i_exp == 9 or self.i_exp == 14 or self.i_exp == 19:
                gamma = 0.75
            elif self.i_exp == 5 or self.i_exp == 10 or self.i_exp == 15 or self.i_exp == 20:
                gamma = 1

            if self.i_exp == 21 or self.i_exp == 22 or self.i_exp == 23 or self.i_exp == 24:
                loss_w = {'cls_label': 1, 'seg_mask': 1}
            else:
                loss_w = {'cls_label': gamma, 'seg_mask': (1-gamma)}    

        print(loss_w)

        model.compile(optimizer    = optimizer,
                      loss         = loss_f,
                      loss_weights = loss_w,
                      metrics      = metrics)
        
        return model
    
    def train_model(self, model, gradcam_freq=5, status_bar=None):
        # parameters
        batch_size, epochs = (
                    self.exp_config['batch_size'],
                    self.exp_config['epoch']
                )

        ckpt_filename    = os.path.join(self.exp_path, "weights")
        bck_path         = os.path.join(self.exp_path, "backup")
        # tensorboard_path = os.path.join(self.base_path, "exp/", "logs/fit/", self.name)

        os.makedirs(ckpt_filename, exist_ok=True)
        os.makedirs(bck_path, exist_ok=True)

        ckpt_filename = os.path.join(ckpt_filename, "best_weights.h5")

        verbose, _ = self.output_mode

        # callbacks
        # tensorboard = TensorBoard(log_dir=tensorboard_path, histogram_freq=1)
        # backup = BackupAndRestore(backup_dir=bck_path)
        checkpoint = ModelCheckpoint(ckpt_filename, monitor='val_loss', save_weights_only=True, save_best_only=True, verbose=verbose)
        early_stop = EarlyStopping(monitor='val_loss', patience=15, verbose=verbose)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=1e-6, verbose=verbose)

        # build callbacks list
        callbacks = [checkpoint, early_stop, reduce_lr]
        
        # compute train and val steps per epoch
        train_steps = self.dataset.frame_counts['train'] // batch_size
        val_steps   = self.dataset.frame_counts['val'] // batch_size

        print('\n')
        print("-"*60)
        print('\t\t   ~~~~~~ TRAINING ~~~~~')
        print("-"*60)

        # neural network fit
        if self.task == 'classification':
            history = model.fit(
                self.x_train,
                epochs              = epochs,
                steps_per_epoch     = train_steps,
                validation_data     = self.x_val,
                class_weight        = self.train_class_weights,
                validation_steps    = val_steps,
                callbacks           = callbacks,
                max_queue_size      = self.max_qsize,
                workers             = self.workers,
                use_multiprocessing = False,
                # verbose             = verbose
                verbose = 1
            )
        else:
            display_call = DisplayCallback(self.task, self.base_path, self.exp_path, model, epoch_interval=gradcam_freq, output_mode=self.output_mode)
            callbacks.append(display_call)
            
            history = model.fit(
                self.x_train,
                epochs              = epochs,
                steps_per_epoch     = train_steps,
                validation_data     = self.x_val,
                validation_steps    = val_steps,
                callbacks           = callbacks,
                max_queue_size      = self.max_qsize,
                workers             = self.workers,
                use_multiprocessing = False,
                # verbose             = verbose
            )
        
        return history

    def get_train_graphs(self, history):
        # Get the loss and metrics from history
        history_keys = history.history.keys()
        task = self.exp_config['task']

        save_path = os.path.join(self.exp_path, "fig/")

        loss_key = [k for k in history_keys if 'loss' in k and 'val' not in k]
        plot_train_history(self.name, history, loss_key, save_path, self.output_mode)

        if task == 'multitask':
            metrics_keys_seg = [key for key in history_keys if((key not in loss_key) and ('mask' in key) and ('val' not in key) and ('lr' not in key))]
            metrics_keys_cls = [key for key in history_keys if((key not in loss_key) and ('cls' in key) and ('val' not in key) and ('lr' not in key))]
            if metrics_keys_seg:
                plot_train_history(self.name, history, metrics_keys_seg, save_path, self.output_mode)
            if metrics_keys_cls:
                plot_train_history(self.name, history, metrics_keys_cls, save_path, self.output_mode)
        elif task == 'segmentation':
            metrics_keys_seg = [key for key in history_keys if((key not in loss_key) and ('val' not in key) and ('lr' not in key))]
            plot_train_history(self.name, history, metrics_keys_seg, save_path, self.output_mode)
        elif task == 'classification':
            plot_train_history(self.name, history, ['accuracy'], save_path, self.output_mode)

        if 'lr' in history_keys:
            plot_train_history(self.name, history, ['lr'], save_path, self.output_mode)

    def evaluation_model(self, model):
        model_path = f"{self.exp_path}/weights/best_weights.h5"

        if os.path.exists(model_path):
            print("Loading best weights")
            model.load_weights(model_path)
        
        model = self.compile_model(model)

        batch_size = self.exp_config['batch_size']    
        test_steps = -(-self.dataset.frame_counts['test'] // batch_size)

        print('\n')
        print("-"*60)
        print('\t\t   ~~~~~ EVALUATION ~~~~')
        print("-"*60)

        # EVALUATION
        evaluate = model.evaluate(
            self.x_test,
            batch_size  = batch_size,
            steps       = test_steps,
            workers     = 1,
            return_dict = True,
            use_multiprocessing = False
            # sample_weight=self.class_weight_dicts[2],
        )

        df = pd.DataFrame([evaluate])
        df.to_csv(f'{self.base_path}/exp/{self.exps_name}/eval_result/{self.i_exp}-{self.name}.csv', index=False, sep=';', decimal=',')
        
        print('\n')
        print("-"*60)
        print('\t\t   ~~~~~ PREDICTION ~~~~')
        print("-"*60)

        # PREDICT & CONFUSION_MTRX
        y_pred = model.predict(
            self.x_test,
            batch_size = batch_size,
            steps      = test_steps,
            workers    = 1,
            use_multiprocessing = False
        )

        if self.task != 'classification':
            display_prediction_mask(self, y_pred)
            if len(y_pred[0].shape) > 1:
                y_pred = tf.argmax(y_pred[0], axis=-1)
            else:
                y_pred = y_pred[0]

        if self.task != 'segmentation':
            if len(y_pred.shape) > 1:
                y_pred = tf.argmax(y_pred, axis=-1)
            confusionmatrix(self, self.y_test, y_pred)

        return evaluate