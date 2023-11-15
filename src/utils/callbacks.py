from keras.callbacks import *
from utils.utilities import *

class CustomCallbacks:
    def __init__(self, model_path, model, task):
        self.model_path = model_path
        self.model      = model
        self.task       = task

    def get_list_callbacks(self):
        class DisplayCallback(keras.callbacks.Callback):
            def __init__(self, task, epoch_interval=None):
                self.epoch_interval = epoch_interval
                self.task = task
                self.test_path = '/Users/lorenzofederici/Univpm/Tesi/LUS-multitask-learning/data/test/'

            def on_epoch_end(self, epoch, logs=None):
                if self.epoch_interval and epoch % self.epoch_interval == 0:
                    if self.task == 'multitask':
                        display_testing_images_multi_hm(self.test_path, self.model, epoch)
                    elif self.task == 'segmentation':
                        display_testing_images(self.test_path, self.model, epoch)

        reducerLR = ReduceLROnPlateau(monitor='val_loss',
                                      factor=0.1,
                                      patience=10,
                                      verbose=0,
                                      min_delta=1e-4,
                                      cooldown=2,
                                      min_lr=1e-6)
        checkpointer = ModelCheckpoint(filepath=f"{self.model_path}/{self.model}.keras", verbose=0, save_best_only=True)
        earlystopper = EarlyStopping(monitor='val_loss',
                                    mode='min',
                                    verbose=1,
                                    patience=20,
                                    restore_best_weights=True)

        # Tensorboard visualization
        # tensLog = TensorBoard(log_dir=f'{self.model_path}/log_dir')

        # Callback that streams epoch results to a CSV file.
        csv_file = f'{self.model_path}/{self.model}-training.csv'
        csvLog = CSVLogger(csv_file)

        callbacks = [
            DisplayCallback(self.task, 5),
            checkpointer,
            reducerLR,
            earlystopper,
            #   tensLog,
            csvLog,
        ]

        return callbacks
