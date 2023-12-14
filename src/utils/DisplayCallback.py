import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
import keras
import tensorflow as tf

import matplotlib

class DisplayCallback(keras.callbacks.Callback):
    def __init__(self, task, base_path, exp_path, model, epoch_interval=None, output_mode = (False, True)):
        self.epoch_interval = epoch_interval
        self.task = task
        self.data_path = os.path.join(base_path, 'dataset/test/')
        self.model = model
        self.output_mode = output_mode
        self.exp_path = exp_path

    def on_epoch_end(self, epoch, logs=None):
        if self.epoch_interval and epoch % self.epoch_interval == 0:
            self.display_testing_images(epoch)

    def display_testing_images(self, epoch = None):
        '''Display test image for multitask callbacks'''
        pickle_files = os.listdir(self.data_path)

        num_rows    = len(pickle_files)
        num_columns = 4
        text_epoch  = f'epoch: {epoch:03d}' if epoch is not None else ''

        _, axes = plt.subplots(num_rows, num_columns, figsize=(10, 10))
        plt.suptitle(text_epoch, fontsize=16)
        for i, pickle_file in enumerate(pickle_files):
            with open(self.data_path + pickle_file, 'rb') as file:
                img, [lbl, mask] = pickle.load(file)

            img_reshaped = tf.reshape(img, (1,224,224,3))
            prediction   = self.model.predict(img_reshaped)

            axes[i, 0].imshow(img, cmap='gray')
            axes[i, 0].set_title(f"Test Image {i+1}")
            if self.task == 'multitask':
                lbl_true = tf.argmax(lbl, axis=-1)
                lbl_pred = tf.argmax(prediction[0], axis=-1)
                text  = f'class: {lbl_true} --> Pred: {lbl_pred}'
                color = 'green' if (int(lbl_true) == int(lbl_pred)) else 'red'
                axes[i, 0].text(20, 200, text, fontsize=10, color=color)
                prediction = prediction[1]

            axes[i, 1].imshow(img)
            axes[i, 1].imshow(mask, cmap='jet', alpha=0.2)
            axes[i, 1].set_title("True Mask")

            heatmap      = self.make_gradcam_heatmap(img_reshaped, 0)
            img_heatmap  = self.merge_gradcam(img, heatmap)
            axes[i, 2].imshow(img_heatmap)
            axes[i, 2].set_title("Heatmap")

            axes[i, 3].imshow(prediction[0])
            axes[i, 3].set_title("Predicted Mask")

        # Hide excess axes
        for ax in axes.ravel():
            ax.axis('off')

        display, save = self.output_mode

        if save:
            os.makedirs(self.exp_path + '/fig/callback', exist_ok=True)
            train_callback_path = os.path.join(self.exp_path, 'fig/callback' , f"epoch_{epoch}.png")
            plt.savefig(train_callback_path)

        # Show the figure
        if display:
            plt.tight_layout()
            plt.show()
            plt.close()
    
    def make_gradcam_heatmap(self, img_array, pred_index=None):
        '''Display heatmap of prediction'''

        # auto-search the last convolutional layer of the model (uselful for GRAD-cams)
        for layer in reversed(self.model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                last_conv_layer_name = layer.name
                break

        last_conv_layer_name = 'dec_1'

        grad_model = keras.models.Model(
            self.model.inputs, [self.model.get_layer(last_conv_layer_name).output, self.model.output]
        )

        with tf.GradientTape() as tape:
            last_conv_layer_output, preds = grad_model(img_array)
            if self.task == 'multitask':
                _, preds = preds
            if pred_index is None:
                pred_index = tf.argmax(preds[0])
            class_channel = preds[:, pred_index]

        grads = tape.gradient(class_channel, last_conv_layer_output)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        last_conv_layer_output = last_conv_layer_output[0]
        heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        return heatmap.numpy()
    
    def merge_gradcam(self, img, heatmap, alpha=0.4):
        img = keras.utils.img_to_array(img)
        heatmap = np.uint8(255 * np.nan_to_num(heatmap, nan=0))

        jet = matplotlib.colormaps.get_cmap("jet")
        jet_colors = jet(np.arange(256))[:, :3]
        jet_heatmap = jet_colors[heatmap]

        jet_heatmap = keras.utils.array_to_img(jet_heatmap)
        jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
        jet_heatmap = keras.utils.img_to_array(jet_heatmap)
        
        jet_heatmap = jet_heatmap / 255
        superimposed_img = jet_heatmap * alpha + img
        superimposed_img = keras.utils.array_to_img(superimposed_img)

        return superimposed_img