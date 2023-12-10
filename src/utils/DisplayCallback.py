import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
import keras
import tensorflow as tf

import matplotlib

class DisplayCallback(keras.callbacks.Callback):
    def __init__(self, task, base_path, model, epoch_interval=None):
        self.epoch_interval = epoch_interval
        self.task = task
        self.data_path = os.path.join(base_path, 'dataset/test/')
        self.model = model

    def on_epoch_end(self, epoch, logs=None):
        if self.epoch_interval and epoch % self.epoch_interval == 0:
            if self.task == 'multitask':
                # display_testing_images_multi_hm(epoch)
                print('DIOCA')
            elif self.task == 'segmentation':
                self.display_testing_images(epoch)

    def display_testing_images(self, epoch = None):
        '''Display test image for callbacks'''
        pickle_files = os.listdir(self.data_path)

        num_rows = len(pickle_files)
        num_columns = 4
        text_epoch = f'epoch: {epoch:03d}' if epoch is not None else ''
        _, axes = plt.subplots(num_rows, num_columns, figsize=(10, 10))
        plt.suptitle(text_epoch, fontsize=16)
        for i, pickle_file in enumerate(pickle_files):
            with open(self.data_path + pickle_file, 'rb') as file:
                img, [_, mask] = pickle.load(file)

            axes[i, 0].imshow(img, cmap='gray')
            axes[i, 0].set_title(f"Test Image {i+1}")

            axes[i, 1].imshow(img)
            axes[i, 1].imshow(mask, cmap='jet', alpha=0.2)
            axes[i, 1].set_title("True Mask")

            img_reshaped = tf.reshape(img, (1,224,224,3))
            heatmap      = self.make_gradcam_heatmap(img_reshaped, 0)
            img_heatmap  = self.merge_gradcam(img, heatmap)
            axes[i, 2].imshow(img_heatmap)
            axes[i, 2].set_title("Heatmap")

            pred_masks = self.model.predict(img_reshaped)
            axes[i, 3].imshow(pred_masks[0])
            axes[i, 3].set_title("Predicted Mask")

        # Hide excess axes
        for ax in axes.ravel():
            ax.axis('off')

        plt.tight_layout()
        plt.show()
    
    def make_gradcam_heatmap(self, img_array, pred_index=None):
        '''Display heatmap of prediction'''

        # auto-search the last convolutional layer of the model (uselful for GRAD-cams)
        for layer in reversed(self.model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                last_conv_layer_name = layer.name
                break

        grad_model = keras.models.Model(
            self.model.inputs, [self.model.get_layer(last_conv_layer_name).output, self.model.output]
        )

        with tf.GradientTape() as tape:
            last_conv_layer_output, preds = grad_model(img_array)
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