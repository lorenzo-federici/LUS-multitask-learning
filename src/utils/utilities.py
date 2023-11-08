import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
import keras
import tensorflow as tf

def display(display_list):
    """Function printing imgs"""
    plt.figure(figsize=(15, 15))

    title = ['Input Image', 'True Mask', 'Predicted Mask']

    len_input = len(display_list)    

    for i in range(len_input):
        plt.subplot(1, len(display_list) + 1, i + 1)
        plt.title(title[i])
        plt.imshow(display_list[i])
        plt.axis('off')

    if len_input == 2:
        plt.subplot(1, len(display_list) + 1, 3)
        plt.title(" Image + mask")
        plt.imshow(display_list[0])
        plt.imshow(display_list[1], cmap='jet', alpha=0.2)
        plt.axis('off')

    if len_input == 3:
        plt.subplot(1, len(display_list) + 1, 4)
        plt.title(" Image + mask")
        plt.imshow(display_list[1])
        plt.imshow(display_list[2], cmap='jet', alpha=0.2)
        plt.axis('off')

def print_split_diagnostic_info(ds_info):
    # Print diagnostic information
    for medical_center in ds_info['medical_center_patients'].keys():
        print(f"Medical Center: {medical_center}")
        print(f"  Frames in center: {ds_info['frames_by_center'][medical_center]}")
        print(f"  Train patients:")
        for patient in ds_info['train_patients_by_center'][medical_center]:
            frame_count = ds_info['frames_by_center_patient'][medical_center][patient]
            print(f"   {patient}: {frame_count} frames")
        print(f"  Val patients:")
        for patient in ds_info['val_patients_by_center'][medical_center]:
            frame_count = ds_info['frames_by_center_patient'][medical_center][patient]
            print(f"   {patient}: {frame_count} frames")
        print(f"  Test patients:")
        for patient in ds_info['test_patients_by_center'][medical_center]:
            frame_count = ds_info['frames_by_center_patient'][medical_center][patient]
            print(f"   {patient}: {frame_count} frames")

def plot_split_graphs(train_subset, val_subset, test_subset, ds_info):
    # Plot the distribution of frames per center
    frame_counts = ds_info['frames_by_center']
    centers = list(frame_counts.keys())
    frame_values = [frame_counts[center] for center in centers]

    plt.figure(figsize=(12, 6))
    plt.bar(centers, frame_values)
    plt.title('Distribution of Frames per Medical Center')
    plt.xlabel('Medical Center')
    plt.ylabel('Number of Frames')
    plt.yscale('log')
    plt.xticks(rotation=45, fontsize=8)
    plt.show()

    # Plot the distribution of patients per center
    train_patients = ds_info['train_patients_by_center']
    val_patients = ds_info['val_patients_by_center']
    test_patients = ds_info['test_patients_by_center']

    train_values = [len(train_patients[center]) for center in centers]
    val_values = [len(val_patients[center]) for center in centers]
    test_values = [len(test_patients[center]) for center in centers]

    width = 0.3
    x = range(len(centers))

    plt.figure(figsize=(12, 6))
    plt.bar(x, train_values, width, label='Train set', align='center')
    plt.bar([i + width for i in x], val_values, width, label='Val set', align='center')
    plt.bar([i + 2 * width for i in x], test_values, width, label='Test set', align='center')
    plt.title('Distribution of Patients per Medical Center')
    plt.xlabel('Medical Center')
    plt.ylabel('Number of Patients')
    plt.yscale('log')
    plt.xticks([i + width for i in x], centers, rotation=45, fontsize=8)
    plt.legend()
    plt.show()    

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    '''Display heatmap of prediction'''
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = keras.models.Model(
        model.inputs, [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def display_testing_images(data_path, model, epoch = None):
    '''Display test image for callbacks'''
    
    pickle_files = os.listdir(data_path)

    num_rows = len(pickle_files)
    num_columns = 4
    text_epoch = f'epoch: {epoch:03d}' if epoch is not None else ''

    _, axes = plt.subplots(num_rows, num_columns, figsize=(10, 10))
    for i, pickle_file in enumerate(pickle_files):
        with open(data_path + pickle_file, 'rb') as file:
            img, [_, mask] = pickle.load(file)

        axes[i, 0].imshow(img, cmap='gray')
        axes[i, 0].set_title(f"Image {text_epoch}")

        axes[i, 1].imshow(img)
        axes[i, 1].imshow(mask, cmap='jet', alpha=0.2)
        axes[i, 1].set_title("True Mask")

        img_reshaped = tf.reshape(img, (1,224,224,3))
        heatmap      = make_gradcam_heatmap(img_reshaped, model, 'dec_4', 0)
        axes[i, 2].imshow(heatmap)
        axes[i, 2].set_title("Heatmap")

        pred_masks = model.predict(img_reshaped)
        axes[i, 3].imshow(pred_masks[0])
        axes[i, 3].set_title("Predicted Mask")

    # Hide excess axes
    for ax in axes.ravel():
        ax.axis('off')

    plt.tight_layout()
    plt.show()

def display_testing_images_multi(data_path, model, epoch = None):
    '''Display test image for callbacks'''
    
    pickle_files = os.listdir(data_path)

    num_rows    = len(pickle_files)
    num_columns = 3 
    text_epoch  = f'epoch: {epoch:03d}' if epoch is not None else ''

    _, axes = plt.subplots(num_rows, num_columns, figsize=(10, 10))
    for i, pickle_file in enumerate(pickle_files):
        with open(data_path + pickle_file, 'rb') as file:
            img, [lbl, mask] = pickle.load(file)

        img_reshaped = tf.reshape(img, (1,224,224,3))
        prediction   = model.predict(img_reshaped)

        lbl_true = tf.argmax(lbl, axis=-1)
        lbl_pred = tf.argmax(prediction[0], axis=-1)

        axes[i, 0].imshow(img, cmap='gray')
        axes[i, 0].set_title(f"Image {text_epoch}")
        text  = f'class: {lbl_true} --> Predicted: {lbl_pred}'
        color = 'green' if (int(lbl_true) == int(lbl_pred)) else 'red'
        axes[i, 0].text(20, 200, text, fontsize=10, color=color)

        axes[i, 1].imshow(img)
        axes[i, 1].imshow(mask, cmap='jet', alpha=0.2)
        axes[i, 1].set_title("True Mask")

        axes[i, 2].imshow(prediction[1][0])
        axes[i, 2].set_title("Predicted Mask")

    # Hide excess axes
    for ax in axes.ravel():
        ax.axis('off')

    plt.tight_layout()
    plt.show()
    plt.close()




# -------------------- multi & heatmap
def make_gradcam_heatmap_multi(img_array, model, last_conv_layer_name, pred_index=None):
    '''Display heatmap of prediction'''
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = keras.models.Model(
        model.inputs, [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, [_, preds] = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def display_testing_images_multi_hm(data_path, model, epoch = None):
    '''Display test image for callbacks'''
    
    pickle_files = os.listdir(data_path)

    num_rows    = len(pickle_files)
    num_columns = 4 
    text_epoch  = f'epoch: {epoch:03d}' if epoch is not None else ''

    _, axes = plt.subplots(num_rows, num_columns, figsize=(10, 10))
    for i, pickle_file in enumerate(pickle_files):
        with open(data_path + pickle_file, 'rb') as file:
            img, [lbl, mask] = pickle.load(file)

        img_reshaped = tf.reshape(img, (1,224,224,3))
        prediction   = model.predict(img_reshaped)

        lbl_true = tf.argmax(lbl, axis=-1)
        lbl_pred = tf.argmax(prediction[0], axis=-1)

        axes[i, 0].imshow(img, cmap='gray')
        axes[i, 0].set_title(f"Image {text_epoch}")
        text  = f'class: {lbl_true} --> Predicted: {lbl_pred}'
        color = 'green' if (int(lbl_true) == int(lbl_pred)) else 'red'
        axes[i, 0].text(20, 200, text, fontsize=10, color=color)

        axes[i, 1].imshow(img)
        axes[i, 1].imshow(mask, cmap='jet', alpha=0.2)
        axes[i, 1].set_title("True Mask")

        heatmap = make_gradcam_heatmap_multi(img_reshaped, model, 'dec_3', 0)
        axes[i, 2].imshow(heatmap)
        axes[i, 2].set_title("Heatmap")

        axes[i, 3].imshow(prediction[1][0])
        axes[i, 3].set_title("Predicted Mask")

    # Hide excess axes
    for ax in axes.ravel():
        ax.axis('off')

    plt.tight_layout()
    plt.show()
    plt.close()