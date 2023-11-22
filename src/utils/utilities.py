import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
import keras
import tensorflow as tf

def print_split_ds_info(ds_info):
    # Print textual split information
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


def plot_frames_split(ds_info, save_path, log_scale=False, output_mode=(False,True)):
    # Create data for the plot
    centers = []
    train_frame_counts = []
    val_frame_counts = []
    test_frame_counts = []

    for medical_center in ds_info['medical_center_patients'].keys():
        centers.append(medical_center)
        train_frame_count = sum(ds_info['frames_by_center_patient'][medical_center][patient] for patient in ds_info['train_patients_by_center'][medical_center])
        val_frame_count = sum(ds_info['frames_by_center_patient'][medical_center][patient] for patient in ds_info['val_patients_by_center'][medical_center])
        test_frame_count = sum(ds_info['frames_by_center_patient'][medical_center][patient] for patient in ds_info['test_patients_by_center'][medical_center])
        train_frame_counts.append(train_frame_count)
        val_frame_counts.append(val_frame_count)
        test_frame_counts.append(test_frame_count)

    # Create the plot
    plt.figure(figsize=(10, 6))
    if log_scale:
        plt.barh(centers, train_frame_counts, label='Train frames', log=True)
        plt.barh(centers, val_frame_counts, left=train_frame_counts, label='Val frames', log=True)
        plt.barh(centers, test_frame_counts, left=[sum(x) for x in zip(train_frame_counts, val_frame_counts)], label='Test frames', log=True)
    else:
        plt.barh(centers, train_frame_counts, label='Train frames')
        plt.barh(centers, val_frame_counts, left=train_frame_counts, label='Val frames')
        plt.barh(centers, test_frame_counts, left=[sum(x) for x in zip(train_frame_counts, val_frame_counts)], label='Test frames')

    # Add labels and legend
    plt.xlabel('Frame Count (Log Scale)' if log_scale else 'Frame Count')
    plt.ylabel('Medical Center')
    plt.title('Frame Distribution by Medical Center')
    plt.legend()

    show_plot, save = output_mode
    if save:
        plt.savefig(save_path)

    # display the plot
    if show_plot:
        plt.show()
    
    plt.clf()
    plt.close()


def plot_patients_split(ds_info, save_path, output_mode=(False,True)):
    # Create data for the plot
    centers = []
    train_patient_counts = []
    val_patient_counts = []
    test_patient_counts = []

    for medical_center in ds_info['medical_center_patients'].keys():
        centers.append(medical_center)
        train_patient_count = len(ds_info['train_patients_by_center'][medical_center])
        val_patient_count = len(ds_info['val_patients_by_center'][medical_center])
        test_patient_count = len(ds_info['test_patients_by_center'][medical_center])
        train_patient_counts.append(train_patient_count)
        val_patient_counts.append(val_patient_count)
        test_patient_counts.append(test_patient_count)

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.barh(centers, train_patient_counts, label='Train patients')
    plt.barh(centers, val_patient_counts, left=train_patient_counts, label='Val patients')
    plt.barh(centers, test_patient_counts, left=[sum(x) for x in zip(train_patient_counts, val_patient_counts)], label='Test patients')

    # Add labels, title and legend
    plt.xlabel('Patient Count')
    plt.ylabel('Medical Center')
    plt.title('Patient Distribution by Medical Center')
    plt.legend()

    show_plot, save = output_mode
    if save:
        plt.savefig(save_path)

    # display the plot
    if show_plot:
        plt.show()

    plt.clf()
    plt.close()

def plot_labels_distr(y_train_ds, y_val_ds, y_test_ds, save_path, output_mode=(False,True)):
    # calculate the class count for each set
    class_counts_val = np.bincount(y_val_ds)
    class_counts_test = np.bincount(y_test_ds)
    class_counts_train = np.bincount(y_train_ds)

    class_labels = np.arange(len(class_counts_val)).astype(int)
    group_labels = np.arange(len(class_labels))
    bar_width = 0.2

    bar_positions_train = class_labels - bar_width
    bar_positions_val = class_labels
    bar_positions_test = class_labels + bar_width

    # Create the plot
    plt.bar(bar_positions_train, class_counts_train, width=bar_width, label='Train frames')
    plt.bar(bar_positions_val, class_counts_val, width=bar_width, label='Validation frames')
    plt.bar(bar_positions_test, class_counts_test, width=bar_width, label='Test frames')
    
    # Add labels, title and legend
    plt.xlabel('Classes')
    plt.ylabel('Frames')
    plt.title('Distribution of labels for each set')
    plt.xticks(group_labels, [0, 1, 2, 3])
    plt.legend()  

    show_plot, save = output_mode
    if save:
        plt.savefig(save_path)

    # display the plot
    if show_plot:
        plt.show()
    
    plt.clf()
    plt.close()

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

def plot_split_graphs(ds_info):
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
    plt.clf()
    plt.close()    

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

def plot_fdistr_per_class_pie(y_train_ds, y_val_ds, y_test_ds, save_path = None, output_mode=(False, True)):
    sets = ['Train', 'Validation', 'Test']
    datasets = [y_train_ds, y_val_ds, y_test_ds]
    colors = ['#ffd7f4','#d896bb', '#b15680', '#880044']

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for i, set_name in enumerate(sets):
        class_counts = np.bincount(datasets[i])
        labels = np.arange(len(class_counts)).astype(int)

        # Ordina le etichette e le frequenze in base alle frequenze decrescenti
        sorted_indices = np.argsort(class_counts)[::-1]
        class_counts = class_counts[sorted_indices]
        labels = labels[sorted_indices]

        # Impostazione di alcuni valori di explode per staccare le fette
        explode = [0.05 if count > 0 else 0 for count in class_counts]

        # Utilizza i colori personalizzati
        wedges, _, _ = axes[i].pie(class_counts, labels=labels, autopct=lambda p: '{:.0f}\n({:.1f}%)'.format(p * sum(class_counts) / 100, p), startangle=90, explode=explode, colors=colors)
        axes[i].set_title(f'{set_name} Set')  # Imposta il grassetto per il titolo


    # Creazione di una legenda unica per tutta la figura
    legend_labels = [f'Class {label}' for label in labels]
    fig.legend(wedges, legend_labels, title='Classes', loc='lower center', ncol=len(set_name))

    plt.suptitle('Frames distribution in sets for each class', y=1.05)
    
    show_plot, save = output_mode
    if save:
        plt.savefig(save_path)

    # Visualizza il grafico
    if show_plot:
        plt.show()

    plt.clf()
    plt.close()