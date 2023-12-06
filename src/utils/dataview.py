import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# ---- GRAPH GENERATION ----
def plot_charts(exp, charts, display, save, save_path):
    # create the charts subfolder 
    charts_path = os.path.join(save_path, 'charts/')
    os.makedirs(charts_path, exist_ok=True)
    
    if "pdistr" in charts:
        pps = plot_patients_split(exp.dataset.split, display=display)
        if save:
            chart_file_path = os.path.join(charts_path, "split_per_patients.png")
            pps.savefig(chart_file_path)
            plt.close()
    
    if "lsdistr_pie" in charts:
        pfpcp = plot_fdistr_per_class_pie(exp.y_train, exp.y_val, exp.y_test, display=display)
        if save:
            chart_file_path = os.path.join(charts_path, "frames_distr_per_class_pie.png")
            pfpcp.savefig(chart_file_path)
            plt.close()

    if "ldistr" in charts:
        ds_labels = list(exp.y_train) + list(exp.y_val) + list(exp.y_test)
        pld = plot_labels_distr(ds_labels, display=display)
        if save:
            chart_file_path = os.path.join(charts_path, "labels_distr.png")
            pld.savefig(chart_file_path)
            plt.close()

def plot_patients_split(dataset_split, display=False):
    # Estrai i centri medici e i pazienti dai dati
    centri_medici = list(set([paziente.split('/')[0] for split, pazienti in dataset_split.items() for paziente in pazienti]))
    sets = list(dataset_split.keys())

    # Conta il numero di pazienti per centro medico e set
    counts = np.zeros((len(centri_medici), len(sets)))
    for i, centro_medico in enumerate(centri_medici):
        for j, split in enumerate(sets):
            counts[i, j] = sum(1 for paziente in dataset_split[split] if paziente.startswith(centro_medico))

    # Crea il grafico a barre impilato
    plt.figure(figsize=(12, 6))
    bottom = np.zeros(len(centri_medici))
    for j, split in enumerate(sets):
        plt.bar(centri_medici, counts[:, j], bottom=bottom, label=split)
        bottom += counts[:, j]

    plt.title('Patient Distribution by Medical Center in sets')
    plt.xlabel('Medical center')
    plt.ylabel('Patient count')
    plt.legend()
    plt.xticks(rotation=45, ha='right')

    # Display the plot
    if display:
        plt.show()

    return plt.gcf()

def plot_fdistr_per_class_pie(y_train_ds, y_val_ds, y_test_ds, display=False):
    sets = ['Train', 'Validation', 'Test']
    datasets = [y_train_ds, y_val_ds, y_test_ds]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for i, set_name in enumerate(sets):
        class_counts = np.bincount(datasets[i])
        labels = np.arange(len(class_counts)).astype(int)

        wedges, _, autotexts = axes[i].pie(class_counts, labels=labels, autopct=lambda p: '{:.1f}%\n({:.0f})'.format(p, p * sum(class_counts) / 100), startangle=90)
        axes[i].set_title(f'{set_name} Set')

        # Aggiungi etichette con il numero di frame per ogni fetta
        #for autotext in autotexts:
            #autotext.set_color('white')  # Imposta il colore del testo a bianco per una migliore leggibilità

    # Creazione di una legenda unica per tutta la figura
    legend_labels = [f'Class {label}' for label in labels]
    fig.legend(wedges, legend_labels, title='Classes', loc='lower center', ncol=len(set_name))

    plt.suptitle('Frames distribution in sets for each class', y=1.05)

    # display the plot
    if display:
        plt.show()

    return plt.gcf()

def plot_labels_distr(labels, display=False):
    # create an occurrence count of each class
    counts = {label: labels.count(label) for label in set(labels)}

    # converts the count into two separate lists for plotting
    class_names, class_counts = zip(*counts.items())

    # create a bar-plot 
    plt.figure(figsize=(10, 6))
    plt.bar(class_names, class_counts)
    plt.title('Labels distribution in the dataset')
    plt.xlabel('Classes')
    plt.ylabel('Frames')
    
    # display the plot
    if display:
        plt.show()
    
    return plt.gcf()

def plot_train_history(exp_name, history, keys, save_path, output_mode):
    display, save = output_mode

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
        fig.suptitle(exp_name)
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
        fig.suptitle(exp_name)

    if save:
        train_graphs_path = os.path.join(save_path, f"train_graphs{keys[0]}.png")
        plt.savefig(train_graphs_path)

    # Show the figure
    if display:
        plt.show()
    
    plt.clf()
    plt.close()

def confusionmatrix(exp, y_test, y_pred):
    cf_matrix_test = confusion_matrix(y_test, y_pred, normalize='true', labels=[list(range(exp.n_class))])

    ax = sns.heatmap(cf_matrix_test, linewidths=1, annot=True, fmt='.2f', cmap="BuPu")
    ax.set_ylabel('Actual')
    ax.set_xlabel('Predicted')
    ax.set_title('Test Set Confusion Matrix')

    display, save = exp.output_mode

    if save:
        train_graphs_path = os.path.join(exp.exp_path, 'fig' , "confusion_matrix.png")
        plt.savefig(train_graphs_path)

    # Show the figure
    if display:
        plt.show()

    plt.clf()
    plt.close()

# --- IMAGE VISUALIZATION ---

def plot_set_batches(exp, set='train', num_batches=10):
    # gather the needed settings and data
    batch_size = exp.exp_config['batch_size']
    # color map
    class_colors = {0: 'green', 1: 'darkblue', 2: 'darkorange', 3: 'darkred'}
    
    sets = {'train': exp.x_train, 'val': exp.x_val}
    selected_set = sets.get(set, exp.x_test)

    for batch in selected_set.take(num_batches):
        #_, axes = plt.subplots(2, 8, figsize=(20, 6))
        _, axes = plt.subplots(batch_size // 8, 8, figsize=(20, 3 * (batch_size // 8)))
        
        frames, labels = batch
        
        for i, (frame, label) in enumerate(zip(frames, labels)):        
            # Stampa l'immagine
            #axes[i % 2, i // 2].imshow(frame)
            axes[i // 8, i % 8].imshow(frame)

            # imposta colore e stampa etichetta
            color = class_colors.get(np.argmax(label), 'black')
            #axes[i % 2, i // 2].set_title(f'Target: {label}', color=color)
            axes[i // 8, i % 8].set_title(f'Target: {label}', color=color)

            # Nascondi gli assi
            #axes[i % 2, i // 2].axis('off')
            axes[i // 8, i % 8].axis('off')

        plt.tight_layout()
        plt.show()
        plt.close()