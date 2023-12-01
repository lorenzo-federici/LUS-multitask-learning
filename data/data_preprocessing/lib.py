# ---------------------------------------------------------------------------- #
#                                 "lib" module
#
# Library Name: lib
# Author: Lorenzo Federici
# Creation Date: October 10, 2023
# Description: This library contains a set of useful tools
# Project Name: LUS-MULTITASK-LEARNING
# ---------------------------------------------------------------------------- #

# ---------------------------------------------------------------------------- #
#                                    Imports                                   #
# ---------------------------------------------------------------------------- #
import numpy as np
from functools import reduce
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------- #
#                                    Methods                                   #
# ---------------------------------------------------------------------------- #
def get_original_mask(new_mask):
    # Ottieni il numero di classi dalla forma della nuova matrice
    num_classes = new_mask.shape[2]
    
    # Inizializza una nuova maschera con la stessa forma della nuova matrice
    shape = new_mask.shape[:2]
    mask  = np.zeros(shape + (1,), dtype=np.int32)
    
    # Utilizza un ciclo for per assegnare i valori di classe alla maschera originale
    for class_id in range(num_classes):
        mask[new_mask[:, :, class_id] > 0.5] = (class_id - 1 )
    
    return mask

def plot_side_by_side(img_arrays):
    flatten_list = reduce(lambda x, y: x + y, img_arrays)
    
    ncol = len(img_arrays)
    nrow = len(flatten_list) // ncol

    f, axs = plt.subplots(nrow, ncol, sharex='all', sharey='all', figsize=(ncol * 4, nrow * 4))

    for i in range(len(flatten_list)):
        axs[i % nrow, i // nrow].imshow(flatten_list[i])

def plot_gt_side_pred(img_arrays):
    img, mask, pred = img_arrays
    ncol = 2
    nrow = len(img)

    f, axs = plt.subplots(nrow, ncol, sharex='all', sharey='all', figsize=(ncol * 4, nrow * 4))

    for i in range(nrow):
        # Mostra l'immagine con la maschera ground truth nella prima colonna
        axs[i, 0].imshow(img[i])
        axs[i, 0].imshow(mask[i], alpha=0.5, cmap='viridis')  # Sovrapponi la maschera

        # Mostra l'immagine con la maschera predetta nella seconda colonna
        axs[i, 1].imshow(img[i])
        axs[i, 1].imshow(pred[i], alpha=0.5, cmap='viridis')  # Sovrapponi la maschera