# ---------------------------------------------------------------------------- #
#                                 "lib" module
#
# Library Name: lib
# Author: Lorenzo Federici
# Creation Date: September 21, 2023
# Description: This library contains a set of useful tools for msak manipulation
# Project Name: LUS-MULTITASK-LEARNING
# ---------------------------------------------------------------------------- #

# ---------------------------------------------------------------------------- #
#                                    Imports                                   #
# ---------------------------------------------------------------------------- #

import os
import io
import json
import base64
import cv2

import PIL
from PIL import Image
import h5py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import uuid

from tqdm import tqdm

from pathlib import Path
from pathlib import PosixPath

# ---------------------------------------------------------------------------- #
#                                    Methods                                   #
# ---------------------------------------------------------------------------- #

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Function For Mask~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def img_data_to_pil(img_data):
    f = io.BytesIO()
    f.write(img_data)
    img_pil = Image.open(f)
    return img_pil

def img_data_to_arr(img_data):
    img_pil = img_data_to_pil(img_data)
    img_arr = np.array(img_pil)
    return img_arr

def img_b64_to_arr(img_b64):
    img_data = base64.b64decode(img_b64)
    img_arr = img_data_to_arr(img_data)
    return img_arr

def convert_json_to_mask(json_file):
    data = json.load(open(json_file))

    imageData = data.get('imageData')

    # se per qualche motivo il json è rotto allora prendo li png nella cartella
    if not imageData:
        imagePath = os.path.join(os.path.dirname(json_file), data['imagePath'])
        with open(imagePath, 'rb') as f:
            imageData = f.read()
            imageData = base64.b64encode(imageData).decode('utf-8')
    img = img_b64_to_arr(imageData)

    label_name_to_value = {'_background_': -1, '0':0,'1':1,'2':2,'3':3}
    for shape in sorted(data['shapes'], key=lambda x: x['label']):
        label_name = shape['label']
    lbl, _ = shapes_to_label(img.shape, data['shapes'], label_name_to_value)
    return lbl

def shape_to_mask(img_shape, points, shape_type=None,line_width=10, point_size=5):
    mask = np.zeros(img_shape[:2], dtype=np.uint8)
    mask = PIL.Image.fromarray(mask)
    draw = PIL.ImageDraw.Draw(mask)
    xy = [tuple(point) for point in points]
    if shape_type == 'circle':
        assert len(xy) == 2, 'Shape of shape_type=circle must have 2 points'
        (cx, cy), (px, py) = xy
        d = math.sqrt((cx - px) ** 2 + (cy - py) ** 2)
        draw.ellipse([cx - d, cy - d, cx + d, cy + d], outline=1, fill=1)
    elif shape_type == 'rectangle':
        assert len(xy) == 2, 'Shape of shape_type=rectangle must have 2 points'
        draw.rectangle(xy, outline=1, fill=1)
    elif shape_type == 'line':
        assert len(xy) == 2, 'Shape of shape_type=line must have 2 points'
        draw.line(xy=xy, fill=1, width=line_width)
    elif shape_type == 'linestrip':
        draw.line(xy=xy, fill=1, width=line_width)
    elif shape_type == 'point':
        assert len(xy) == 1, 'Shape of shape_type=point must have 1 points'
        cx, cy = xy[0]
        r = point_size
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=1, fill=1)
    else:
        assert len(xy) > 2, 'Polygon must have points more than 2'
        draw.polygon(xy=xy, outline=1, fill=1)
    mask = np.array(mask, dtype=bool)
    return mask

def shapes_to_label(img_shape, shapes, label_name_to_value):

    cls = np.ones(img_shape[:2], dtype=np.int32)*label_name_to_value['_background_']
    ins = np.ones_like(cls)*label_name_to_value['_background_']
    instances = []
    for shape in shapes:
        points = shape['points']
        label = shape['label']
        group_id = shape.get('group_id')
        if group_id is None:
            group_id = uuid.uuid1()
        shape_type = shape.get('shape_type', None)

        cls_name = label
        instance = (cls_name, group_id)

        if instance not in instances:
            instances.append(instance)
        ins_id = instances.index(instance) + 1
        cls_id = label_name_to_value[cls_name]

        mask = shape_to_mask(img_shape[:2], points, shape_type)
        cls[mask] = cls_id
        ins[mask] = ins_id

    return cls, ins

def stringToRGB(base64_string):
    imgdata = base64.b64decode(str(base64_string))
    img = Image.open(io.BytesIO(imgdata))
    opencv_img= cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)
    return opencv_img

def get_mask(json_list):
    masks_list = []
    imgs_list  = []

    # TODO: gestire img e le shape delle maschere
    #with open(json_list[0], 'r') as file:
    #    data = json.load(file)

    for json_file in json_list:
        with open(json_file, 'r') as file:
            data = json.load(file)

        imageData = data.get('imageData')
        img = stringToRGB(imageData)

        label_name_to_value = {'_background_': -1, '0':0,'1':1,'2':2,'3':3}
        for shape in sorted(data['shapes'], key=lambda x: x['label']):
            label_name = shape['label']
        lbl, _ = shapes_to_label(img.shape, data['shapes'], label_name_to_value)
        masks_list.append(lbl)
        imgs_list.append(img)

    # list to mask
    masks_mat = np.stack(masks_list, axis=-1)
    imgs_mat  = np.stack(imgs_list, axis=-1)
    return [masks_mat, imgs_mat]
    
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
