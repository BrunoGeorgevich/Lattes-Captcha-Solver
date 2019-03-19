#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 01:50:44 2019

@author: bruno
"""
import os
import cv2
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf

import matplotlib.pyplot as plt

from Tools import label_map_util

MODEL_FOLDER = 'Model/Modified/'

CWD_PATH = os.getcwd()
NUM_CLASSES = 28

PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_FOLDER,'frozen_inference_graph.pb')
PATH_TO_LABELS = os.path.join(CWD_PATH,MODEL_FOLDER,'labelmap.pbtxt')
LABEL_MAP = label_map_util.load_labelmap(PATH_TO_LABELS)
CATEGORIES = label_map_util.convert_label_map_to_categories(
        LABEL_MAP,
        max_num_classes=NUM_CLASSES,
        use_display_name=True)


detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.Session(graph=detection_graph)

#%%

layers = []

for a in detection_graph.get_operations():
    if 'relu' in a.name.lower():
        layers.append(a.name)  
#%%
    
images_per_row = [16 for i in range(len(layers))]

for idx in range(48,len(layers)):
    l = layers[idx]
    # if(idx == 12):
    #     break
    
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')
    activations = detection_graph.get_tensor_by_name('{}:0'.format(l))
         
    IMAGE_NAME = 'images/Modified_Captcha/validation/0.png'
    PATH_TO_IMAGE = os.path.join(CWD_PATH,IMAGE_NAME)
    
    image = cv2.imread(PATH_TO_IMAGE)
    image_expanded = np.expand_dims(image, axis=0)
    
    (boxes, scores, classes, num, activations) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections, activations],
        feed_dict={image_tensor: image_expanded})
    
    n_features = activations.shape[-1]
    size = activations.shape[1]
    width = activations.shape[1]
    height = activations.shape[2]
    
    n_cols = n_features // images_per_row[idx]
    display_grid = np.zeros((width*n_cols, images_per_row[idx]*height))
    
    for col in range(n_cols):
        for row in range(images_per_row[idx]):
            channel_image = activations[0,:,:,col * images_per_row[idx] + row]
            channel_image -= channel_image.mean()
            channel_image /= channel_image.std()
            channel_image *= 64
            channel_image += 128
            channel_image = np.clip(channel_image, 0, 255).astype('uint8')
            display_grid[col * width : (col + 1) * width,
                         row * height: (row + 1) * height] = channel_image
            # cv2.imshow('a', channel_image)
            # cv2.waitKey(100)
    
    # cv2.destroyAllWindows()
    # plt.figure(figsize=(16,9))
    # plt.title(l)
    # plt.grid(False)
    # plt.imshow(display_grid, aspect='auto', cmap='viridis') 
    plt.imsave('test/{}.png'.format(idx),display_grid, cmap='viridis')

    idx += 1       
    
    print('{} -> {:.2f}%'.format(idx, idx*100/len(layers)))

#%%
selected = []

for idx in range(4):
    selected.append((np.squeeze(boxes)[idx],CATEGORIES[np.squeeze(classes).astype(np.int32)[idx] - 1]))
    
selected.sort(key=lambda tup: tup[0][1])
word = str.upper(''.join([select[1]['name'] for select in selected]))
print(word)


#%%
weights_data = []

for i in range(0,64):
    weights_data.append(w[0,:,:,i])

fig = plt.figure(1, figsize=(16,9))

cols = 8
rows = 8

# CATEGORIES.append({'id':0,'name':'None'})
CATEGORIES = sorted(CATEGORIES, key=lambda d:d['id'])

i = 1
for weight in weights_data:
    fig.add_subplot(rows, cols, i)
    # plt.title(CATEGORIES[i -1]['name'])
    plt.imshow(cv2.normalize(weight, None, 0, 255, cv2.NORM_MINMAX), cmap='viridis')
    i += 1
    
plt.plot() 
    
#%%
    
print(detection_graph.get_operation_by_name('num_detections'))