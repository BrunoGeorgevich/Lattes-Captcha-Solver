import os
import cv2
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf

from Tools import label_map_util

def process_captcha(file_path):
    captcha = cv2.imread(file_path, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(captcha, cv2.COLOR_BGR2GRAY)

    filter = np.ones(gray.shape, dtype=np.uint8)
    filter[:, int(filter.shape[1] * 0.60):] *= 0

    _, foreground = cv2.threshold(gray, 238, 255, cv2.THRESH_BINARY)

    foreground = cv2.morphologyEx(foreground, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)),
                                  iterations=2)

    cv2.normalize(foreground, foreground, 255, 0, cv2.NORM_MINMAX)

    return foreground


def to_text(best):
    word = ""
    word += best[0][0]
    word += best[1][0]
    word += best[2][0]
    word += best[3][0]
    return word

def naive_method():
    words = []
    for i in range(0, NUM_VALIDATION_SAMPLES):
        img = process_captcha(VALIDATION_DIR + str(i) + ".png")

        cv2.normalize(img, img, 1, 0, cv2.NORM_MINMAX)
        img = np.float32(img)

        templates_files = os.listdir(TEMPLATE_DIR)
        probabilities = []

        for template_path in templates_files:
            template_name = template_path.replace(".JPG", "")
            template = cv2.imread(TEMPLATE_DIR + template_path, cv2.IMREAD_GRAYSCALE)
            _, thresh = cv2.threshold(template, 80, 255, cv2.THRESH_BINARY)

            cv2.normalize(thresh, thresh, 1, 0, cv2.NORM_MINMAX)
            template = np.float32(thresh)

            w, h = template.shape[::-1]
            res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

            probabilities.append((template_name, max_val,
                                  (max_loc[0], max_loc[1]), (w, h)))

        probabilities.sort(key=lambda tup: tup[1], reverse=True)
        best = probabilities[0:4]
        best.sort(key=lambda tup: tup[2][0], reverse=False)

        word = str.upper(to_text(best))
        words.append(word)
        print("{:6.2f}% -> {}".format(100 * (i / (NUM_VALIDATION_SAMPLES - 1)), word))

    file = open("Naive.csv", 'w')
    file.write('filename,text\n')
    correct = 0

    index = 0
    for word in words:
        if word == FEEDBACK_DATA[index]:
            correct += 1
        file.write('{},{}\n'.format(str(index) + '.png', word))
        index += 1

    print("Accuracy:", str(float(correct / NUM_VALIDATION_SAMPLES) * 100) + "%")   

def naive_method_one_image(imagename):
    img = process_captcha(imagename)

    cv2.normalize(img, img, 1, 0, cv2.NORM_MINMAX)
    img = np.float32(img)

    templates_files = os.listdir(TEMPLATE_DIR)
    probabilities = []

    for template_path in templates_files:
        template_name = template_path.replace(".JPG", "")
        template = cv2.imread(TEMPLATE_DIR + template_path, cv2.IMREAD_GRAYSCALE)
        _, thresh = cv2.threshold(template, 80, 255, cv2.THRESH_BINARY)

        cv2.normalize(thresh, thresh, 1, 0, cv2.NORM_MINMAX)
        template = np.float32(thresh)

        w, h = template.shape[::-1]
        res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        probabilities.append((template_name, max_val,
                              (max_loc[0], max_loc[1]), (w, h)))

    probabilities.sort(key=lambda tup: tup[1], reverse=True)
    best = probabilities[0:4]
    best.sort(key=lambda tup: tup[2][0], reverse=False)

    word = str.upper(to_text(best))
    print(word)

def cnn_method():
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    
        sess = tf.Session(graph=detection_graph)
        
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')
     
    
    f = open('CNN.csv', 'w')
    
    first_line = 'filename,text\n'
    f.write(first_line)
    correct = 0
    
    for index in range(NUM_VALIDATION_SAMPLES):
        IMAGE_NAME = str(index) + '.png'
        PATH_TO_IMAGE = os.path.join(VALIDATION_DIR,IMAGE_NAME)
        
        image = cv2.imread(PATH_TO_IMAGE)
        image_expanded = np.expand_dims(image, axis=0)
        
        (boxes, scores, classes, num) = sess.run(
            [detection_boxes, detection_scores, detection_classes, num_detections],
            feed_dict={image_tensor: image_expanded})
        
        selected = []
        
        for idx in range(4):
            selected.append((np.squeeze(boxes)[idx],CATEGORIES[np.squeeze(classes).astype(np.int32)[idx] - 1]))
            
        selected.sort(key=lambda tup: tup[0][1])
        word = str.upper(''.join([select[1]['name'] for select in selected]))
        
        if word == FEEDBACK_DATA[index]:
            correct += 1
            
        f.write('{},{}\n'.format(IMAGE_NAME, word))
        print('{:6.2f}% -> {} (Accuracy: {:6.2f}%)'
              .format((index/(NUM_VALIDATION_SAMPLES - 1))*100, word, float(correct / NUM_VALIDATION_SAMPLES) * 100))
        
    
    print("Accuracy:", str(float(correct / NUM_VALIDATION_SAMPLES) * 100) + "%")   
    
    f.close() 

def cnn_method_one_image(imagename):
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    
        sess = tf.Session(graph=detection_graph)
        
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')
     
        
    IMAGE_NAME = imagename
    PATH_TO_IMAGE = os.path.join(os.getcwd(),IMAGE_NAME)
    
    image = cv2.imread(PATH_TO_IMAGE)
    image_expanded = np.expand_dims(image, axis=0)
    
    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: image_expanded})
    
    selected = []
    
    for idx in range(4):
        selected.append((np.squeeze(boxes)[idx],CATEGORIES[np.squeeze(classes).astype(np.int32)[idx] - 1]))
        
    selected.sort(key=lambda tup: tup[0][1])
    word = str.upper(''.join([select[1]['name'] for select in selected]))
    print(word)
        
        
#%%

parser = argparse.ArgumentParser(description='Lattes Captcha Solver.')
parser.add_argument('-m','--method', required=False, default='Naive', type=str, help='The method that will be used')
parser.add_argument('--image', required=False, default=False, type=bool, help='If is a batch of images or only one')
parser.add_argument('--imagepath', required=False, default='', type=str, help='Image path')
parser.add_argument('-fp','--folderpath', required=False, default='', type=str, help='Folder path')
parser.add_argument('-ff','--feedbackfile', required=False, default='', type=str, help='Folder path')
parser.add_argument('-n','--num_samples', required=False, default='1000', type=int, help='Number of Files')
parser.add_argument('-mf','--model_folder', required=False, default='Model/Original', type=str, help='Number of Files')
args = parser.parse_args()

if not args.image:
    VALIDATION_DIR = args.folderpath
    NUM_VALIDATION_SAMPLES = args.num_samples
    FEEDBACK_DATA = pd.read_csv(args.feedbackfile)['text']
    
MODEL_FOLDER = args.model_folder

TEMPLATE_DIR = "Best/"
DATA_DIR = "images/"

CWD_PATH = os.getcwd()
NUM_CLASSES = 28

PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_FOLDER,'frozen_inference_graph.pb')
PATH_TO_LABELS = os.path.join(CWD_PATH,MODEL_FOLDER,'labelmap.pbtxt')
LABEL_MAP = label_map_util.load_labelmap(PATH_TO_LABELS)
CATEGORIES = label_map_util.convert_label_map_to_categories(LABEL_MAP, max_num_classes=NUM_CLASSES, use_display_name=True)

if args.method == 'Naive':
    if args.image:
        naive_method_one_image(args.imagepath)
    else:
        naive_method()
elif args.method == 'CNN':
    if args.image:
        cnn_method_one_image(args.imagepath)
    else:
        cnn_method()
else:
    print("Choose a possible method : Naive or CNN")
