import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize

template_dir = "Best/"
captchas_dir = "Captchas/"
feedback_data = pd.read_csv(captchas_dir + 'feedback.csv')['text']

#%%

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
    for i in range(0, 1000):
        img = process_captcha(captchas_dir + str(i) + ".png")

        cv2.normalize(img, img, 1, 0, cv2.NORM_MINMAX)
        img = np.float32(img)

        templates_files = os.listdir(template_dir)
        probabilities = []

        for template_path in templates_files:
            template_name = template_path.replace(".JPG", "")
            template = cv2.imread(template_dir + template_path, cv2.IMREAD_GRAYSCALE)
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
        print("{:6.2f}% -> {}".format(100 * (i / 999), word))

    file = open("Naive.csv", 'w')
    file.write('filename,text\n')
    correct = 0

    index = 0
    for word in words:
        if word == feedback_data[index]:
            correct += 1
        file.write('{},{}\n'.format(str(index) + '.png', word))
        index += 1

    print("Accuracy:", str(float(correct / 1000) * 100) + "%")

#%%

naive_method()
