import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize

template_dir = "Best/"
captchas_dir = "Captchas/"


# def skeletonize(img):
#     size = np.size(img)
#     skeleton_img = np.zeros(img.shape, np.uint8)
#
#     ret, img = cv2.threshold(img, 127, 255, 0)
#     element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
#     done = False
#
#     while not done:
#         eroded = cv2.erode(img, element)
#         temp = cv2.dilate(eroded, element)
#         temp = cv2.subtract(img, temp)
#         skeleton_img = cv2.bitwise_or(skeleton_img, temp)
#         img = eroded.copy()
#
#         zeros = size - cv2.countNonZero(img)
#         if zeros == size:
#             done = True
#
#     return skeleton_img


def reconstruct(seeds, mask):
    last_image = 0

    while True:
        seeds = cv2.dilate(seeds, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
        seeds = cv2.bitwise_and(seeds, mask)

        if (seeds == last_image).all():
            break

        last_image = seeds

    return seeds


def color_thresholding(image, min_val_hsv, max_val_hsv):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, min_val_hsv, max_val_hsv)
    imask = mask > 0
    extracted = np.zeros_like(image, np.uint8)
    extracted[imask] = image[imask]
    return extracted


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


def has_someone(chosen_chars, min, max):
    if len(chosen_chars) == 0:
        return -1

    i = 0
    for char in chosen_chars:
        if min <= char <= max:
            return i
        i += 1

    return -1


def at_least_distance(chosen_chars, new_char):
    index = 0
    for char in chosen_chars:
        if len(np.intersect1d(range(char[2][0], char[2][0] + char[3][0]),
                              range(new_char[2][0], new_char[2][0] + new_char[3][0]))):
            return index
        index += 1
    return -1


def method_1():
    words = []
    for i in range(605):
        img = process_captcha(captchas_dir + str(i) + ".png")
        templates_files = os.listdir(template_dir)
        probabilities = []

        for template_path in templates_files:
            template_name = template_path.replace(".JPG", "")
            template = cv2.imread(template_dir + template_path, cv2.IMREAD_GRAYSCALE)

            w, h = template.shape[::-1]
            res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            probabilities.append((template_name, max_val, (max_loc[0], max_loc[1]), (w, h)))

        probabilities.sort(key=lambda tup: tup[1], reverse=True)
        chosen_chars = []

        for p in probabilities:
            if len(chosen_chars) == 0:
                chosen_chars.append(p)
            elif at_least_distance(chosen_chars, p):
                index = at_least_distance(chosen_chars, p)
                if index == -1 and len(chosen_chars) < 4:
                    chosen_chars.append(p)
                elif chosen_chars[index][1] < p[1]:
                    chosen_chars[index] = p

        chosen_chars.sort(key=lambda tup: tup[2][0])

        word = ""
        for char in chosen_chars:
            word += char[0]

        print(word)
        words.append(word)
        print("%.2f %%" % (100 * (i / 604)))

    file = open("Method_1.txt", 'w')
    for word in words:
        file.write("%s\n" % word)


def to_text(best):
    word = ""
    word += best[0][0]
    word += best[1][0]
    word += best[2][0]
    word += best[3][0]
    return word


def method_2():
    template_max_pixels = pd.read_csv("max_pixels_templates.csv", sep=',')
    template_max_pixels = template_max_pixels.set_index("Template", drop=False)

    words = []
    for i in range(605):
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

            template = skeletonize(np.array(template, dtype=np.float32))
            template = np.array(template, dtype=np.float32)

            w, h = template.shape[::-1]
            res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

            probabilities.append((template_name, max_val / template_max_pixels.loc[template_name, "Max Pixels"],
                                  (max_loc[0], max_loc[1]), (w, h)))

        probabilities.sort(key=lambda tup: tup[1], reverse=True)
        best = probabilities[0:4]
        best.sort(key=lambda tup: tup[2][0], reverse=False)

        word = to_text(best)
        print(word)
        words.append(word)
        print("%.2f %%" % (100 * (i / 604)))

    file = open("Method_2.txt", 'w')
    correct = 0

    feedback_file = open("first_letter.txt", 'r')
    feedback_data = feedback_file.read()
    feedback_file.close()

    feedback_data = feedback_data.split(sep='\n')

    index = 0
    for word in words:
        if word == feedback_data[index]:
            correct += 1
        file.write("%s\n" % word)
        index += 1

    print("Acerto:", str(float(correct / 605)))


def method_3():
    words = []
    for i in range(605):
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

        word = to_text(best)
        print(word)
        words.append(word)
        print("%.2f %%" % (100 * (i / 604)))

    file = open("Method_3.txt", 'w')
    correct = 0

    feedback_file = open("first_letter.txt", 'r')
    feedback_data = feedback_file.read()
    feedback_file.close()

    feedback_data = feedback_data.split(sep='\n')

    index = 0
    for word in words:
        if word == feedback_data[index]:
            correct += 1
        file.write("%s\n" % word)
        index += 1

    print("Acerto:", str(float(correct / 605) * 100) + "%")


def retrieve_scores(captchas):
    file = open('scores.csv', 'w')
    file.write('Captcha,1,2,3,4,5,6,7,8,9,b,c,d,f,g,h,j,k,l,m,n,p,r,s,t,v,w,x,z\n')
    for i in captchas:
        img = process_captcha(captchas_dir + str(i) + ".png")

        cv2.normalize(img, img, 1, 0, cv2.NORM_MINMAX)
        img = np.float32(img)

        templates_files = os.listdir(template_dir)
        probabilities = [str(i)]

        for template_path in templates_files:
            template = cv2.imread(template_dir + template_path, cv2.IMREAD_GRAYSCALE)
            _, thresh = cv2.threshold(template, 80, 255, cv2.THRESH_BINARY)

            cv2.normalize(thresh, thresh, 1, 0, cv2.NORM_MINMAX)
            template = np.float32(thresh)

            res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

            res_temp = cv2.matchTemplate(template, template, cv2.TM_CCOEFF_NORMED)
            _, max_val_temp, _, _ = cv2.minMaxLoc(res_temp)
            print(max_val_temp)

            probabilities.append("\"" + str(max_val).replace(".", ",") + "\"")

        file.write(','.join(probabilities) + '\n')

    file.close()


# errors = [11, 13, 32, 38, 44, 45, 61, 98, 105, 122, 126, 147, 155, 185, 243, 261, 287, 297, 301, 316, 342, 364, 397,
#           404, 406, 444, 489, 531, 557, 572, 582, 598]

# retrieve_scores(errors)

method_3()

# templates_files = os.listdir(template_dir)
#
# print("Template,Max Pixels")
# for template_path in templates_files:
#     template_name = template_path.replace(".JPG", "")
#     img = cv2.imread(template_dir + template_path, cv2.IMREAD_GRAYSCALE)
#     _, thresh = cv2.threshold(img, 80, 255, cv2.THRESH_BINARY)
#
#     cv2.normalize(thresh, thresh, 1, 0, cv2.NORM_MINMAX)
#     thresh = np.float32(thresh)
#
#     # skel = skeletonize(np.array(thresh, dtype=np.float32))
#     # skel = np.array(skel, dtype=np.float32)
#
#     res = cv2.matchTemplate(thresh, thresh, cv2.TM_CCOEFF_NORMED)[0][0]
#     print(template_name + "," + str(res))
#
# template_max_pixels = pd.read_csv("max_pixels_templates.csv", sep=',')
# template_max_pixels = template_max_pixels.set_index("Template", drop = False)
#
# print(template_max_pixels.loc["1", "Max Pixels"])

# file = open('Method_3.txt', 'r')
# data = file.read().split(sep='\n')
# file.close()
#
# positions = []
# i = 0
#
# file = open('gabarito.txt', 'r')
# gabarito = file.read().split(sep='\n')
# file.close()
#
# print("Captcha,Predição,Correta")
#
# for val in data:
#     if val != gabarito[i]:
#         print(str(i) + ',' + val + ',' + gabarito[i])
#
#     i += 1

# print(positions)
# print(data[positions])
# print(len(positions))
