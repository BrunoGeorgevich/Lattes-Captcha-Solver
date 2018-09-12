import os
import cv2
import numpy as np

def skeletonize(img):
    size = np.size(img)
    skeleton_img = np.zeros(img.shape, np.uint8)

    ret, img = cv2.threshold(img, 127, 255, 0)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    done = False

    while not done:
        eroded = cv2.erode(img, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(img, temp)
        skeleton_img = cv2.bitwise_or(skeleton_img, temp)
        img = eroded.copy()

        zeros = size - cv2.countNonZero(img)
        if zeros == size:
            done = True

    return skeleton_img


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

    only_text = color_thresholding(captcha, (0, 0, 225), (115, 73, 255))
    # only_text = color_thresholding(captcha, (0, 0, 255), (0, 0, 255))

    filter = np.ones(gray.shape, dtype=np.uint8)
    filter[:, int(filter.shape[1] * 0.60):] *= 0

    # only_text = cv2.bitwise_or(cv2.bitwise_or(only_text[:, :, 0], only_text[:, :, 1]), only_text[:, :, 0])
    # only_text = cv2.bitwise_and(only_text, filter)

    _, foreground = cv2.threshold(gray, 238, 255, cv2.THRESH_BINARY)

    foreground = cv2.morphologyEx(foreground, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)),
                                  iterations=2)

    cv2.normalize(foreground, foreground, 255, 0, cv2.NORM_MINMAX)

    # cv2.imshow("Image", foreground)
    #
    # while cv2.waitKey(0) != ord('q'):
    #     pass

    return foreground

template_dir = "Best/"
captchas_dir = "Captchas/"


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
        if len(np.intersect1d(range(char[2][0], char[2][0] + char[3][0]), range(new_char[2][0], new_char[2][0] + new_char[3][0]))):
            return index
        index += 1
    return -1


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

file = open("teste.txt", 'w')
for word in words:
    file.write("%s\n" % word)
