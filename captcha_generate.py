#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 00:46:02 2019

@author: bruno
"""
# coding: utf-8
"""
    captcha.image
    ~~~~~~~~~~~~~

    Generate Image CAPTCHAs, just the normal image CAPTCHAs you are using.
"""

import os
import random
from PIL import Image
from PIL import ImageFilter
from PIL.ImageDraw import Draw
from PIL.ImageFont import truetype
try:
    from cStringIO import StringIO as BytesIO
except ImportError:
    from io import BytesIO
try:
    from wheezy.captcha import image as wheezy_captcha
except ImportError:
    wheezy_captcha = None

if wheezy_captcha:
    __all__ = ['ImageCaptcha', 'WheezyCaptcha']
else:
    __all__ = ['ImageCaptcha']


table  =  []
for  i  in  range( 256 ):
    table.append( i * 1.97 )


class _Captcha(object):
    def generate(self, chars, format='png'):
        """Generate an Image Captcha of the given characters.

        :param chars: text to be generated.
        :param format: image file format
        """
        im = self.generate_image(chars)
        out = BytesIO()
        im.save(out, format=format)
        out.seek(0)
        return out

    def write(self, chars, output, format='png'):
        """Generate and write an image CAPTCHA data to the output.

        :param chars: text to be generated.
        :param output: output destination.
        :param format: image file format
        """
        im = self.generate_image(chars)
        return im.save(output, format=format)


class ImageCaptcha(_Captcha):
    """Create an image CAPTCHA.

    Many of the codes are borrowed from wheezy.captcha, with a modification
    for memory and developer friendly.

    ImageCaptcha has one built-in font, DroidSansMono, which is licensed under
    Apache License 2. You should always use your own fonts::

        captcha = ImageCaptcha(fonts=['/path/to/A.ttf', '/path/to/B.ttf'])

    You can put as many fonts as you like. But be aware of your memory, all of
    the fonts are loaded into your memory, so keep them a lot, but not too
    many.

    :param width: The width of the CAPTCHA image.
    :param height: The height of the CAPTCHA image.
    :param fonts: Fonts to be used to generate CAPTCHA images.
    :param font_sizes: Random choose a font size from this parameters.
    """
    def __init__(self, width=160, height=60, fonts=None, font_sizes=None):
        self._width = width
        self._height = height
        self._fonts = fonts
        self._font_sizes = font_sizes or (42, 50, 56)
        self._truefonts = []

    @property
    def truefonts(self):
        if self._truefonts:
            return self._truefonts
        self._truefonts = tuple([
            truetype(n, s)
            for n in self._fonts
            for s in self._font_sizes
        ])
        return self._truefonts

    @staticmethod
    def create_noise_curve(image, color):
        w, h = image.size
        x1 = random.randint(0, int(w / 5))
        x2 = random.randint(w - int(w / 5), w)
        y1 = random.randint(int(h / 5), h - int(h / 5))
        y2 = random.randint(y1, h - int(h / 5))
        points = [x1, y1, x2, y2]
        end = random.randint(160, 200)
        start = random.randint(0, 20)
        Draw(image).arc(points, start, end, fill=color)
        return image

    @staticmethod
    def create_noise_dots(image, color, width=3, number=30):
        draw = Draw(image)
        w, h = image.size
        while number:
            x1 = random.randint(0, w)
            y1 = random.randint(0, h)
            draw.line(((x1, y1), (x1 - 1, y1 - 1)), fill=color, width=width)
            number -= 1
        return image

    def create_captcha_image(self, chars, color, background):
        """Create the CAPTCHA image itself.

        :param chars: text to be generated.
        :param color: color of the text.
        :param background: color of the background.

        The color should be a tuple of 3 numbers, such as (0, 255, 255).
        """
        image = Image.new('RGB', (self._width, self._height), background)
        draw = Draw(image)

        def _draw_character(c):
            font = random.choice(self.truefonts)
            w, h = draw.textsize(c, font=font)

            dx = 4
            dy = 6
            im = Image.new('RGBA', (w + dx, h + dy))
            Draw(im).text((dx, dy), c, font=font, fill=color)
            return im

        images = []
        for c in chars:
            if random.random() > 0.5:
                images.append(_draw_character(" "))
            images.append(_draw_character(c))

        text_width = sum([im.size[0] for im in images])

        width = max(text_width, self._width)
        image = image.resize((width, self._height))

        average = int(text_width / len(chars))
        rand = int(0.25 * average)
        offset = int(average * 0.1)

        for im in images:
            w, h = im.size
            mask = im.convert('L').point(table)
            image.paste(im, (offset, int((self._height - h) / 2)), mask)
            offset = offset + w + random.randint(-rand, 0)

        if width > self._width:
            image = image.resize((self._width, self._height))

        return image


#%%

import os
import cv2
import imutils
import numpy as np
import pandas as pd
from PIL import Image

WHITE = (255,255,255)
HEIGHT = 100
WIDTH = 600

BACKGROUND_HEIGHT = WIDTH

FONT_DIR = 'fonts/'

fonts = os.listdir(FONT_DIR)

for i in range(len(fonts)):
    fonts[i] = FONT_DIR + fonts[i]

XML_DATA = '''<annotation>
  <folder>test</folder>
  <filename>{}</filename>
  <path>{}</path>
  <source>
    <database>Unknown</database>
  </source>
  <size>
    <width>{}</width>
    <height>{}</height>
    <depth>{}</depth>
  </size>
  <segmented>0</segmented>
'''
XML_OBJECT = '''  <object>
    <name>{}</name>
    <pose>Unspecified</pose>
    <truncated>0</truncated>
    <difficult>0</difficult>
    <bndbox>
      <xmin>{}</xmin>
      <ymin>{}</ymin>
      <xmax>{}</xmax>
      <ymax>{}</ymax>
    </bndbox>
  </object>
'''
XML_CLOSURE = '</annotation>'
    
def randColor():
    return tuple(np.random.randint(0,255,(3)))

def randInt(min, max):
    return np.random.randint(min,max)

def imshow(imgs):
    i = 0
    for img in imgs:
        cv2.imshow(str(i), img)    
        i += 1
        
    while cv2.waitKey(1) != ord('q'):
        pass
    cv2.destroyAllWindows()
    
def holo(image):
      image = 255 - image
      image = cv2.morphologyEx(image, cv2.MORPH_GRADIENT, 
                      cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2,2)), iterations=2)
      image = 255 - image
      return image
    
def transform_character(letter):
    _, thresh = cv2.threshold(cv2.cvtColor(letter, cv2.COLOR_BGR2GRAY), 240,255,cv2.THRESH_BINARY_INV)  
    letter[thresh == 0] = 0
    letter = imutils.rotate_bound(letter, randInt(-20, 20))
    letter[letter < 120] = 255
    _, thresh = cv2.threshold(cv2.cvtColor(letter, cv2.COLOR_BGR2GRAY), 240,255,cv2.THRESH_BINARY_INV)    
    cnt1,_ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(cnt1) == 0:        
        letter = np.ones((1,1))
        return letter,0
    
    x,y,w,h = cv2.boundingRect(cnt1[0])
    letter = letter[y:y+h, x:x+w] 
    
    return letter,len(cnt1)

def generate_letter(char, LETTER_WIDTH):
    letter = np.ones((1,1))
    while letter.shape[0] < 30 and  letter.shape[1] < 30:
        captcha_generator = ImageCaptcha(LETTER_WIDTH,HEIGHT, fonts=[fonts[randInt(0,len(fonts) - 1)]],
                                         font_sizes=((i for i in range(int(HEIGHT*0.7), int(HEIGHT*0.9), int(HEIGHT*0.05)))))
        letter = np.copy(captcha_generator.create_captcha_image(char,randColor(),WHITE))
        letter, num_cnt = transform_character(letter)
        if num_cnt == 0:
            continue
    
    # if randInt(1,3) == 1:
    #     letter = np.uint8(holo(letter))
    
    return letter
    

def annotate_captcha(text, filename, path, w, h, d, contours):
    xml = ''
    xml += XML_DATA.format(filename, path + filename, w, h, d)

    for i in range(len(text)):
        xmin,ymin,xmax,ymax = contours[i]
        xml += XML_OBJECT.format(text[i], xmin,ymin,xmax,ymax)
        i += 1
        
    xml += XML_CLOSURE
    return xml

def generate_captcha(text, index, path):
    FIRST_X = randInt(20,400)
    LETTER_WIDTH = randInt(40, 50)
    OFFSET = randInt(15, 18)
    
    background_generator = ImageCaptcha(WIDTH,HEIGHT)
          
    letter_size_array = []
    letter_1 = generate_letter(text[0], LETTER_WIDTH)
    letter_2 = generate_letter(text[1], LETTER_WIDTH)
    letter_3 = generate_letter(text[2], LETTER_WIDTH)
    letter_4 = generate_letter(text[3], LETTER_WIDTH)
    
    letter_size_array.append(list(letter_1.shape[:-1]))
    letter_size_array.append(list(letter_2.shape[:-1]))
    letter_size_array.append(list(letter_3.shape[:-1]))
    letter_size_array.append(list(letter_4.shape[:-1]))
    
    background = np.ones((HEIGHT,WIDTH,3), dtype=np.uint8)*255
    noise = Image.fromarray(np.ones((HEIGHT,WIDTH,3), dtype=np.uint8)*255)
    dots = Image.fromarray(np.ones((HEIGHT,WIDTH,3), dtype=np.uint8)*255)
    
    for i in range(10):
        noise = background_generator.create_noise_curve(noise, randColor())
        background_generator.create_noise_dots(noise, randColor(), width=randInt(2,8), number=4)
        
    for i in range(60):
        background_generator.create_noise_dots(dots, randColor(), width=randInt(2,16), number=randInt(2,16))
        
    letters = [letter_1,letter_2,letter_3,letter_4]
    
    last_letter_size = (0,0)
    
    for i in range(0,4):
        max_distance = 20
        min_distance = 10
        distance = randInt(min_distance,max_distance)/100    
        if(i==0):
            x = randInt(FIRST_X,FIRST_X + OFFSET)
        else:
            x = randInt(x + OFFSET + int(last_letter_size[1]*distance),x + 2*OFFSET + int(last_letter_size[1]*distance))
            
        if letter_size_array[i][0] >= HEIGHT:
            letters[i] = cv2.resize(letter_2, (letter_size_array[i][1], HEIGHT - 2))
            letter_size_array[i][0] = HEIGHT - 2
            
        y = randInt(0,HEIGHT - letter_size_array[i][0])
        
        if background[y:y + letter_size_array[i][0],x:x+letter_size_array[i][1]].shape != letters[i].shape:
            letters[i] = cv2.resize(letters[i],background[y:y + letter_size_array[i][0],x:x+letter_size_array[i][1]].shape[:-1])
            
        background[y:y + letter_size_array[i][0],   x:x+letter_size_array[i][1]] = \
        cv2.bitwise_and(background[y:y + letter_size_array[i][0],   x:x+letter_size_array[i][1]],letters[i])
        last_letter_size = letter_size_array[i]
        letter_size_array[i] = (x,y,x+letter_size_array[i][1],y + letter_size_array[i][0])
            
    # for letter_size in letter_size_array:
    #     cv2.rectangle(background, letter_size[:-2], letter_size[-2:], (0,0, 255), 3)


    _,thresh = cv2.threshold(cv2.cvtColor(background, cv2.COLOR_BGR2GRAY),240,255,cv2.THRESH_BINARY_INV)
    contours,_ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    xml = annotate_captcha(text, '{}.png'.format(index), path, WIDTH, HEIGHT, 3, letter_size_array)
        
    edges = cv2.Canny(background,100,200)
    edges = cv2.dilate(edges, (2,2), iterations=2)
    edges = cv2.merge((edges,edges,edges))
    background[edges ==255] = 0
    # background[background == 0] = 255
    background[background == 255] = np.uint8(dots)[background == 255]

    FIRST_Y = randInt(0, BACKGROUND_HEIGHT - HEIGHT)
    style = cv2.imread('backgrounds/{}.png'.format(randInt(0,99)))
    style = cv2.resize(style, (WIDTH,WIDTH))[FIRST_Y:FIRST_Y+HEIGHT,:]
    
    background[background == 255] = style[background == 255]
    
    wavy = cv2.imread('wavy.jpg')
    wavy = imutils.rotate(wavy, randInt(-120, 120))[randInt(300, 400):randInt(1000, 1100),randInt(300, 400):randInt(1000, 1100)]
    wavy = cv2.resize(wavy, (600, 100))
    _,wavy = cv2.threshold(cv2.cvtColor(wavy, cv2.COLOR_BGR2GRAY),180,255,cv2.THRESH_BINARY)
    
    background[wavy == 0] = 0
    
    
    
    return background, xml

#%%
index = 0
path = '.'
while cv2.waitKey(1) != ord('q'):
    if index == 10:
        break
    text = texts[randInt(0,len(texts))]
    cap,_ = generate_captcha(text, 0, '.')
    cv2.imwrite('{}.png'.format(index),cap)
    # cv2.imwrite('mask.png',mask)
    # cv2.imshow('a', cap)    
    index += 1
cv2.destroyAllWindows()


#%%

texts = pd.read_csv('images/Modified_Captcha/validation_annotation.csv')['text']
path = 'images/Modified_Captcha/validation/'

index = 0
for text in texts:
    cap, xml = generate_captcha(text, index, path);
    cv2.imwrite('{}{}.png'.format(path,index), cap)
    
    f = open('{}{}.xml'.format(path,index), 'w')
    f.write(xml)
    f.close()
    
    print('{:.2f}% -> {}'.format(100*index/len(texts),index))
    index += 1
#    
#    cv2.imshow('cap', cap)
#    if cv2.waitKey(200) == ord('q'):
#        break

cv2.destroyAllWindows()
#%%

texts = pd.read_csv('images/Modified_Captcha/train_annotation.csv')['text']
path = 'images/Modified_Captcha/train/'

index = 0
texts = texts[index:]

for text in texts:
    cap, xml = generate_captcha(text, index, path);
    cv2.imwrite('{}{}.png'.format(path,index), cap)
    
    f = open('{}{}.xml'.format(path,index), 'w')
    f.write(xml)
    f.close()
        
    print('{:.2f}% -> {}'.format(100*index/len(texts),index))
    index += 1
    
#    cv2.imshow('cap', cap)
#    if cv2.waitKey(200) == ord('q'):
#        break

cv2.destroyAllWindows()
#%%

texts = pd.read_csv('images/Modified_Captcha/test_annotation.csv')['text']
path = 'images/Modified_Captcha/test/'

index = 0
texts = texts[index:]

for text in texts:
    cap, xml = generate_captcha(text, index, path);
    cv2.imwrite('{}{}.png'.format(path,index), cap)
    
    f = open('{}{}.xml'.format(path,index), 'w')
    f.write(xml)
    f.close()
        
    print('{:.2f}% -> {}'.format(100*index/len(texts),index))
    index += 1
    
#    cv2.imshow('cap', cap)
#    if cv2.waitKey(200) == ord('q'):
#        break

cv2.destroyAllWindows()