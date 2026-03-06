#!/usr/bin/env python

# All of these functions written originally by K. Nelson for segment
#
# 2016-03: Adapted for run2dmorph and OpenCV image format by A. Hsiang

import glob
import os
import time
import cv2
import numpy as np
from PIL import Image
import math
import tifffile  # Ensure tifffile is imported for handling .tif files

def list_files(directory, file_extension):
    '''
    This function takes a directory and returns a list of all images in that
    directory with extension specified by user.
    '''

    file_list = glob.glob(directory + os.sep + "*" + file_extension)

    print(f'INFO: images.find() found {len(file_list)} files\n')

    return sorted(file_list)


def load(filename, run):
    '''
    Load image from given path and resizes it.
    '''

    start = time.time()

    try:
        img = cv2.imread(filename)  # Image is in BGR
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        run['bigtiff'] = False
    except IOError:
        if run['input_ext'] == 'tif':
            print("File may be a BIGtiff, attempting alternate read...")
            img = tifffile.imread(filename)
            run['bigtiff'] = True
        else:
            raise

    end = time.time()
    print(f'INFO: images.load() processed {filename} ( {end - start:.6f} seconds)')

    if (run['pixel_size_x'] != run['pixel_size_y']) or run['pixel_size_x'] != 1 or run['pixel_size_y'] != 1:
        img = resize(img, run)

    return img


def resize(image, run):
    '''
    Scales image (cv2 format) based on units per pixel
    Only works on non-bigtiffs
    '''
    print('INFO: Resizing image...')
    height, width, _ = image.shape

    # Calculate the new dimensions
    m_resized = int(math.ceil(run['pixel_size_x'] * width))
    n_resized = int(math.ceil(run['pixel_size_y'] * height))

    resized = cv2.resize(image, (m_resized, n_resized))

    return resized


def save(image, filename, tags=''):
    if isinstance(image, (np.ndarray, np.generic)):
        image = Image.fromarray(image)
    if tags:
        image.save(filename, tiffinfo=tags)
    else:
        image.save(filename)
