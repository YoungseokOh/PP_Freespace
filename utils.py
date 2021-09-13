# Copyright Nextchip 2021. All rights reserved.
import csv
import os, random
import numpy as np
import pandas as pd
import cv2


def readlines(filename):
    """Read all the lines in a text file and return as a list
    """
    with open(filename, 'r') as f:
        lines = f.read().splitlines()
    return lines


def write_csv(list, path):
    path = change_ext(path, '.csv')
    with open(path, 'w', newline='') as f:
        write = csv.writer((f))
        write.writerow(list)
    return True


def read_folder_list(path):
    folder_list = os.listdir(path)
    return folder_list


def check_exist(path):
    return os.path.exists(path)


def make_folder(path):
    return os.makedirs(path)


def check_folder(path):
    if not check_exist(path):
        make_folder(path)
    return True


def rand_img(path):
    return random.choice([file for file in os.listdir(path) if file.endswith('.png') or file.endswith('.jpg')])


def img_read(path):
    return cv2.imread(path, 0)


def img_resize(img, width, height):
    return cv2.resize(img, [width, height])


def road_pp_seg(seg):
    height, width = seg.shape
    for w in range(0, width):
        road_flag = 1
        for h in range(0, height):
            if seg[h][w] == 1:
                pass
                if road_flag == 0:
                    seg[h][w] = 0
                    continue
            elif seg[h][w] == 0:
                road_flag = 0
    return seg


def showing_seg(seg):
    seg = road_pp_seg(seg)
    height, width = seg.shape
    for w in range(0, width):
        for h in range(0, height):
            if seg[h][w] == 1:
                seg[h][w] = 0
                pass
            elif seg[h][w] == 0:
                seg[h][w] = 255
                pass
    return seg


def change_ext(path, ext):
    path, ext_b = os.path.splitext(path)
    return path + ext


def get_lowerpath(seg):
    lowerpath = []
    height, width = seg.shape
    for w in range(0, width):
        for h in range(0, height):
            if seg[h][w] == 0:
                lowerpath.append(h)
                break
    return lowerpath