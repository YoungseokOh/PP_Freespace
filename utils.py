# Copyright Nextchip 2021. All rights reserved.
import csv
import os, random
import numpy as np
import pandas as pd
import cv2

class file_manager:
    def __init__(self):
        # Paths
        self.ori_path = 'Y:/monodepth_results/Monodepth_results_1110/frontview_test'
        self.seg_path = self.ori_path + '/seg'
        self.seg_hood_path = self.ori_path + '/seg_hood'
        self.seg_white = self.ori_path + '/seg_white'
        self.seg_free_depth = self.ori_path + '/seg_free_depth'
        self.img_path = self.ori_path + '/test_images'
        self.dep_path = self.ori_path + '/disp_640x192'
        self.lower_save_path = self.ori_path + '/lower_path'
        self.point_cloud_path = self.ori_path + '/point_cloud'
        self.pc3D_results_path = self.ori_path + '/3d_results'
        self.stixel_ori_path = self.ori_path + '/stixel_results_ori'
        self.stixel_free_path = self.ori_path + '/stixel_results_freespace'
        self.stixel_freeroad_path = self.ori_path + '/stixel_results_real_final'
        self.stixel_upgrade_path = self.ori_path + '/stixel_results_upgrade'
        self.stixel_3_save_path = self.ori_path + '/stixel_3_results'
        # settings
        self.alpha = 1.0
        self.font = cv2.FONT_HERSHEY_SIMPLEX # put text
        self.rotation = [-1, 1]
        self.view_angle = [-185, -90]


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


def depth_to_freespace(seg, dep):
    height, width = seg.shape
    for w in range(0, width):
        for h in range(0, height):
            if seg[h][w] == 0:
                seg[h][w] = 0
                pass
            elif seg[h][w] == 255:
                seg[h][w] = dep[h][w]
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
            if not seg[h][w] == 0:
                lowerpath.append(h)
                break
            if h == 191:
                lowerpath.append(h)
    return lowerpath


def showing_lower(lower):
    seg_for_lower_test = np.zeros((192, 213))
    height, width = seg_for_lower_test.shape
    for w in range(0, width):
        for h in range(0, height):
            if h == lower[w]:
                seg_for_lower_test[h][w] = lower[w]
                break
    return seg_for_lower_test


def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1
