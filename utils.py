# Copyright Nextchip 2021. All rights reserved.
import csv
import os, random
import numpy as np
import pandas as pd
import cv2
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression


class file_manager:
    def __init__(self):
        # Paths
        self.ori_path = 'Y:/monodepth_results/Monodepth_results_1110/frontview_test'
        self.seg_path = self.ori_path + '/seg'
        self.seg_hood_path = self.ori_path + '/seg_hood'
        self.seg_white = self.ori_path + '/seg_white'
        self.seg_free_depth = self.ori_path + '/seg_free_depth'
        self.img_path = self.ori_path + '/test_images'
        self.disp_path = self.ori_path + '/disp_640x192'
        self.depth_path = self.ori_path + '/depth'
        self.lower_save_path = self.ori_path + '/lower_path'
        self.final_fs_path = self.ori_path + '/final_fs_results'
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
        self.view_angle = [360, 450]
        # self.view_angle = [360, 450] # rotation left


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


def read_lower(path):
    data = pd.read_csv(path, header=None)
    data = data.iloc[:, 0].tolist()
    return data


def data_save(path, name, data):
    save_name = change_ext(path + os.path.join('/', name), '.jpg')
    cv2.imwrite(save_name, data)
    return True


def min_max_normalize(lst):
    normalized = []
    for value in lst:
        normalized_num = (value - np.min(lst)) / (np.max(lst) - np.min(lst))
        normalized.append(normalized_num)

    return normalized


def z_score_normalize(lst):
    normalized = []
    for value in lst:
        normalized_num = (value - np.mean(lst)) / np.std(lst)
        normalized.append(normalized_num)
    return normalized


def create_pointcloud(img, disp_color, x_count=None, final_lower=None):
    img_b, img_g, img_r = cv2.split(img)
    point_cloud = []
    if not x_count is None:
        if not final_lower is None:
            for x in range(x_count, x_count+1):
                for y in range(final_lower[x], img.shape[0]):
                    point_cloud.append((y, x, disp_color[y][x], img_r[y][x], img_g[y][x], img_b[y][x]))
        else:
            return True
    else:
        if not final_lower is None:
            for x in range(0, img.shape[1]):
                for y in range(final_lower[x], img.shape[0]):
                    point_cloud.append((y, x, disp_color[y][x], img_r[y][x], img_g[y][x], img_b[y][x]))
        else:
            for x in range(0, img.shape[1]):
                for y in range(0, img.shape[0]):
                    point_cloud.append((y, x, disp_color[y][x], img_r[y][x], img_g[y][x], img_b[y][x]))
    return point_cloud


def poly_feature(df_feature, degree=2):
    poly_reg = PolynomialFeatures(degree=degree)
    X_poly = poly_reg.fit_transform(df_feature)
    return X_poly, poly_reg


def set_regression(depth_list, depth_name_list, degree=2):
    df_depth_reggresion = pd.DataFrame(depth_list, depth_name_list).T.melt().dropna(axis=0)
    df_feature, poly_leg = poly_feature(df_depth_reggresion['value'].values.reshape(-1, 1), degree)
    lin_reg_2 = LinearRegression()
    # lin_reg_2 = Ridge()
    lin_reg_2.fit(df_feature, df_depth_reggresion['variable'].values)
    # lin_reg_2 = RANSACRegressor(random_state=0).fit(df_feature, df_depth_reggresion['variable'].values)
    predict_Y = lin_reg_2.predict(poly_leg.fit_transform(df_depth_reggresion['value'].values.reshape(-1, 1)))
    df_depth_reggresion['regression'] = lin_reg_2.predict(
        poly_leg.fit_transform(df_depth_reggresion['value'].values.reshape(-1, 1)))
    return df_depth_reggresion
