# Copyright Nextchip 2021. All rights reserved.

import os
import cv2
import numpy as np
import utils


def main():
    seg_path = 'Y:/monodepth_results/Monodepth_results_1110/frontview_test/seg'
    img_path = 'Y:/monodepth_results/Monodepth_results_1110/frontview_test/test_images'
    dep_path = 'Y:/monodepth_results/Monodepth_results_1110/frontview_test/disp_640x192'
    seg_save_path = 'Y:/monodepth_results/Monodepth_results_1110/frontview_test/seg'
    alpha = 0.3
    file_list = utils.read_folder_list(seg_path)
    for i in file_list:
        seg = cv2.imread(seg_path + os.path.join('/', i), 0)
        img = cv2.imread(utils.change_ext(img_path + os.path.join('/', i), '.jpg'), 0)
        dep = cv2.imread(utils.change_ext(dep_path + os.path.join('/', i), '.jpg'), 0)
        img = utils.img_resize(img, 213, 192)
        seg = utils.img_resize(seg, 213, 192)
        lowerpath = utils.get_lowerpath(seg)

        dep = utils.img_resize(dep, 312, 192)
        seg = utils.showing_seg(seg)
        dst = cv2.addWeighted(seg, alpha, dep, (1-alpha), 0)
        cv2.imshow('show', dst)
        cv2.waitKey(1)
    # seg = utils.road_pp_seg(seg)

    print('Hello world!')

if __name__ == "__main__":
    main()