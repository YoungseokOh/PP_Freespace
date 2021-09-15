# Copyright Nextchip 2021. All rights reserved.

import os
import cv2
import numpy as np
import utils


def main():
    seg_path = 'Y:/monodepth_results/Monodepth_results_1110/frontview_test/seg'
    seg_white = 'Y:/monodepth_results/Monodepth_results_1110/frontview_test/seg_white'
    seg_free_depth = 'Y:/monodepth_results/Monodepth_results_1110/frontview_test/seg_free_depth'
    img_path = 'Y:/monodepth_results/Monodepth_results_1110/frontview_test/test_images'
    dep_path = 'Y:/monodepth_results/Monodepth_results_1110/frontview_test/disp_640x192'
    lower_save_path = 'Y:/monodepth_results/Monodepth_results_1110/frontview_test/lower_path'
    alpha = 1.0
    file_list = utils.read_folder_list(seg_path)
    for i in file_list:
        save_name = utils.change_ext(seg_free_depth + os.path.join('/', i), '.jpg')
        seg = cv2.imread(seg_path + os.path.join('/', i), 0)
        img = cv2.imread(utils.change_ext(img_path + os.path.join('/', i), '.jpg'), 0)
        dep = cv2.imread(utils.change_ext(dep_path + os.path.join('/', i), '.jpg'), 0)
        img = utils.img_resize(img, 640, 192)
        seg = utils.img_resize(seg, 640, 192)
        lower = utils.get_lowerpath(seg)
        # utils.write_csv(lower, lower_save_path + os.path.join('/', i))
        dep = utils.img_resize(dep, 640, 192)
        seg = utils.showing_seg(seg)
        dep_free = utils.depth_to_freespace(seg, dep)
        dst = cv2.addWeighted(seg, alpha, dep, (1-alpha), 0)
        cv2.imshow('show', dep_free)
        cv2.imwrite(save_name, dep_free)
        cv2.waitKey(1)
    # seg = utils.road_pp_seg(seg)

    print('Hello world!')

if __name__ == "__main__":
    main()