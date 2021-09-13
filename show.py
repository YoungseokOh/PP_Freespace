# Copyright Nextchip 2021. All rights reserved.

import os
import cv2
import numpy as np
import utils


def main():
    depth_path = 'Y:/monodepth_results/Monodepth_results_1110/frontview_test/disp_640x192'
    seg_white_path = 'Y:/monodepth_results/Monodepth_results_1110/frontview_test/seg_white'
    stixel_ori_path = 'Y:/monodepth_results/Monodepth_results_1110/frontview_test/stixel_results_ori'
    stixel_free_path = 'Y:/monodepth_results/Monodepth_results_1110/frontview_test/stixel_results_freespace'
    stixel_3_save_path = 'Y:/monodepth_results/Monodepth_results_1110/frontview_test/stixel_3_results'
    alpha = 1.0
    file_list = utils.read_folder_list(stixel_ori_path)
    for i in file_list:
        save_name = utils.change_ext(stixel_3_save_path + os.path.join('/', i), '.jpg')
        depth = cv2.imread(depth_path + os.path.join('/', i), 1)
        seg = cv2.imread(seg_white_path + os.path.join('/', i), 1)
        stixel_ori = cv2.imread(utils.change_ext(stixel_ori_path + os.path.join('/', i), '.jpg'), 1)
        stixel_free = cv2.imread(utils.change_ext(stixel_free_path + os.path.join('/', i), '.jpg'), 1)
        seg = utils.img_resize(seg, 640, 192)
        stixel_ori = utils.img_resize(stixel_ori, 640, 192)
        stixel_free = utils.img_resize(stixel_free, 640, 192)
        add_v = np.vstack([depth, stixel_ori, stixel_free, seg])
        cv2.imshow('show', add_v)
        cv2.imwrite(save_name, add_v)
        cv2.waitKey(1)
    # seg = utils.road_pp_seg(seg)

    print('Hello world!')

if __name__ == "__main__":
    main()