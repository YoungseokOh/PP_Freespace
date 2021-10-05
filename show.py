# Copyright Nextchip 2021. All rights reserved.

import os
import cv2
import numpy as np
import utils


def main():
    alpha = 1.0 # addweight
    home_path = 'Y:/monodepth_results/Monodepth_results_1110/frontview_test'
    depth_path = home_path + os.path.join('/', 'disp_640x192')
    seg_white_path = home_path + os.path.join('/', 'seg_white')
    stixel_ori_path = home_path + os.path.join('/', 'stixel_results_ori')
    stixel_free_path = home_path + os.path.join('/', 'stixel_results_freespace')
    stixel_freeroad_path = home_path + os.path.join('/', 'stixel_results_real_final')
    stixel_upgrade_path = home_path + os.path.join('/', 'stixel_results_upgrade')
    stixel_3_save_path = home_path + os.path.join('/', 'stixel_3_results')
    file_list = utils.read_folder_list(stixel_ori_path)
    font = cv2.FONT_HERSHEY_SIMPLEX # put text
    if not utils.check_exist(stixel_3_save_path):
        utils.make_folder(stixel_3_save_path)
    for i in file_list:
        save_name = utils.change_ext(stixel_3_save_path + os.path.join('/', i), '.jpg')
        depth = cv2.imread(depth_path + os.path.join('/', i), 1)
        seg = cv2.imread(seg_white_path + os.path.join('/', i), 1)
        stixel_freeroad = cv2.imread(stixel_freeroad_path + os.path.join('/', i), 1)
        stixel_upgrade = cv2.imread(stixel_upgrade_path + os.path.join('/', i), 1)
        stixel_ori = cv2.imread(utils.change_ext(stixel_ori_path + os.path.join('/', i), '.jpg'), 1)
        stixel_free = cv2.imread(utils.change_ext(stixel_free_path + os.path.join('/', i), '.jpg'), 1)
        seg = utils.img_resize(seg, 640, 192)
        stixel_ori = utils.img_resize(stixel_ori, 640, 192)
        stixel_free = utils.img_resize(stixel_free, 640, 192)
        cv2.putText(depth, "Depth", (570, 20), font, 0.5, (255, 255, 255), 1)
        cv2.putText(stixel_ori, "Stixel_original", (520, 20), font, 0.5, (255, 255, 255), 1)
        cv2.putText(stixel_freeroad, "Stixel_w_roadmodel", (470, 20), font, 0.5, (255, 255, 255), 1)
        cv2.putText(stixel_free, "Stixel_w_freespace", (480, 20), font, 0.5, (255, 255, 255), 1)
        cv2.putText(seg, "Freespace", (550, 20), font, 0.5, (255, 255, 255), 1)
        add_v = np.vstack([depth, stixel_ori, stixel_freeroad, stixel_free, seg])

        # cv2.putText(stixel_free, "Before", (570, 20), font, 0.5, (255, 255, 255), 1)
        # cv2.putText(stixel_upgrade, "After", (580, 20), font, 0.5, (255, 255, 255), 1)
        # versus_v = np.vstack([stixel_free, stixel_upgrade])
        cv2.imshow('show', add_v)
        cv2.imwrite(save_name, add_v)
        cv2.waitKey(1)
    # seg = utils.road_pp_seg(seg)
    print('Hello world!')

if __name__ == "__main__":
    main()