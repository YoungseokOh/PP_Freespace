# Copyright Nextchip 2021. All rights reserved.
import os
import cv2
import numpy as np
import utils
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import open3d as o3d
from PIL import Image

file_manager = utils.file_manager()

def main():
    alpha = 1.0 # addweight
    home_path = 'Y:/monodepth_results/Monodepth_results_1110/frontview_test'
    depth_path = home_path + os.path.join('/', 'disp_640x192')
    file_list = utils.read_folder_list(file_manager.stixel_ori_path)
    font = cv2.FONT_HERSHEY_SIMPLEX # put text
    if not utils.check_exist(file_manager.stixel_3_save_path):
        utils.make_folder(file_manager.stixel_3_save_path)
    for i in file_list:
        save_name = utils.change_ext(file_manager.stixel_3_save_path + os.path.join('/', i), '.jpg')
        depth = cv2.imread(file_manager.dep_path + os.path.join('/', i), 1)
        seg = cv2.imread(file_manager.seg_white + os.path.join('/', i), 1)
        point_cloud_data = np.loadtxt(utils.change_ext(file_manager.point_cloud_path + os.path.join('/', i), '.txt'))
        stixel_freeroad = cv2.imread(file_manager.stixel_freeroad_path + os.path.join('/', i), 1)
        stixel_upgrade = cv2.imread(file_manager.stixel_upgrade_path + os.path.join('/', i), 1)
        stixel_ori = cv2.imread(utils.change_ext(file_manager.stixel_ori_path + os.path.join('/', i), '.jpg'), 1)
        stixel_free = cv2.imread(utils.change_ext(file_manager.stixel_free_path + os.path.join('/', i), '.jpg'), 1)
        seg = utils.img_resize(seg, 640, 192)
        stixel_ori = utils.img_resize(stixel_ori, 640, 192)
        stixel_free = utils.img_resize(stixel_free, 640, 192)
        point_cloud_xyz = point_cloud_data[:, :3]
        point_cloud_rgb = point_cloud_data[:, 3:]
        fig = plt.figure(figsize=(8,6))
        ax = Axes3D(fig)
        ax.scatter(point_cloud_xyz[:, 1], point_cloud_xyz[:, 2], point_cloud_xyz[:, 0],
                   c=point_cloud_rgb / 255, s=1.0)
        ax.view_init(-180, -90)
        plt.show()
        # cv2.putText(depth, "Depth", (570, 20), font, 0.5, (255, 255, 255), 1)
        # cv2.putText(stixel_ori, "Stixel_original", (520, 20), font, 0.5, (255, 255, 255), 1)
        # cv2.putText(stixel_freeroad, "Stixel_w_roadmodel", (470, 20), font, 0.5, (255, 255, 255), 1)
        # cv2.putText(stixel_free, "Stixel_w_freespace", (480, 20), font, 0.5, (255, 255, 255), 1)
        # cv2.putText(seg, "Freespace", (550, 20), font, 0.5, (255, 255, 255), 1)
        # add_v = np.vstack([depth, stixel_ori, stixel_freeroad, stixel_free, seg])

        # upgrade show
        cv2.putText(stixel_free, "Before", (570, 20), font, 0.5, (255, 255, 255), 1)
        cv2.putText(stixel_upgrade, "After", (580, 20), font, 0.5, (255, 255, 255), 1)
        cv2.putText(stixel_freeroad, "Final", (580, 20), font, 0.5, (255, 255, 255), 1)
        versus_v = np.vstack([stixel_free, stixel_upgrade, stixel_freeroad])
        cv2.imshow('show', versus_v)
        cv2.imwrite(save_name, versus_v)
        cv2.waitKey(1)
    # seg = utils.road_pp_seg(seg)
    print('Hello world!')

if __name__ == "__main__":
    main()