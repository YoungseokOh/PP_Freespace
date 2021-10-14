# Copyright Nextchip 2021. All rights reserved.
import os
import cv2
import numpy as np
import utils
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

file_manager = utils.file_manager()

def main():
    file_list = utils.read_folder_list(file_manager.stixel_ori_path)
    if not utils.check_exist(file_manager.stixel_3_save_path):
        utils.make_folder(file_manager.stixel_3_save_path)
    if not utils.check_exist(file_manager.pc3D_results_path):
        utils.make_folder(file_manager.pc3D_results_path)
    rotation_count = 0
    azim = file_manager.view_angle[1]
    fig = plt.figure(figsize=(8, 6))
    for i in file_list:
        save_name = utils.change_ext(file_manager.stixel_3_save_path + os.path.join('/', i), '.jpg')
        save_name_pc = utils.change_ext(file_manager.pc3D_results_path + os.path.join('/', i), '.jpg')
        depth = cv2.imread(file_manager.dep_path + os.path.join('/', i), 1)
        seg = cv2.imread(file_manager.seg_white + os.path.join('/', i), 1)
        point_cloud_data = np.loadtxt(utils.change_ext(file_manager.point_cloud_path + os.path.join('/', i), '.txt'))
        stixel_freeroad = cv2.imread(file_manager.stixel_freeroad_path + os.path.join('/', i), 1)
        stixel_upgrade = cv2.imread(file_manager.stixel_upgrade_path + os.path.join('/', i), 1)
        stixel_ori = cv2.imread(utils.change_ext(file_manager.stixel_ori_path + os.path.join('/', i), '.jpg'), 1)
        stixel_free = cv2.imread(utils.change_ext(file_manager.stixel_free_path + os.path.join('/', i), '.jpg'), 1)
        seg_depth = cv2.imread(utils.change_ext(file_manager.seg_free_depth + os.path.join('/', i), '.jpg'), 1)
        final_lower = utils.read_lower(utils.change_ext(file_manager.final_fs_path + os.path.join('/', i), '.csv'))
        show_final_lower = utils.showing_lower(final_lower)
        show_final_lower = utils.img_resize(show_final_lower, 640, 192)
        utils.data_save(file_manager.final_fs_path, i, show_final_lower)
        show_final_lower = cv2.imread(utils.change_ext(file_manager.final_fs_path + os.path.join('/', i), '.jpg'), 1)
        seg = utils.img_resize(seg, 640, 192)
        stixel_ori = utils.img_resize(stixel_ori, 640, 192)
        stixel_free = utils.img_resize(stixel_free, 640, 192)
        point_cloud_xyz = point_cloud_data[:, :3]
        point_cloud_rgb = point_cloud_data[:, 3:]
        ax = Axes3D(fig)
        ax.scatter(point_cloud_xyz[:, 1], point_cloud_xyz[:, 2], point_cloud_xyz[:, 0],
                   c=point_cloud_rgb / 255, s=1.0)
        ax.view_init(file_manager.view_angle[0], azim)
        if not rotation_count <= 50:
            azim = azim - file_manager.rotation[0]
        else:
            azim = azim - file_manager.rotation[1]
        if rotation_count == 100:
            rotation_count = 0
        rotation_count += 1
        plt.show()
        plt.tight_layout()
        fig.savefig(save_name_pc)
        plt.clf()

        point_cloud_fig = cv2.imread(save_name_pc)
        # cv2.putText(depth, "Depth", (570, 20), font, 0.5, (255, 255, 255), 1)
        # cv2.putText(stixel_ori, "Stixel_original", (520, 20), font, 0.5, (255, 255, 255), 1)
        # cv2.putText(stixel_freeroad, "Stixel_w_roadmodel", (470, 20), font, 0.5, (255, 255, 255), 1)
        # cv2.putText(stixel_free, "Stixel_w_freespace", (480, 20), font, 0.5, (255, 255, 255), 1)
        # cv2.putText(seg, "Freespace", (550, 20), font, 0.5, (255, 255, 255), 1)
        # add_v = np.vstack([depth, stixel_ori, stixel_freeroad, stixel_free, seg])

        # upgrade show
        # cv2.putText(stixel_free, "Before", (570, 20), file_manager.font, 0.5, (255, 255, 255), 1)
        cv2.putText(depth, "Depth", (580, 20), file_manager.font, 0.5, (255, 255, 255), 1)
        cv2.putText(stixel_freeroad, "Final", (580, 20), file_manager.font, 0.5, (255, 255, 255), 1)
        cv2.putText(show_final_lower, "Freespace", (550, 20), file_manager.font, 0.5, (255, 255, 255), 1)
        versus_v = np.vstack([depth, stixel_freeroad, show_final_lower])
        point_cloud_fig = utils.img_resize(point_cloud_fig, versus_v.shape[1], versus_v.shape[0])
        versus_h = np.hstack([versus_v, point_cloud_fig])
        cv2.imshow('show', versus_h)
        cv2.imwrite(save_name, versus_h)
        cv2.waitKey(1)
    # seg = utils.road_pp_seg(seg)
    print('Hello world!')

if __name__ == "__main__":
    main()