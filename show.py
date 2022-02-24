# Copyright Nextchip 2021. All rights reserved.
import os
import cv2
import matplotlib as mpl
import numpy as np
import utils
import matplotlib.pyplot as plt
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D

file_manager = utils.file_manager()


def main():
    file_list = utils.read_folder_list(file_manager.img_path)
    if not utils.check_exist(file_manager.stixel_3_save_path):
        utils.make_folder(file_manager.stixel_3_save_path)
    if not utils.check_exist(file_manager.pc3D_results_path):
        utils.make_folder(file_manager.pc3D_results_path)
    rotation_count = 0
    azim = file_manager.view_angle[1]
    fig = plt.figure(figsize=(10, 12))
    for i in file_list:
        # The name for saving
        save_name = utils.change_ext(file_manager.stixel_3_save_path + os.path.join('/', i), '.jpg')
        save_name_pc = utils.change_ext(file_manager.pc3D_results_path + os.path.join('/', i), '.jpg')
        save_3d_folder = file_manager.pc3D_results_path + os.path.join('/', i[:-4])
        if not utils.check_exist(save_3d_folder):
            utils.make_folder(save_3d_folder)
        # Image
        img_color = cv2.imread(utils.change_ext(file_manager.img_path + os.path.join('/', i), '.jpg'), cv2.IMREAD_COLOR)
        img_color = cv2.resize(img_color, (213, 192))
        img_b, img_g, img_r = cv2.split(img_color)
        # Disparity
        disparity = cv2.imread(file_manager.disp_path + os.path.join('/', i), 1)
        disparity_pc = cv2.imread(file_manager.disp_path + os.path.join('/', i), 0)
        disp_color = cv2.resize(disparity_pc, (213, 192))
        # Depth
        # Final lower
        final_lower = utils.read_lower(utils.change_ext(file_manager.final_fs_path + os.path.join('/', i), '.csv'))
        seg = cv2.imread(file_manager.seg_white + os.path.join('/', i), 1)
        point_cloud_data = np.loadtxt(utils.change_ext(file_manager.point_cloud_path + os.path.join('/', i), '.txt'))
        stixel_freeroad = cv2.imread(file_manager.stixel_freeroad_path + os.path.join('/', i), 1)
        # stixel_prev_results = cv2.imread(file_manager.stixel_previous_results_path + os.path.join('/', i), 1)
        stixel_upgrade = cv2.imread(file_manager.stixel_upgrade_path + os.path.join('/', i), 1)
        stixel_ori = cv2.imread(utils.change_ext(file_manager.stixel_ori_path + os.path.join('/', i), '.jpg'), 1)
        stixel_free = cv2.imread(utils.change_ext(file_manager.stixel_free_path + os.path.join('/', i), '.jpg'), 1)
        seg_depth = cv2.imread(utils.change_ext(file_manager.seg_free_depth + os.path.join('/', i), '.jpg'), 1)
        show_final_lower = utils.showing_lower(final_lower)
        show_final_lower = utils.img_resize(show_final_lower, 640, 192)
        utils.data_save(file_manager.final_fs_path, i, show_final_lower)
        show_final_lower = cv2.imread(utils.change_ext(file_manager.final_fs_path + os.path.join('/', i), '.jpg'), 1)
        ### Outlier ###
        # outlier_path = utils.change_ext(file_manager.distance_path + os.path.join('/', i), '.txt')
        # f = open(outlier_path)
        # file_size = os.path.getsize(outlier_path)
        # outlier_coord = []
        # if file_size == 0:
        #     pass
        # else:
        #     while True:
        #         line = f.readline()
        #         if not line: break
        #         outlier_coord.append(line[-8:-1])
        #     f.close()
        #     count = 0
        #     for count in range(0, len(outlier_coord)):
        #         if outlier_coord[count] == '':
        #             break
        #         split_coord = outlier_coord[count].split()
        #         show_final_lower = cv2.line(show_final_lower, (int(split_coord[0]) * 3, int(split_coord[1])),
        #              (int(split_coord[0]) * 3, int(split_coord[1])), (255, 255, 0))
        #     # cv spot
        #     print('read done')
        # seg = utils.img_resize(seg, 640, 192)
        # stixel_ori = utils.img_resize(stixel_ori, 640, 192)
        # stixel_free = utils.img_resize(stixel_free, 640, 192)

        # Save the Point Cloud
        # fig_xbar = plt.figure(figsize=(10, 12))
        # for x in tqdm(range(0, img_color.shape[1]), desc='Saving columns', leave=False):
        #     point_cloud_xbar = utils.create_pointcloud(img_color, disp_color, x, final_lower)
        #     point_cloud_xbar = np.asarray(point_cloud_xbar)
        #     point_cloud_xbar_xyz = point_cloud_xbar[:, :3]
        #     x_columns_csv_name = utils.change_ext(save_3d_folder + os.path.join('/%010d' %x), '.npy')
        #     np.save(x_columns_csv_name, point_cloud_xbar_xyz)
        # print('{} columns is saved!'.format(i))

        # pure freespace
        pure_freespace = cv2.imread(
            os.path.join(file_manager.stixel_freeroad_path, 'pure_freespace_results') + os.path.join('/', i), 1)

        # pure stixel
        pure_stixel = cv2.imread(
            os.path.join(file_manager.stixel_freeroad_path, 'pure_stixel_results') + os.path.join('/', i), 1)

        # stixel+freespace
        sti_free_xel = cv2.imread(
            os.path.join(file_manager.stixel_freeroad_path, 'stixel+freespace_results') + os.path.join('/', i), 1)

        freespace_error_results = cv2.imread(
            os.path.join(file_manager.stixel_freeroad_path, 'stixel+freespace_results') + os.path.join('/', i), 1)

        point_cloud_beta = utils.create_pointcloud(img_color, disp_color, None, None)
        point_cloud_data = np.asarray(point_cloud_beta)
        point_cloud_xyz = point_cloud_data[:, :3]
        point_cloud_rgb = point_cloud_data[:, 3:]
        ax = Axes3D(fig, auto_add_to_figure=True)
        ax.scatter(point_cloud_xyz[:, 0], point_cloud_xyz[:, 1], point_cloud_xyz[:, 2],
                   c=point_cloud_rgb / 255, s=2.0, cmap='plasma')
        # Original
        # ax.scatter(point_cloud_xyz[:, 0], point_cloud_xyz[:, 1], point_cloud_xyz[:, 2],
        #            c=point_cloud_xyz[:, 2], s=2.0, cmap='plasma')
        ax.view_init(file_manager.view_angle[0], azim)
        # if not rotation_count <= 50:
        #     azim = azim - file_manager.rotation[0]
        # else:
        #     azim = azim + file_manager.rotation[1]
        # if rotation_count == 100:
        #     rotation_count = 0
        # rotation_count += 1
        # plt.show()
        # plt.tight_layout()
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
        cv2.putText(disparity, "Depth", (580, 20), file_manager.font, 0.5, (255, 255, 255), 1)
        # cv2.putText(stixel_prev_results, "Before", (580, 20), file_manager.font, 0.5, (255, 255, 255), 1)
        cv2.putText(stixel_freeroad, "Stixel + Freespace Fusion", (480, 20), file_manager.font, 0.5, (255, 255, 255), 1)
        cv2.putText(stixel_free, "Only Feespace", (520, 20), file_manager.font, 0.5, (255, 255, 255), 1)
        # cv2.putText(pure_freespace, "Only freespace", (510, 20), file_manager.font, 0.5, (255, 255, 255), 1)
        # cv2.putText(pure_stixel, "Only Stixel", (550, 20), file_manager.font, 0.5, (255, 255, 255), 1)
        # cv2.putText(sti_free_xel, "Stixel + Freespace", (480, 20), file_manager.font, 0.5, (255, 255, 255), 1)
        versus_v = np.vstack([disparity, stixel_free, stixel_freeroad])
        point_cloud_fig = utils.img_resize(point_cloud_fig, versus_v.shape[1], versus_v.shape[0])
        versus_h = np.hstack([versus_v, point_cloud_fig])
        cv2.imshow('show', versus_h)
        cv2.imwrite(save_name, versus_h)
        cv2.waitKey(1)
    # seg = utils.road_pp_seg(seg)
    print('Hello world!')

if __name__ == "__main__":
    main()
