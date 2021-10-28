# Copyright Nextchip 2021. All rights reserved.

import matplotlib
matplotlib.use('TKAgg')
import os
import cv2
import utils

file_manager = utils.file_manager()
file_list = utils.read_folder_list(file_manager.seg_path)
# Create folder
if not utils.check_exist(file_manager.point_cloud_path):
    utils.make_folder(file_manager.point_cloud_path)

def main():
    for i in file_list:
        point_cloud_txt = utils.change_ext(file_manager.point_cloud_path + os.path.join('/', i), '.txt')
        k = open(point_cloud_txt, 'w')
        # # Img read
        img_color = cv2.imread(utils.change_ext(file_manager.img_path + os.path.join('/', i), '.jpg'), cv2.IMREAD_COLOR)
        img_color = cv2.resize(img_color, (213, 192))
        img_b, img_g, img_r = cv2.split(img_color)
        disp_color = cv2.imread(utils.change_ext(file_manager.disp_path + os.path.join('/', i), '.jpg'), 0)
        depth = cv2.imread(utils.change_ext(file_manager.depth_path + os.path.join('/', i), '.png'), cv2.CV_16U)
        disp_color = cv2.resize(disp_color, (213, 192))
        depth = cv2.resize(depth, (213, 192))
        final_lower = utils.read_lower(utils.change_ext(file_manager.final_fs_path + os.path.join('/', i), '.csv'))
        # disp_x, disp_y, disp_z = cv2.split(disp_color)
        # disp_z = cv2.split(disp_color)
        # Point cloud data saved 256x80\
        if not os.path.exists(point_cloud_txt) or os.path.getsize(point_cloud_txt) == 0:
            for x in range(0, 213):
                for y in range(0, 192):
                    point_cloud_dataname = "%d %d %d %d %d %d\n" %(y, x, disp_color[y][x], img_r[y][x], img_g[y][x], img_b[y][x])
                    k.write(point_cloud_dataname)
                    #print(point_cloud_dataname)
            k.close()
        print('{} saving is done.'.format(point_cloud_txt))
    # All works done
print("All txt files are saved!")

if __name__ == "__main__":
    main()



