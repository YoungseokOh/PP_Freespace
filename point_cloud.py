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
        disp_color = cv2.imread(utils.change_ext(file_manager.dep_path + os.path.join('/', i), '.jpg'))
        disp_color = cv2.resize(disp_color, (213, 192))
        final_lower = utils.read_lower(utils.change_ext(file_manager.final_fs_path + os.path.join('/', i), '.csv'))
        disp_x, disp_y, disp_z = cv2.split(disp_color)
        # Point cloud data saved 256x80\
        if not os.path.exists(point_cloud_txt) or os.path.getsize(point_cloud_txt) == 0:
            for y in range(0, 213):
                for x in range(final_lower[y], 192):
                    point_cloud_dataname = "%d %d %d %d %d %d\n" %(x, y, disp_z[x][y], img_r[x][y], img_g[x][y], img_b[x][y])
                    k.write(point_cloud_dataname)
                    #print(point_cloud_dataname)
            k.close()
        print('{} saving is done.'.format(point_cloud_txt))
    # All works done
print("All txt files are saved!")

if __name__ == "__main__":
    main()



