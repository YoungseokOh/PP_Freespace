# Copyright NEXTCHIP Co. Ltd. All rights reserved.
import cv2
import errno
import os
import utils
from tqdm import tqdm

db_move_path = 'Y:/monodepth_results/Monodepth_results_1110/frontview_parking_test/seg'
save_path = 'Y:/monodepth_results/Monodepth_results_1110/frontview_parking_test/seg_hood'
move_folder_list = utils.read_folder_list(db_move_path)
count = 0
folder_path = os.path.join(db_move_path)
file_list = utils.read_folder_list(folder_path)
# for n in tqdm(move_folder_list):
#     folder_path = os.path.join(db_move_path, n)
#     file_list = utils.read_folder_list(folder_path)
for i in tqdm(file_list):
    if not os.path.splitext(i)[1] == '.png':
        continue
    filename = os.path.join(folder_path, i)
    image = cv2.imread(filename)
    image = utils.img_resize(image, 1920, 1080)
    # Crop  car-hood
    dst = image.copy()
    roi = image[0:800, 0:1920]
    image_height = 800
    image_width = 1920
    image = cv2.resize(roi, (image_width, image_height))
    image = utils.img_resize(image, 320, 192)
    cv2.imwrite(save_path + '/' + i, image)
    count += 1


