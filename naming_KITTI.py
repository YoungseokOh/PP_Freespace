import os
import cv2
import utils

# path = 'Y:/monodepth_results/Monodepth_results_1110/frontview_test/seg'
path = 'Y:/monodepth_results/Monodepth_results_1110/frontview_test/test_images'
folder_list = utils.read_folder_list(path)
for file in folder_list:
    filename, ext = os.path.splitext(file)
    filename = int(filename)
    filename = '%010d'%(filename) + ext
    os.rename(os.path.join(path, file), os.path.join(path, filename))
print('done!')