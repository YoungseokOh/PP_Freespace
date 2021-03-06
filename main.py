# Copyright Nextchip 2021. All rights reserved.
import os
import cv2
import numpy as np
import utils
from tqdm import tqdm

file_manager = utils.file_manager()
def main():
    f = open(file_manager.ori_path + '/' + 'filelist.txt', 'w')
    file_list = utils.read_folder_list(file_manager.seg_path)
    for i in tqdm(file_list):
        if not os.path.splitext(i)[1] == '.png':
            continue
        save_name = utils.change_ext(file_manager.seg_free_depth + os.path.join('/', i), '.png')
        seg = cv2.imread(file_manager.seg_path + os.path.join('/', i), 0)
        # seg_hood = cv2.imread(file_manager.seg_hood_path + os.path.join('/', i), 0)
        seg_height, seg_width = seg.shape
        img = cv2.imread(utils.change_ext(file_manager.img_path + os.path.join('/', i), '.png'), 0)
        dep = cv2.imread(utils.change_ext(file_manager.disp_path + os.path.join('/', i), '.png'), 0)
        img = utils.img_resize(img, 640, 192)
        seg_hood = utils.img_resize(seg, 640, 192)
        dep = utils.img_resize(dep, 640, 192)
        seg = utils.showing_seg(seg)
        seg_hood = utils.showing_seg(seg_hood)
        dep_free = utils.depth_to_freespace(seg_hood, dep)
        ''' addWeighted '''
        # dst = cv2.addWeighted(seg, alpha, dep, (1-alpha), 0)
        ''' lower path save '''
        seg_for_lower = utils.img_resize(dep_free, 213, 192)
        # lower = utils.get_lowerpath(seg_for_lower)
        lower = utils.get_lowerpath_from_bottom(seg_for_lower)
        utils.write_csv(lower, file_manager.lower_save_path + os.path.join('/', i))
        show_lower = utils.showing_lower(lower)
        file_name = '%s\n' %i
        f.write(file_name)
        cv2.imshow('show', show_lower)
        cv2.imwrite(save_name, dep_free)
        cv2.waitKey(1)
    f.close()
    # seg = utils.road_pp_seg(seg)
    print('Hello world!')

if __name__ == "__main__":
    main()