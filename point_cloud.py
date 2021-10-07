# Point Cloud Saving

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
matplotlib.use('TKAgg')
from matplotlib.backends.backend_agg import FigureCanvasAgg
from mpl_toolkits import mplot3d
import os
import errno
from os import listdir
import scipy.misc
import cv2

def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1
# main
stereo_train_n = 'stereo_train_001'
disp_txt_name = "Y:/Stereo_dataset/Apollo_disparity_list_" + stereo_train_n + ".txt"
img_txt_name = "Y:/Stereo_dataset/Apollo_image_list_" + stereo_train_n + ".txt"
line_count_disp = file_len(disp_txt_name)
line_count_img = file_len(img_txt_name)
point_cloud_path = 'Y:/Stereo_dataset/stereo_train_001/disparity/point_cloud_data/'
# img and disparity open
f = open(disp_txt_name, 'r')
j = open(img_txt_name, 'r')
# Pointcloud saving txt file
count = 0
while True:
    point_cloud_txt = '%010d.txt' % count
    point_cloud_savename = point_cloud_path + point_cloud_txt
    k = open(point_cloud_savename, 'w')
    disp_line = f.readline()
    img_line = j.readline()
    disparity_slash_count = disp_line.count('\\')
    disparity_path = disp_line[:-1].replace("\\", "/", disparity_slash_count)
    img_slash_count = img_line.count('\\')
    img_path = img_line[:-1].replace("\\", "/", img_slash_count)
    # Img read
    img_color = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img_color = cv2.resize(img_color, (256, 80))
    img_b, img_g, img_r = cv2.split(img_color)
    disp_color = cv2.imread(disparity_path)
    disp_color = cv2.resize(disp_color, (256, 80))
    disp_x, disp_y, disp_z = cv2.split(disp_color)
    # Point cloud data saved 256x80\
    if not os.path.exists(point_cloud_savename) or os.path.getsize(point_cloud_savename) == 0:
        for x in range(0, 80):
            for y in range(0, 256):
                point_cloud_dataname = "%d %d %d %d %d %d\n" %(x, y, disp_z[x][y], img_r[x][y], img_g[x][y], img_b[x][y])
                k.write(point_cloud_dataname)
                #print(point_cloud_dataname)
        k.close()
    if not disp_line: break
    count += 1
    print('{} saving is done.'.format(point_cloud_txt))
# All works done
print("All txt files are saved!")

'''
point_cloud_data = np.loadtxt(point_cloud_savename)
xyz = point_cloud_data[:,:3]
rgb = point_cloud_data[:,3:]
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(xyz[:,2], xyz[:,1], xyz[:,0], c = xyz[:,2]/255, s=2.0)
plt.show()
'''


#Disp read

# BGR Checker
''' 
zeros = np.zeros((img_color.shape[0], img_color.shape[1]), dtype="uint8")
img_b = cv2.merge([img_b, zeros, zeros])
img_g = cv2.merge([zeros, img_g, zeros])
img_r = cv2.merge([zeros, zeros, img_r])
cv2.imshow("BGR", img_color)
cv2.imshow("B", img_b)
cv2.imshow("G", img_g)
cv2.imshow("R", img_r)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''

k.close()
f.close()
j.close()



