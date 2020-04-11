from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


map = Image.open('map_pic.jpg', 'r')
scale = Image.open('scale_pic.jpg', 'r')


#pix_val = list(im.getdata())
np_map = np.array(map)
np_scale = np.array(scale)

rows, cols, _ = np_map.shape

pixel_map = np.empty([rows, cols,], dtype = int)
for row in range(0,rows):
    for col in range(0,cols):
        if sum(np_map[row][col]) < 200:
            pixel_map[row][col] = 0
        else:
            pixel_map[row][col] = 1

rows, cols, _ = np_scale.shape

max_width = 0
for row in range(0, rows):
    width = 0
    for col in range(0,cols):
        if sum(np_scale[row][col]) < 600:
            width += 1
        elif (width != 0):
            if width > max_width:
                max_width = width
            break

p_per_m = np.double(max_width/100)
print(p_per_m)
#print(pix_val)
