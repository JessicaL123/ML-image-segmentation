import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from PIL import Image
import os
import sys

# command:
#  python3 pca-compressed.py image1.jpg 30

if len(sys.argv) != 3:
    print("usage: python3 " + sys.argv[0] + " image_file num_components")
    exit()

image_file = sys.argv[1]
num_components = int(sys.argv[2])

image = plt.imread(image_file)

r = image[:, :, 0]
g = image[:, :, 1]
b = image[:, :, 2]

r_pca = PCA(n_components=num_components)
g_pca = PCA(n_components=num_components)
b_pca = PCA(n_components=num_components)

low_dim_r = r_pca.fit_transform(r)
low_dim_g = g_pca.fit_transform(g)
low_dim_b = b_pca.fit_transform(b)

approx_r = r_pca.inverse_transform(low_dim_r)
approx_g = g_pca.inverse_transform(low_dim_g)
approx_b = b_pca.inverse_transform(low_dim_b)

new_pic = np.dstack((approx_r, approx_g, approx_b))
new_pic = new_pic.reshape(image.shape)
new_pic = np.uint8(np.absolute(new_pic))

output_path = "./compressedImage"
if not os.path.exists(output_path):
    os.mkdir(output_path)

pca_img = Image.fromarray(new_pic)
pca_img.save(output_path + "/compressed_" + str(num_components) + "_components_" + image_file)