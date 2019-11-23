import numpy as np
from sklearn.cluster import KMeans
from PIL import Image
import matplotlib.pyplot as plt
import sys
import os
import random

# command:
# python3 color-quantization.py image1.jpg 3


if len(sys.argv) != 3:
    print("usage: python3 " + sys.argv[0] + " 'image_file' 'k'")
    exit()

image_file = sys.argv[1]
k = int(sys.argv[2])

img = plt.imread(image_file)
image = np.array(img, dtype=np.float64)

rows, cols, depth = image.shape
image_array = image.reshape(rows * cols, depth)

# randomly select some samples for training
idx = [random.randrange(len(image_array)) for i in range(len(image_array) // 100)]
training_sample = image_array[idx]
kmeans = KMeans(n_clusters=k, random_state=0).fit(training_sample)

# Get labels for all points
labels = kmeans.predict(image_array)

quan_img = kmeans.cluster_centers_[labels].reshape(img.shape)
new_pic = np.uint8(np.absolute(quan_img))

output_path = "./quantizedImages"
if not os.path.exists(output_path):
    os.mkdir(output_path)

pca_img = Image.fromarray(new_pic)
pca_img.save(output_path + "/quantized_" + str(k) + "_" + image_file)
