import os
import numpy as np
from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.manifold import TSNE
import transforms as tr
import cv2

__all__ = ['load_raw', 'save_flat', 'load_flat']


def load_raw(map_file_path, n=float('inf')):
    data = {}
    fp = open(map_file_path, 'r')
    count = 0
    for line in fp:
        vals = line.strip().split(',')
        image_path = vals[0].strip()
        ts = os.path.split(image_path)[1].split('.')[0]
        pose = np.array([float(v) for v in vals[1:8]]).astype(float)
        camera_intrinsics = np.array([float(v) for v in vals[8:12]]).astype(float)
        distortion_coeffs = np.array([float(v) for v in vals[12:-1]]).astype(float)
        
        image = cv2.imread(image_path, 0)
        image = tr.unfish(image, camera_intrinsics, distortion_coeffs)
        scale_perc = 0.05 # percent of original size
        w, h = int(image.shape[1] * scale_perc), int(image.shape[0] * scale_perc)
        image = cv2.resize(image, (w, h), interpolation = cv2.INTER_AREA)  
        
        data[ts] = [pose, image]

        count = count + 1
        if count > n:
            break
        if count % 100 == 0:
            print("Loaded {0:d} images and poses.".format(count))

    fp.close()
    return data, (w,h)


def save_flat(data, path):
    fp = open(path, 'w')
    for key in data.keys():
        ts = key
        pose = data[key][0]
        image = data[key][1]
        fp.write(key + ',' + ','.join([v for v in pose]) + ',' + ','.join([v for v in image] + '\n')
    fp.close()


def load_flat(path):
    fp = open(path, 'r')
    data = {}
    for line in fp:
        vals = line.strip().split(',')
        key = vals[0]
        pose = np.array([float(v) for v in vals[1:8]]).astype(float)
        image = np.array([int(v) for v in vals[8:]]).astype(int)
        data[key] = [pose, image]
    fp.close()
    return data


if __name__ == "__main__":
    data, dim = load_raw("split-files/jellyfish-train-map.csv", n=10)
    print(len(data))
    keys = list(data.keys())
    print(keys)
    cv2.imwrite(data[keys[0]][1].reshape(dim), "test.png")
    save_flat(data, "flat.csv")
    data = load_flat("flat.csv")

    # digits = datasets.load_digits()
    # 
    # print("len(digits):", len(digits), "type(digits):", type(digits))
    # print("len(digits.data[0]):", len(digits.data[0]), "digits.data[0]:", \
    #         digits.data[0], "digits.target[0]:", digits.target[0])
    # 
    # # Take the first 500 data points: it's hard to see 1500 points
    # X = digits.data[:500]
    # y = digits.target[:500]

    # tsne = TSNE(n_components=2, random_state=0)
    # X_2d = tsne.fit_transform(X)
    # target_ids = range(len(digits.target_names))

    # plt.figure(figsize=(6, 5))
    # colors = 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'w', 'orange', 'purple'
    # for i, c, label in zip(target_ids, colors, digits.target_names):
    #     plt.scatter(X_2d[y == i, 0], X_2d[y == i, 1], c=c, label=label, s=5)
    # plt.legend()
    # plt.show()
