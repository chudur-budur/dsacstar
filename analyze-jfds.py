import os
import numpy as np
from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.manifold import TSNE
from torchvision import transforms
import transforms as tr

def make_pileline(img, came):
    pass

def load_data(map_file_path):

    data = {}
    fp = open(map_file_path, 'r')
    for line in fp:
        vals = line.strip().split(',')
        image_path = vals[0]

        path,file_name = os.path.split(image_path)
        ts = file_name.split('.')[0]
        camera_intrinsics = np.array([float(v) for v in vals[1:8]]).astype(float)
        distortion_coeffs = np.array([float(v) for v in vals[8:-1]]).astype(float)

        pipeline = transforms.Compose([
            transforms.Lambda(lambda img: tr.unfish(
                img, camera_intrinsics, distortion_coeffs)),
            transforms.Lambda(lambda img: tr.cambridgify(img)),
            transforms.ToPILImage(),
            transforms.Resize(120),
            transforms.Grayscale(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1)])

        image = pipeline(image_path)
        data[ts] = [pose, image]
    fp.close()
    return data

if __name__ == "__main__":
    # data = load_data("split-files/jellyfish-train-map.csv")
    # print(data)

    digits = datasets.load_digits()
    
    print("len(digits):", len(digits), "type(digits):", type(digits))
    print("len(digits.data[0]):", len(digits.data[0]), "digits.data[0]:", \
            digits.data[0], "digits.target[0]:", digits.target[0])
    
    # Take the first 500 data points: it's hard to see 1500 points
    X = digits.data[:500]
    y = digits.target[:500]

    tsne = TSNE(n_components=2, random_state=0)
    X_2d = tsne.fit_transform(X)
    target_ids = range(len(digits.target_names))

    plt.figure(figsize=(6, 5))
    colors = 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'w', 'orange', 'purple'
    for i, c, label in zip(target_ids, colors, digits.target_names):
        plt.scatter(X_2d[y == i, 0], X_2d[y == i, 1], c=c, label=label, s=1)
    plt.legend()
    plt.show()
