import os
from itertools import cycle, islice
import numpy as np
from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.manifold import TSNE
from sklearn.neighbors import kneighbors_graph
from sklearn import cluster 
import transforms as tr
import cv2

__all__ = ['load_raw', 'save_flat', 'load_flat']


def load_raw(map_file_path, n=float('inf'), scale=0.05):
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
        w, h = int(image.shape[1] * scale), int(image.shape[0] * scale)
        image = cv2.resize(image, (w, h), interpolation = cv2.INTER_AREA)  
        
        data[ts] = [pose, image.reshape(-1)]

        count = count + 1
        if count >= n:
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
        fp.write(key + ',' + ','.join([str(v) for v in pose]) + ',' \
                + ','.join([str(v) for v in image]) + '\n')
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


def get_colors(L, cmap=None):
    if cmap is None:
        clist = [
                '#377eb8', '#ff7f00', '#4daf4a', 
                '#f781bf', '#a65628', '#984ea3', 
                '#999999', '#e41a1c', '#dede00']
    else:
        cm = plt.get_cmap(cmap)
        clist = [cm(1. * i / len(L)) for i in range(len(L))]
    
    colors = np.array(list(islice(cycle(clist), int(max(Y) + 1))))
    # add black color for outliers (if any)
    colors = np.append(colors, ["#000000"])
    return colors


def search_dbscan_eps(P):
    for eps in np.arange(0.02, 0.04, 0.001):
        clustering = cluster.DBSCAN(eps=eps)
        clustering.fit(P)

        if hasattr(clustering, 'labels_'):
            Y = clustering.labels_.astype(int)
        else:
            Y = clustering.predict(P)

        L = set(Y)
        print(eps, len(L), L)

if __name__ == "__main__":
    np.random.seed(123456)

    # data, dim = load_raw("split-files/jellyfish-train-map.csv")
    # print(len(data), dim)
    # keys = list(data.keys())
    # print(keys)
    # cv2.imwrite("test.png", data[keys[0]][1].reshape(dim[1], dim[0]))
    # save_flat(data, "flat.csv")

    # data = load_flat("flat.csv")
    # keys = list(data.keys())
    # P = np.array([data[k][0] for k in keys]).astype(float)
    # M = np.array([data[k][1] for k in keys]).astype(int)

    # tsne_pose = TSNE(n_components=2, random_state=111111, verbose=True, n_iter=5000)
    # tsne_image = TSNE(n_components=2, random_state=333333, verbose=True, n_iter=5000)
    # P_ = tsne_pose.fit_transform(P)
    # M_ = tsne_image.fit_transform(M)
    # P_ = (P_ - np.min(P_)) / np.ptp(P_)
    # M_ = (M_ - np.min(M_)) / np.ptp(M_) 

    # np.savetxt("pose-tsne.csv", P_, delimiter=',')
    # np.savetxt("image-tsne.csv", M_, delimiter=',')
    
    P = np.loadtxt("pose-tsne.csv", delimiter=',')
    M = np.loadtxt("image-tsne.csv", delimiter=',')

    clustering = cluster.DBSCAN(eps=0.029)
    # clustering = cluster.OPTICS(min_samples=100, xi=0.35, min_cluster_size=0.3)
    # clustering = cluster.MeanShift(bandwidth=cluster.estimate_bandwidth(P, quantile-0.2), \
    #         bin_seeding=True)
    # clustering = cluster.SpectralClustering(n_clusters=12, eigen_solver='arpack', \
    #         affinity="nearest_neighbors")
    # clustering = cluster.AffinityPropagation(damping=0.75, preference=-10)
    # C = kneighbors_graph(P, n_neighbors=2, include_self=False)
    # C = 0.5 * (C + C.T)
    # clustering = cluster.AgglomerativeClustering(n_clusters=12, linkage='ward', connectivity=C)

    clustering.fit(P)

    if hasattr(clustering, 'labels_'):
        Y = clustering.labels_.astype(int)
    else:
        Y = clustering.predict(P)

    L = set(Y)
    print(len(L), L)

    colors = get_colors(L) 
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.8, 4.8))
    
    ax1.set_title("Poses")
    ax1.set_xlabel('u')
    ax1.set_ylabel('v')
    ax1.scatter(P[:,0], P[:,1], s=2, color=colors[Y])
    
    ax2.set_title("Images")
    ax2.set_xlabel('u')
    ax2.set_ylabel('v')
    ax2.scatter(M[:,0], M[:,1], s=2, color=colors[Y])
    
    plt.show()
