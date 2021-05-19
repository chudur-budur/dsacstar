import os
import numpy as np

def load_data(map_file_path):
    data = {}
    fp = open(map_file_path, 'r')
    for line in fp:
        vals = line.strip().split(',')
        image_path = vals[0]
        path,file_name = os.path.split(image_path)
        ts = file_name.split('.')[0]
        pose = np.array([float(v) for v in vals[1:8]]).astype(float)
        data[ts] = [pose, image_path]
    fp.close()
    return data

if __name__ == "__main__":
    data = load_data("split-files/jellyfish-train-map.csv")
    print(data)

