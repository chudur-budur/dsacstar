import os
import random
import warnings
import yaml


def binary_search(value, array, delta=10):
    """Simple binary search to find a closest value in an array.
        Given an `array` and given a `value`, returns an index `j` such that 
        `value` is between `array[j]` and `array[j+1]`. `array` must be monotonic increasing. 
        `j=-1` or `j=len(array)` is returned to indicate that ``value`` is out of range 
        below and above respectively.
    """
    n = len(array)
    if value < array[0]:
        return -1
    elif value > array[n-1]:
        return n
    jl = 0 # Initialize lower
    ju = n-1 # and upper limits.
    while ju - jl > 1: # If we are not yet done,
        jm = (ju + jl) >> 1 # compute a midpoint with a bitshift
        if abs(value - array[jm]) < delta: # if they are closer than delta, then that's it
            jl = jm
            break
        if value >= array[jm]:
            jl = jm # and replace either the lower limit
        else:
            ju = jm # or the upper limit, as appropriate.
        # Repeat until the test condition is satisfied.
    if value == array[0]: # edge cases at bottom
        return 0
    elif value == array[n-1]: # and top
        return n-1
    else:
        return jl


def parse_camera_intrinsics(path):
    with open(path) as fp:
        config = yaml.load(fp, Loader=yaml.FullLoader)
        return config['cams']['cam0']['intrinsics']


def collect_images(root, takes):
    images = {}
    duplicates = 0
    for take in takes:
        image_path = os.path.join(root, take, 'nodvi/device/data/images0')
        files = sorted(os.listdir(image_path))
        for file_name in files:
            ts = file_name.split('.')[0]
            if ts not in images:
                images[ts] = os.path.join(image_path, file_name)
            else:
                duplicates = duplicates + 1
    if duplicates > 0:
        warnings.warn("Found {0:d} different images with the same timestamp. ".format(duplicates) \
                + "Please realign the timestamps with images or collect data again.")
    return images


def collect_poses(root, takes):
    poses = {}
    duplicates = 0
    for take in takes:
        nodconfig_path = os.path.join(root, take, 'nodvi/device/nodconfig.yaml')
        intrinsics = parse_camera_intrinsics(nodconfig_path)
        focal_length = intrinsics[0]
        pose_path = os.path.join(root, take, 'nodvi/groundtruth/data.csv')
        with open(pose_path, 'r') as fp:
            for line in fp.readlines()[1:]:
                vals = line.strip().split(',')
                ts,pose = vals[0], [float(v) for v in vals[1:]]
                pose.append(focal_length)
                if ts not in poses:
                    poses[ts] = pose
                else:
                    duplicates = duplicates + 1
    if duplicates > 0:
        warnings.warn("Found {0:d} different poses with the same timestamp. ".format(duplicates) \
                + "Please realign the timestamps with images or collect data again.")
    return poses


def build_image_to_pose_map(images, poses):
    images_keys = sorted([[float(v), v] for v in images.keys()])
    poses_keys = sorted([[float(v), v] for v in poses.keys()])
    image_ts = [v[0] for v in images_keys]
    pose_ts = [v[0] for v in poses_keys]
    
    image_to_pose_map = {}
    n = len(image_ts)
    mean_delta = 0.0
    found = 0
    missing = 0
    for i in range(n):
        j = binary_search(image_ts[i], pose_ts)
        if 0 < j < n:
            image_to_pose_map[images_keys[i][1]] = poses_keys[j][1]
            mean_delta = mean_delta + abs(image_ts[i] - pose_ts[j])
            found = found + 1
        else:
            missing = missing + 1
    mean_delta = mean_delta / n
    print("{0:d} images have closest matching poses.".format(found))
    print("Mean Time-stamp deviations: {0:.3f}".format(mean_delta))
    if missing > 0:
        warnings.warn("Timestamp misalignment, " \
                + "{0:d} images don't have closest time-stamped poses.".format(missing) \
                + " May be you need to collect the data again?")
    return image_to_pose_map


if __name__ == "__main__":

    # name of the folder where we download the original 7scenes dataset to
    # we restructure the dataset by creating symbolic links to that folder
    data_home = os.environ['DATA_HOME']
    root = os.path.join(data_home, 'recordvi')
    takes = ['recordvi-4-02-000', 'recordvi-4-02-003', 'recordvi-4-02-004']
    train_perc = 0.75

    images = collect_images(root, takes)
    print("Found {0:d} camera frames.".format(len(images)))
    poses = collect_poses(root, takes)
    print("Found {0:d} camera poses.".format(len(poses)))
    
    itpmap = build_image_to_pose_map(images, poses)
    its = list(itpmap.keys())
    random.shuffle(its)
    train_count = int(len(its) * train_perc)

    train_map_path = os.path.join(root, "train-map.csv")
    print("Writing training data mapping in {0:s}.".format(train_map_path))
    with open(train_map_path, 'w') as fp:
        for i in range(train_count):
            img_key, pose_key = its[i], itpmap[its[i]]
            line = images[img_key] + ',' + ','.join([str(v) for v in poses[pose_key]]) + '\n'
            fp.write(line)
    
    test_map_path = os.path.join(root, "test-map.csv")
    print("Writing testing data mapping in {0:s}.".format(test_map_path))
    with open(test_map_path, 'w') as fp:
        for i in range(train_count,len(its)):
            img_key, pose_key = its[i], itpmap[its[i]]
            line = images[img_key] + ',' + ','.join([str(v) for v in poses[pose_key]]) + '\n'
            fp.write(line)
    print("Done.")

