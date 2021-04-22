"""setup_nodvi.py -- A script to preprocess nslam data.

    This script will be used to preprocess and build train/test
    split for the nslam data. The data will be used for training
    DSAC*.

.. moduleauthor:: Khaled Talukder <khaled@nod-labs.com>
"""

import os
import random
import warnings
import yaml


def binary_search_approx(value, array):
    r"""Simple binary search to find a closest value in an array.

    Given an `array` and given a `value`, returns an index `j` such that `value` 
    is between `array[j]` and `array[j+1]`. `array` must be monotonic increasing. 
    `j=-1` or `j=len(array)` is returned to indicate that ``value`` is out of range 
    below and above respectively.
    """
    n = len(array)
    if value < array[0]:
        return -1
    elif value > array[n-1]:
        return n
    lower, upper = 0, n-1  # Initialize lower and upper
    while upper - lower > 1:  # If we are not yet done,
        mid = (upper + lower) >> 1  # compute a midpoint with a bitshift
        # if they are closer than delta, then that's it
        if value - array[mid] == 0:
            return mid
        if value >= array[mid]:
            lower = mid  # and replace either the lower limit
        else:
            upper = mid  # or the upper limit, as appropriate.
        # Repeat until the test condition is satisfied.
    if value == array[0]:  # edge cases at bottom
        return 0
    elif value == array[n-1]:  # and top
        return n-1
    else:
        # this means we couldn't find exact match.
        # so now minimize the difference.
        if upper - lower > 1:
            minimum = [lower, float('inf')]
            for i in range(lower, upper+1):
                delta = abs(value - array[i])
                if delta <= minimum[1]:
                    minimum = [i, delta]
            return minimum[0]
        else:
            return lower


def binary_search_delta(value, array, delta=0):
    r"""Simple binary search to find a closest value in an array.

    Given an `array` and given a `value`, returns an index `j` such that `value` 
    is between `array[j]` and `array[j+1]`. `array` must be monotonic increasing. 
    `j=-1` or `j=len(array)` is returned to indicate that ``value`` is out of range 
    below and above respectively.
    """
    n = len(array)
    if value < array[0]:
        return -1
    elif value > array[n-1]:
        return n
    lower, upper = 0, n-1  # Initialize lower and upper
    while upper - lower > 1:  # If we are not yet done,
        mid = (upper + lower) >> 1  # compute a midpoint with a bitshift
        # if they are closer than delta, then that's it
        if abs(value - array[mid]) <= delta:
            return mid
        if value >= array[mid]:
            lower = mid  # and replace either the lower limit
        else:
            upper = mid  # or the upper limit, as appropriate.
        # Repeat until the test condition is satisfied.
    if value == array[0]:  # edge cases at bottom
        return 0
    elif value == array[n-1]:  # and top
        return n-1
    else:
        return lower


def parse_nodconfig(path):
    r"""Get the camera intrinsics and other stuffs from the config file.
    """
    with open(path) as fp:
        config = yaml.load(fp, Loader=yaml.FullLoader)
        return (config['cams']['cam0']['intrinsics'],
                config['cams']['cam0']['timeshift_cam_imu0'])


def collect_images(root):
    r"""Collect all the camera frames from the image folder.

    Takes the root of the data folder and subfolders as different `take`.
    Then it collates all the images in a single dict. The key is the timestamp
    and the values are the path to im`age file.
    """
    image_path = os.path.join(root, 'nodvi/device/data/images0')
    files = sorted(os.listdir(image_path))
    
    images = {}
    duplicates = 0
    for file_name in files:
        ts = int(file_name.split('.')[0])
        if ts not in images:
            images[ts] = os.path.join(image_path, file_name)
        else:
            duplicates = duplicates + 1
    if duplicates > 0:
        warnings.warn("Found {0:d} different images with the same timestamp. ".format(duplicates)
                      + "Please realign the timestamps with images or collect data again.")
    return images


def collect_poses(root):
    r"""Collect all the ground-truth camera poses from the `groundtruth` folder.

    This function collects all the ground-truth camera poses and returns a dict.
    The keys are the timestamps of each pose and the values are camera position in
    3D and orientation in quarternion.
    """
    config_path = os.path.join(root, 'nodvi/device/nodconfig.yaml')
    intrinsics, shift = parse_nodconfig(config_path)
    focal_length = intrinsics[0]
    pose_path = os.path.join(root, 'nodvi/groundtruth/data.csv')
    
    poses = {}
    duplicates = 0
    with open(pose_path, 'r') as fp:
        for line in fp.readlines()[1:]:
            vals = line.strip().split(',')
            ts, pose = int(vals[0]) + int(shift), [float(v)
                                                   for v in vals[1:]]
            pose.append(focal_length)
            if ts not in poses:
                poses[ts] = pose
            else:
                duplicates = duplicates + 1
    if duplicates > 0:
        warnings.warn("Found {0:d} different poses with the same timestamp. ".format(duplicates)
                      + "Please realign the timestamps with images or collect data again.")
    return poses


def build_image_to_pose_map(images, poses, delta=0):
    r"""Makes a mapping from image timestamp to pose timestamp.

    From `image` and `pose` dicts. This function creates a mapping
    between them. Takes each item from `image` and search through
    `poses` to find the closest pose with respect to the timestamp. 
    """
    image_ts = sorted(images.keys())
    pose_ts = sorted(poses.keys())
    n = len(image_ts)

    image_to_pose_map = {}
    mean_delta = 0
    found = 0
    missing = 0
    for i in range(n):
        if delta > 0:
            j = binary_search_delta(image_ts[i], pose_ts, delta=delta)
        else:
            j = binary_search_approx(image_ts[i], pose_ts)
        if 0 <= j < n:
            image_to_pose_map[image_ts[i]] = pose_ts[j]
            mean_delta = mean_delta + abs(image_ts[i] - pose_ts[j])
            found = found + 1
        else:
            missing = missing + 1
    mean_delta = mean_delta / n
    print("{0:d} images have closest matching poses.".format(found))
    print("Mean Time-stamp deviations: {0:.3f}".format(mean_delta))
    if missing > 0:
        warnings.warn("Timestamp misalignment, "
                      + "{0:d} images don't have closest time-stamped poses.".format(missing)
                      + " May be you need to collect the data again?")
    return image_to_pose_map, mean_delta


def search_best_delta(take):
    r"""Find the best delta. 
        Search for the best delta so that the mean deviation from the image timestamps
        and pose timestamps are minimized.
    """
    root = os.path.join(data_home, 'recordvi')
    train_perc = 0.75

    # collect images
    images = collect_images(os.path.join(root, take))
    print("Found {0:d} camera frames.".format(len(images)))
    # collect ground truth poses
    poses = collect_poses(os.path.join(root, take))
    print("Found {0:d} camera poses.".format(len(poses)))
    # find the best delta to minimize mismatch
    fp = open('delta.txt', 'w')
    min_mean_dev = [0, float('inf')]
    for delta in range(1, 3000000, 1000):
        # build the image timestamp to pose timestamp map
        itpmap, mean_dev = build_image_to_pose_map(images, poses, delta)
        print(delta, mean_dev)
        if mean_dev < min_mean_dev[1]:
            min_mean_dev = [delta, mean_dev]
        fp.write("{0:d}\t{1:.3f}\n".format(delta, mean_dev))
    fp.close()
    print(min_mean_dev)


if __name__ == "__main__":

    # This script assumes that the data folder location is set in `DATA_HOME`.
    data_home = os.environ['DATA_HOME']

    # The nslam data are saved in this way:
    # $DATA_HOME/recordvi
    #   - recordvi-#-##-### (takes)
    #       - nodvi
    #           - device
    #               - data
    #                   - images0 (frames)
    #                   - data.csv (list of file names with timestamps)
    #                   - imu0.csv (pose info of imu0)
    #               - nodconfig.yml
    #           - groundtruth
    #               - config.yaml
    #               - optitrack.csv (optitrack pose with timestamps)
    #               - data.csv (ground truth pose with timestamps and header, we will use this)
    #               - data_no_header.csv (data.csv with no header)
    root = os.path.join(data_home, 'recordvi')
    takes = ['recordvi-4-02-000', 'recordvi-4-02-003', 'recordvi-4-02-004']
    # these deltas were found using the search_best_delta() function
    deltas = [2068001, 2072001, 2059001]
    train_perc = 0.75

    data = []
    for i in range(len(takes)):
        path = os.path.join(root, takes[i])
        # collect images
        images = collect_images(path)
        print("Found {0:d} camera frames in {1:s}.".format(len(images), path))
        # collect ground truth poses
        poses = collect_poses(path)
        print("Found {0:d} camera poses in {1:s}.".format(len(poses), path))
        # build the image timestamp to pose timestamp map
        img_to_pose, mean_dev = build_image_to_pose_map(images, poses, delta=deltas[i])
        for k,v in img_to_pose.items():
            data.append([images[k],poses[v]])
   
    # shuffle the consolidated data
    random.shuffle(data)
    train_count = int(len(data) * train_perc)

    # makes the train data file mapping
    train_map_path = os.path.join(root, "train-map.csv")
    print("Writing training data mapping in {0:s}.".format(train_map_path))
    with open(train_map_path, 'w') as fp:
        for i in range(train_count):
            line = data[i][0] + ',' + ','.join([str(v) for v in data[i][1]]) + '\n'
            fp.write(line)

    # makes the test data file mapping
    test_map_path = os.path.join(root, "test-map.csv")
    print("Writing testing data mapping in {0:s}.".format(test_map_path))
    with open(test_map_path, 'w') as fp:
        for i in range(train_count, len(data)):
            line = data[i][0] + ',' + ','.join([str(v) for v in data[i][1]]) + '\n'
            fp.write(line)
    
    print("Done.")

    # search_best_delta(takes[2])
