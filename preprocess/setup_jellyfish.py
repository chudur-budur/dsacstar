"""setup_jellyfish.py -- A script to preprocess Jellyfish SLAM data.

    This script will be used to preprocess and build train/test
    split for the nslam data. The data will be used for training
    DSAC*.

.. moduleauthor:: Khaled Talukder <khaled@nod-labs.com>
"""

import os
import random
import warnings
import yaml
import numpy as np
from scipy import interpolate


def binary_search_approx(value, array):
    r"""Simple binary search to find a closest value in an array.

    Given an `array` and given a `value`, returns an index `j` such that `value` 
    is between `array[j]` and `array[j+1]`. `array` must be monotonic increasing. 
    `j=-1` or `j=len(array)` is returned to indicate that ``value`` is out of range 
    below and above respectively.
    """
    n = len(array)
    if value < array[0]: 
        return 0
    if value > array[n-1]:
        return n-1
    lower, upper = 0, n-1  # Initialize lower and upper
    while upper - lower > 1:  # If we are not yet done,
        mid = (upper + lower) >> 1  # compute a midpoint with a bitshift
        # if they are closer than delta, then that's it
        if value == array[mid]:
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
        # return the index with minimum difference.
        if abs(value - array[lower]) <= abs(value - array[upper]):
            return lower
        else:
            return upper


def parse_nodconfig(path):
    r"""Get the camera intrinsics and other stuffs from the config file.
    """
    with open(path) as fp:
        config = yaml.load(fp, Loader=yaml.FullLoader)
        return (config['cams']['cam0']['intrinsics'],
                config['cams']['cam0']['timeshift_cam_imu0'])


def get_intrinsics(root):
    r"""Get the camera intrinsics.

    This function only handles focal length and shift.
    """
    config_path = os.path.join(root, 'nodvi/device/nodconfig.yaml')
    if os.path.exists(config_path):
        try:
            intrinsics, shift = parse_nodconfig(config_path)
            focal_length = intrinsics[0]
        except:
            focal_length, shift = 628.562541875901, 0 # Default values.
    return focal_length,shift


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
    focal_length,shift = get_intrinsics(root)
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


def interpolate_pose(poses, poses_ts, j, x, k=5):
    r"""The linear interpolation function.

    Given the image timestamp x, we interpolate it's pose from `2k` data points
    around the closest pose timestamped at `j`. We take `k` poses before `j` and
    `k` poses after `j`, with bound constraint checked.
    """
    xp = [poses_ts[j+i] for i in range(-k,k+1) if 0 <= j+i < len(poses_ts)]
    pose = []
    if x > xp[-1]:
        for i in range(7):
            fp = [poses[l][i] for l in xp] 
            f = interpolate.interp1d(xp, fp, fill_value = "extrapolate")
            pose.append(f(x))
    else:
        for i in range(7):
            fp = [poses[l][i] for l in xp] 
            pose.append(np.interp(x, xp, fp))
    return pose


def align_timestamps(its, pts):
    r"""Align image and pose timestamps.

    Trim the image timestamps with respect to the
    range of the pose timestamps.
    """
    # if the image timestamp starts way 
    # earlier than the pose timestamp
    if its[0] < pts[0]: 
        j = binary_search_approx(pts[0], its)
        its = its[0:j+1]
    # if the pose timestamp ends way 
    # later than the image timestamp
    if its[-1] < pts[-1]:
        j = binary_search_approx(its[-1], pts)
        pts = pts[0:j+1]
    
    return its, pts


def approximate(images, poses, interpolate=True):
    r"""Approximate the missing poses.

    This function approximate poses for the corresponding images
    with respect to the timestamps. If the there are matching timestamps
    for images and poses, we will take the pose for that timestamp. However
    if there is no pose for that image at the same timestamp, we do 
    a linear approximation.

    If `interpolate` is `False`, the we take the pose that is closest
    to the timestamp of the image, otherwise we do a linear interpolation
    (i.e. approximation).
    """
    image_ts = sorted(images.keys())
    pose_ts = sorted(poses.keys())
    image_ts, pose_ts = align_timestamps(image_ts, pose_ts)

    poses_ = {}
    n,m = len(image_ts), len(pose_ts)
    missing = 0
    for i in range(n):
        if image_ts[i] in poses: # Found a pose at the same image timestamp.
            poses_[image_ts[i]] = poses[image_ts[i]]
        else: # Not found.
            missing = missing + 1
            j = binary_search_approx(image_ts[i], pose_ts)
            if interpolate:
                approx_pose = interpolate_pose(poses, pose_ts, j, image_ts[i])
                # All values after 7th item are intrinsics
                # taking poses[pose_ts[j]][7:] since there 
                # is no way to approximate it
                approx_pose.extend(poses[pose_ts[j]][7:])
            else:
                approx_pose = poses[pose_ts[j]]
            poses_[image_ts[i]] = approx_pose
    if missing > 0:
        warnings.warn("Total of {0:d} poses approximated ({1:.2f}%). "\
                .format(missing, (missing / n) * 100.0))
    return poses_



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
    
    # root = os.path.join(data_home, 'recordvi')
    # takes = ['recordvi-4-02-000', 'recordvi-4-02-003', 'recordvi-4-02-004']
    root = os.path.join(data_home, 'jellyfish')
    takes = ['04-21-0100']
    train_perc = 0.75

    data = []
    for i in range(len(takes)):
        path = os.path.join(root, takes[i])
        # Collect images.
        images = collect_images(path)
        print("Found {0:d} camera frames in {1:s}.".format(len(images), path))
        # Collect ground truth poses.
        poses = collect_poses(path)
        print("Found {0:d} camera poses in {1:s}.".format(len(poses), path))
        # Approximate poses from the images, if there are matching timestamps,
        # no need to approximate.
        poses_ = approximate(images, poses)
        for k in images:
            data.append([images[k], poses_[k]])

    # Shuffle the consolidated data.
    random.shuffle(data)
    train_count = int(len(data) * train_perc)

    # Create folder to keep the split files.
    split_file_root = "../split-files"
    if not os.path.exists(split_file_root):
        os.mkdir(split_file_root)

    # Make the train data file mapping.
    train_map_path = os.path.join(split_file_root, "jellyfish-train-map.csv")
    print("Writing training data mapping in {0:s}.".format(train_map_path))
    with open(train_map_path, 'w') as fp:
        for i in range(train_count):
            line = data[i][0] + ',' + \
                ','.join([str(v) for v in data[i][1]]) + '\n'
            fp.write(line)

    # Make the test data file mapping.
    test_map_path = os.path.join(split_file_root, "jellyfish-test-map.csv")
    print("Writing testing data mapping in {0:s}.".format(test_map_path))
    with open(test_map_path, 'w') as fp:
        for i in range(train_count, len(data)):
            line = data[i][0] + ',' + \
                ','.join([str(v) for v in data[i][1]]) + '\n'
            fp.write(line)

    print("Done.")

    # search_best_delta(takes[2])