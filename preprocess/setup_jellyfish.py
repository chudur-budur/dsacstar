"""setup_jellyfish.py -- A script to preprocess Jellyfish SLAM data.

    This script will be used to preprocess and build train/test
    split for the nslam data. The data will be used for training
    DSAC*. This script assumes the data are saved roughly in this way --

        - home/
            - root/<take1, take2, ... takeN>/
                - nodvi/
                    - device/
                        - data/
                            - images0/ (frames)
                            - data.csv (list of file names with timestamps)
                            - imu0.csv (pose info of imu0)
                        - nodconfig.yml
                - groundtruth/
                    - config.yaml
                    - optitrack.csv (optitrack pose with timestamps)
                    - data.csv (ground truth pose with timestamps and header, we will use this)
                    - data_no_header.csv (data.csv with no header)

.. moduleauthor:: Khaled Talukder <khaled@nod-labs.com>
"""

import os
import random
import warnings
import argparse
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
                config['cams']['cam0']['distortions'],
                config['cams']['cam0']['timeshift_cam_imu0'])


def get_intrinsics(root):
    r"""Get the camera intrinsics.

    This function only handles focal length and shift.
    """
    config_path = os.path.join(root, 'nodvi/device/nodconfig.yaml')
    if os.path.exists(config_path):
        try:
            intrinsics, distortion_coeff, timeshift = parse_nodconfig(config_path)
        except:
            # Default values
            # intrinsics [fx, fy, cx, cy]
            intrinsics = [628.562541875901, 627.2138591418039, 949.2413699450868, 519.1072917895697]
            # distortion coefficients [k1, k2, k3 k4]
            distortion_coeff = [0.20157950702488608, -0.05621291427717055, \
                                -0.030506199533652974, 0.021067301350824064]            
            timeshift = 0.0
    return intrinsics, distortion_coeff, timeshift


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
    intrinsics, distortion_coeff, timeshift = get_intrinsics(root)
    pose_path = os.path.join(root, 'nodvi/groundtruth/data.csv')
    poses = {}
    duplicates = 0
    with open(pose_path, 'r') as fp:
        for line in fp.readlines()[1:]:
            vals = line.strip().split(',')
            ts, pose = int(vals[0]), [float(v) for v in vals[1:]]
            pose.extend(intrinsics)
            pose.extend(distortion_coeff)
            pose.append(timeshift)
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
    xp = [poses_ts[j+i] for i in range(-k, k+1) if 0 <= j+i < len(poses_ts)]
    pose = []
    if x > xp[-1]:
        for i in range(7):
            fp = [poses[l][i] for l in xp]
            f = interpolate.interp1d(xp, fp, fill_value="extrapolate")
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
    # this doesn't work in all situations, so turining it off
    # image_ts, pose_ts = align_timestamps(image_ts, pose_ts)

    poses_ = {}
    n, m = len(image_ts), len(pose_ts)
    missing = 0
    for i in range(n):
        if image_ts[i] in poses:  # Found a pose at the same image timestamp.
            poses_[image_ts[i]] = poses[image_ts[i]]
        else:  # Not found.
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
        warnings.warn("{0:d} out of {1:d} ({2:.2f}%) poses had to approximate."
                      .format(missing, n, (missing / n) * 100.0))
    return poses_


def downsample(data, n):
    r"""Take a subset of the original data.

    Take `n` number of data points from data.
    """
    if n == float('inf'):
        n = len(data)
    data_ = []
    delta = int(np.ceil(len(data)/n))
    for i in range(0, len(data), delta):
        data_.append(data[i])
    return data_


def prepare_root(root, train_perc):
    data = []
    path = root  # os.path.join(root, takes[i])
    # Collect images.
    images = collect_images(path)
    print("Found {0:d} camera frames in {1:s}.".format(len(images), path))
    # Collect ground truth poses.
    poses = collect_poses(path)
    print("Found {0:d} camera poses in {1:s}.".format(len(poses), path))
    # Approximate poses from the images, if there are matching timestamps,
    # no need to approximate.
    poses_ = approximate(images, poses)
    for k in poses_.keys():
        data.append([images[k], poses_[k]])

    # Shuffle the consolidated data.
    random.shuffle(data)
    train_count = int(len(data) * train_perc)

    # make training and testing data
    train = []
    for i in range(train_count):
        train.append(data[i])
    test = []
    for i in range(train_count, len(data)):
        test.append(data[i])

    return train, test


def prepare_recordvi(data_home, train_perc):
    r"""Prepare datasets for older jellyfish SLAM data.

    This is an older dataset found from the Jellyfish. 
    These data points are not 100% accurate sensor readings.
    """
    root = os.path.join(data_home, 'recordvi')
    takes = ['recordvi-4-02-000', 'recordvi-4-02-003', 'recordvi-4-02-004']
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
        for k in poses_.keys():
            data.append([images[k], poses_[k]])

    # Shuffle the consolidated data.
    random.shuffle(data)
    train_count = int(len(data) * train_perc)

    # make training and testing data
    train = []
    for i in range(train_count):
        train.append(data[i])
    test = []
    for i in range(train_count, len(data)):
        test.append(data[i])

    return train, test

def prepare_jellyfish(data_home, train_perc, n_samples=float('inf')):
    r"""Prepare the latest Jellyfish SLAM data.

    This scheme will prepare and load Jellyfish SLAM data collected
    on 04/21/21. Most likely these data points are accurate sensor readings.
    """
    root = os.path.join(data_home, "jellyfishdata/converted/2021-4-23") 
    takes = [
            "1/vlc-record-2021-04-23-17h13m50s-rtsp___192.168.2.1_stream1-",
            "1/vlc-record-2021-04-23-17h21m47s-rtsp___192.168.2.1_stream1-"]
    data = []
    # for i in range(len(takes)):
    path = os.path.join(root, takes[0])
    # Collect images.
    images = collect_images(path)
    print("Found {0:d} camera frames in {1:s}.".format(len(images), path))
    # Collect ground truth poses.
    poses = collect_poses(path)
    print("Found {0:d} camera poses in {1:s}.".format(len(poses), path))
    # Approximate poses from the images, if there are matching timestamps,
    # no need to approximate.
    poses_ = approximate(images, poses)
    for k in poses_.keys():
        data.append([images[k], poses_[k]])

    # downsample
    if n_samples < float('inf'):
        data = downsample(data, n=n_samples)

    # Shuffle the consolidated data.
    random.shuffle(data)
    train_count = int(len(data) * train_perc)

    # make training and testing data
    train = []
    for i in range(train_count):
        train.append(data[i])
    test = []
    for i in range(train_count, len(data)):
        test.append(data[i])

    return train, test


def prepare_jellyfish_separated(data_home):
    r"""Prepare the latest Jellyfish SLAM data.

    This scheme will prepare and load Jellyfish SLAM data collected
    on 04/21/21. Most likely these data points are accurate sensor readings.
    """
    root = os.path.join(data_home, "jellyfishdata/converted/2021-4-23") 
    takes = [
            "1/vlc-record-2021-04-23-17h13m50s-rtsp___192.168.2.1_stream1-",
            "1/vlc-record-2021-04-23-17h21m47s-rtsp___192.168.2.1_stream1-"]
    train, test = [],[]
    
    path = os.path.join(root, takes[0])
    # Collect images.
    images = collect_images(path)
    print("Found {0:d} camera frames in {1:s}.".format(len(images), path))
    # Collect ground truth poses.
    poses = collect_poses(path)
    print("Found {0:d} camera poses in {1:s}.".format(len(poses), path))
    # Approximate poses from the images, if there are matching timestamps,
    # no need to approximate.
    poses_ = approximate(images, poses)
    for k in poses_.keys():
        train.append([images[k], poses_[k]])
    
    path = os.path.join(root, takes[1])
    # Collect images.
    images = collect_images(path)
    print("Found {0:d} camera frames in {1:s}.".format(len(images), path))
    # Collect ground truth poses.
    poses = collect_poses(path)
    print("Found {0:d} camera poses in {1:s}.".format(len(poses), path))
    # Approximate poses from the images, if there are matching timestamps,
    # no need to approximate.
    poses_ = approximate(images, poses)
    for k in poses_.keys():
        test.append([images[k], poses_[k]])

    # Shuffle the consolidated data.
    random.shuffle(train)
    random.shuffle(test)

    return train, test


# entry point
if __name__ == "__main__":
    # seed random
    random.seed(123456)

    # Setup argparse
    parser = argparse.ArgumentParser(
        description="Preprocess Jellyfish data to train with DSAC*.",
        # formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('home', type=str,
                        help="""Path to the data home folder. 
        This script assumes the entry point to the folder to be like this:
        - home/
            - root/
                - take1/
                    - nodvi/
                        - device/
                            - data/
                                - images0/ (frames)
                                - data.csv (list of frames)
                                - imu0.csv (pose info of imu0)
                            - nodconfig.yml
                    - groundtruth/
                        - config.yaml
                        - optitrack.csv (optitrack pose)
                        - data.csv (ground truth pose)
                        - data_no_header.csv (data.csv with no header)
                - take2/
                    - nodvi/ ...
                - take3/
                    - nodvi/ ...
                ...
                - takeN/
                    - nodvi ...""")
    parser.add_argument('--recordvi', '-rv', action='store_true',
                        help="If set, data loading will be performed with recordvi scheme.")
    parser.add_argument('--jellyfish', '-jf', action='store_true',
                        help="If set, data loading will be performed for jellyfish scheme.")
    parser.add_argument('--jellyfishseparated', '-jfs', action='store_true',
                        help="If set, data loading will be performed for jellyfish (separated) scheme.")
    parser.add_argument('--nsamples', '-ns', type=int, default=float('inf'),
                        help="Total number of subsamples to be prepared.")
    parser.add_argument('--trainperc', '-p', type=float, default=0.75,
                        help=r'Total percentage of training data.')
    # parse now
    opt = parser.parse_args()

    n_samples = opt.nsamples
    train_perc = opt.trainperc

    # prepate the data
    train, test = None, None
    if opt.recordvi:
        train, test = prepare_recordvi(opt.home, train_perc)
    elif opt.jellyfish:
        train, test = prepare_jellyfish(opt.home, train_perc, n_samples=n_samples)
    elif opt.jellyfishseparated:
        train, test = prepare_jellyfish_separated(opt.home)
    else:
        train, test = prepare_root(opt.home, train_perc)

    # Create folder to keep the split files.
    split_file_root = "../split-files"
    if not os.path.exists(split_file_root):
        os.mkdir(split_file_root)

    # Make the train data file mapping.
    train_map_path = os.path.join(split_file_root, "jellyfish-train-map.csv")
    print("Writing training data mapping in {0:s}.".format(train_map_path))
    with open(train_map_path, 'w') as fp:
        for value in train:
            line = value[0] + ',' + ','.join([str(v) for v in value[1]]) + '\n'
            fp.write(line)

    # Make the test data file mapping.
    test_map_path = os.path.join(split_file_root, "jellyfish-test-map.csv")
    print("Writing testing data mapping in {0:s}.".format(test_map_path))
    with open(test_map_path, 'w') as fp:
        for value in test:
            line = value[0] + ',' + ','.join([str(v) for v in value[1]]) + '\n'
            fp.write(line)

    print("Done.")
