import torch
import torch.nn.functional as F
import numpy as np
import cv2

import dsacstar

import os
import sys
import time
import argparse
import math
from datetime import datetime

from dataset import CamLocDataset, JellyfishDataset
from network import Network

parser = argparse.ArgumentParser(
    description='Test a trained network on a specific scene.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument(
    'scene', help='name of a scene in the dataset folder, e.g. Cambridge_GreatCourt')

parser.add_argument(
    'network', help='file name of a network trained for the scene')

parser.add_argument('--hypotheses', '-hyps', type=int, default=64,
                    help='number of hypotheses, i.e. number of RANSAC iterations')

parser.add_argument('--threshold', '-t', type=float, default=10,
                    help='inlier threshold in pixels (RGB) or centimeters (RGB-D)')

parser.add_argument('--inlieralpha', '-ia', type=float, default=100,
                    help='alpha parameter of the soft inlier count; controls the softness of the hypotheses score distribution; lower means softer')

parser.add_argument('--maxpixelerror', '-maxerrr', type=float, default=100,
                    help='maximum reprojection (RGB, in px) or 3D distance (RGB-D, in cm) error when checking pose consistency towards all measurements; error is clamped to this value for stability')

parser.add_argument('--mode', '-m', type=int, default=1, choices=[1, 2],
                    help='test mode: 1 = RGB, 2 = RGB-D')

parser.add_argument('--tiny', '-tiny', action='store_true',
                    help='Load a model with massively reduced capacity for a low memory footprint.')

now = datetime.now()
parser.add_argument('--session', '-sid', default=now.strftime("%d-%m-%y-%H-%M-%S") ,
                    help='custom session name appended to output files, useful to separate different runs of a script')

opt = parser.parse_args()

# setup dataset
if opt.mode < 2:
    opt.mode = 0  # we do not load ground truth scene coordinates when testing

if not os.path.exists(opt.network):
    raise FileNotFoundError("Error: file {0:s} not found.".format(opt.network))
    sys.exit(1)

# testset = CamLocDataset(opt.scene + "/test", mode=opt.mode)
testset = JellyfishDataset(opt.scene, mode=opt.mode)
testset_loader = torch.utils.data.DataLoader(testset, shuffle=False, num_workers=6)

# load network
network = Network(torch.zeros((3)), opt.tiny)
network.load_state_dict(torch.load(opt.network))
network = network.cuda()
network.eval()

network_name = os.path.split(opt.network)[-1].split('.')[0]
test_log = open('log_test_{0:s}_{1:s}.txt'.format(network_name, opt.session), 'w', 1)
pose_log = open('log_poses_{0:s}_{1:s}.txt'.format(network_name, opt.session), 'w', 1)

print('Test images found: ', len(testset))

# keep track of rotation and translation errors for calculation of the median error
rErrs = []
tErrs = []
avg_time = 0

pct5 = 0
pct2 = 0
pct1 = 0

with torch.no_grad():

    for image, gt_pose, init, focal_length, file_, timestamp in testset_loader:

        focal_length = float(focal_length[0])
        file_ = file_[0].split('/')[-1]  # remove path from file name
        gt_pose = gt_pose[0]
        image = image.cuda()

        start_time = time.time()

        # predict scene coordinates and neural guidance
        scene_coordinates = network(image)
        scene_coordinates = scene_coordinates.cpu()

        out_pose = torch.zeros((4, 4))

        if opt.mode < 2:
            # pose from RGB
            dsacstar.forward_rgb(
                scene_coordinates,
                out_pose,
                opt.hypotheses,
                opt.threshold,
                focal_length,
                # principal point assumed in image center
                float(image.size(3) / 2),
                float(image.size(2) / 2),
                opt.inlieralpha,
                opt.maxpixelerror,
                network.OUTPUT_SUBSAMPLE)
        else:
            # pose from RGB-D
            dsacstar.forward_rgbd(
                scene_coordinates,
                init,  # contains precalculated camera coordinates
                out_pose,
                opt.hypotheses,
                opt.threshold,
                opt.inlieralpha,
                opt.maxpixelerror)

        avg_time += time.time()-start_time

        # calculate pose errors
        t_err = float(torch.norm(gt_pose[0:3, 3] - out_pose[0:3, 3]))

        gt_R = gt_pose[0:3, 0:3].numpy()
        out_R = out_pose[0:3, 0:3].numpy()

        r_err = np.matmul(out_R, np.transpose(gt_R))
        r_err = cv2.Rodrigues(r_err)[0]
        r_err = np.linalg.norm(r_err) * 180 / math.pi

        print("\nRotation Error: {0:.2f}deg, Translation Error: {1:.1f}cm"
                .format(r_err, t_err*100))

        rErrs.append(r_err)
        tErrs.append(t_err * 100)

        if r_err < 5 and t_err < 0.05:
            pct5 += 1
        if r_err < 2 and t_err < 0.02:
            pct2 += 1
        if r_err < 1 and t_err < 0.01:
            pct1 += 1

        # write estimated pose to pose file
        out_pose = out_pose.inverse()

        t = out_pose[0:3, 3]

        # rotation to axis angle
        rot, _ = cv2.Rodrigues(out_pose[0:3, 0:3].numpy())
        angle = np.linalg.norm(rot)
        axis = rot / angle

        # axis angle to quaternion
        q_w = math.cos(angle * 0.5)
        q_xyz = math.sin(angle * 0.5) * axis

        print(timestamp, file_) 
        line = "{0:s}\t{1:s}".format(timestamp[0], file_)
        print(q_w, q_xyz[0][0], q_xyz[1][0], q_xyz[2][0])
        line = line + "\t{2:.6f}\t{3:.6f}\t{4:.6f}\t{5:.6f}"\
                .format(q_w, q_xyz[0][0], q_xyz[1][0], q_xyz[2][0])
        print(t)
        line = line + "\t{6:.6f}\t{7:.6f}\t{8:.6f}".format(t[0].item(), t[1].tiem(), t[2].item())
        print(r_err, t_err)
        line = line + "\t{9:.6f}\t{10:.6f}\n".format(r_err, t_err)
        pose_log.write(line)

median_idx = int(len(rErrs)/2)
tErrs.sort()
rErrs.sort()
avg_time /= len(rErrs)

print("\n===================================================")
print("\nTest complete.")

print('\nAccuracy:')
print('\n5cm5deg: {0:.1f}%%'.format(pct5 / len(rErrs) * 100))
print('2cm2deg: {0:.1f}%%'.format(pct2 / len(rErrs) * 100))
print('1cm1deg: {0:.1f}%%'.format(pct1 / len(rErrs) * 100))

print("\nMedian Error: {0:.1f}deg, {1:.1f}cm".format(rErrs[median_idx], tErrs[median_idx]))
print("Avg. processing time: {0:4.1f}ms".format(avg_time * 1000))
test_log.write('Median rError: {0:f}\nMedian tError: {1:f}\nAvg. Time{2:f}\n'
        .format(rErrs[median_idx], tErrs[median_idx], avg_time))

test_log.close()
pose_log.close()
