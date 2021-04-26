import torch
import torch.optim as optim

import os
import argparse
import time
import random
from datetime import datetime

import dsacstar

from dataset import CamLocDataset, JellyfishDataset
from network import Network

parser = argparse.ArgumentParser(
    description='Train scene coordinate regression in an end-to-end fashion.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('scene', help='name of a scene in the dataset folder')
parser.add_argument(
    'network_in', help='file name of a network initialized for the scene')
parser.add_argument('network_out', help='output file name for the new network')
parser.add_argument('--modelpath', '-mp', type=str, default='models',
                    help='path where the models will be saved')
parser.add_argument('--hypotheses', '-hyps', type=int, default=64,
                    help='number of hypotheses, i.e. number of RANSAC iterations')
parser.add_argument('--threshold', '-t', type=float, default=10,
                    help='inlier threshold in pixels (RGB) or centimeters (RGB-D)')
parser.add_argument('--inlieralpha', '-ia', type=float, default=100,
                    help='alpha parameter of the soft inlier count; ' 
                    + 'controls the softness of the hypotheses score distribution; lower means softer')
parser.add_argument('--learningrate', '-lr', type=float, default=0.000001,
                    help='learning rate')
parser.add_argument('--iterations', '-it', type=int,
                    help='number of training iterations, i.e. network parameter updates')
parser.add_argument('--epochs', '-e', type=int,
                    help='number of training epochs, i.e. |iterations / no. training images|')
parser.add_argument('--weightrot', '-wr', type=float, default=1.0,
                    help='weight of rotation part of pose loss')
parser.add_argument('--weighttrans', '-wt', type=float, default=100.0,
                    help='weight of translation part of pose loss')
parser.add_argument('--softclamp', '-sc', type=float, default=100,
                    help='robust square root loss after this threshold')
parser.add_argument('--maxpixelerror', '-maxerrr', type=float, default=100,
                    help='maximum reprojection (RGB, in px) or 3D distance (RGB-D, in cm)' 
                    + ' error when checking pose consistency towards all measurements;' 
                    + ' error is clamped to this value for stability')
parser.add_argument('--mode', '-m', type=int, default=1, choices=[1, 2],
                    help='test mode: 1 = RGB, 2 = RGB-D')
parser.add_argument('--tiny', '-tiny', action='store_true',
                    help='Train a model with massively reduced capacity for a low memory footprint.')
now = datetime.now()
parser.add_argument('--session', '-sid', default=now.strftime("%d-%m-%y-%H-%M-%S"),
                    help='custom session name appended to output files. ' 
                    + 'Useful to separate different runs of the program')

# parse now
opt = parser.parse_args()

# check if model output folder exists
model_root = opt.modelpath
if not os.path.exists(model_root):
    raise NotADirectoryError("Error: folder {0:s} not found, ".format(model_root)
                             + "you might need to run `train_init.py` first?")
    sys.exit(1)

# check if input network is there
if not os.path.exists(opt.network_in):
    raise FileNotFoundError(
        "Error: model file {0:s} not found.".format(opt.network_in))
    sys.exit(1)

# use only photometric augmentation, not rotation and scaling
# trainset = CamLocDataset(opt.scene + "/train", mode=(0 if opt.mode < 2 else opt.mode),
#                          augment=True, aug_rotation=0, aug_scale_min=1, aug_scale_max=1)
trainset = JellyfishDataset(opt.scene, mode=(0 if opt.mode < 2 else opt.mode),
                            augment=True, aug_rotation=0, aug_scale_min=1, aug_scale_max=1)
trainset_loader = torch.utils.data.DataLoader(
    trainset, shuffle=True, num_workers=6)

print("Found {0:d} training images for {1:s}.".format(
    len(trainset), opt.scene))

# load network
network = Network(torch.zeros((3)), opt.tiny)
network.load_state_dict(torch.load(opt.network_in))
network = network.cuda()
network.train()
print("Successfully loaded {0:s}.".format(opt.network_in))

optimizer = optim.Adam(network.parameters(), lr=opt.learningrate)

# decide iterations from the number of training data
n = len(trainset)
if not opt.iterations and not opt.epochs:
    # iterations: 100000, frames: 4000
    # iterations: 197250, frames: 789 ... etc.
    iterations = int((100000 / 4000) * n)
    epochs = int(iterations / n)
elif not opt.iterations and opt.epochs:
    iterations = opt.epochs * n
    epochs = opt.epochs
elif opt.iterations and not opt.epochs:
    iterations = opt.iterations if opt.iterations >= n else n
    epochs = int(iterations / n)
else:
    epochs, iterations = opt.epochs, opt.iterations
epochs = 1 if epochs < 1 else epochs
print("Total epochs: {0:d}, Total iterations: {1:d}".format(
    epochs, iterations))

# keep track of training progress
# train_iter_log = open('log_e2e__.txt' % (opt.scene, opt.session), 'w', 1)
train_iter_log = open('log_e2e_iter_{0:s}_{1:s}.txt'.format(
    opt.network_out, opt.session), 'w', 1)
train_epoch_log = open('log_e2e_epoch_{0:s}_{1:s}.txt'.format(
    opt.network_out, opt.session), 'w', 1)

training_start = time.time()

iteration = 0
for epoch in range(1, epochs+1):

    now = datetime.now()
    print("========== Stamp: {0:s} / Epoch: {1:d} =========="
          .format(now.strftime("%d/%m/%y [%H-%M-%S]"), epoch))

    count = 0
    mean_loss = 0.0
    for image, pose, camera_coordinates, focal_length, _, _ in trainset_loader:

        start_time = time.time()

        focal_length = float(focal_length[0])
        pose = pose[0]

        # predict scene coordinates
        scene_coordinates = network(image.cuda())
        scene_coordinate_gradients = torch.zeros(scene_coordinates.size())

        if opt.mode == 2:
            # RGB-D mode
            loss = dsacstar.backward_rgbd(
                scene_coordinates.cpu(),
                camera_coordinates,
                scene_coordinate_gradients,
                pose,
                opt.hypotheses,
                opt.threshold,
                opt.weightrot,
                opt.weighttrans,
                opt.softclamp,
                opt.inlieralpha,
                opt.maxpixelerror,
                random.randint(0, 1000000))  # used to initialize random number generator in C++

        else:
            # RGB mode
            loss = dsacstar.backward_rgb(
                scene_coordinates.cpu(),
                scene_coordinate_gradients,
                pose,
                opt.hypotheses,
                opt.threshold,
                focal_length,
                # principal point assumed in image center
                float(image.size(3) / 2),
                float(image.size(2) / 2),
                opt.weightrot,
                opt.weighttrans,
                opt.softclamp,
                opt.inlieralpha,
                opt.maxpixelerror,
                network.OUTPUT_SUBSAMPLE,
                random.randint(0, 1000000))  # used to initialize random number generator in C++

        # update network parameters
        torch.autograd.backward((scene_coordinates),
                                (scene_coordinate_gradients.cuda()))
        optimizer.step()
        optimizer.zero_grad()

        end_time = time.time()-start_time
        print('Epoch: {0:d},\tIteration: {1:6d},\tLoss: {2:.2f},\tTime: {3:.2f}s\n'
              .format(epoch, iteration, loss, end_time), flush=True)

        train_iter_log.write('{0:d}\t{1:f}\n'.format(iteration, loss))
        mean_loss = mean_loss + loss
        count = count + 1
        iteration = iteration + 1

    mean_loss = mean_loss / count
    train_epoch_log.write('{0:d}\t{1:f}\n'.format(epoch, mean_loss))
    if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
        model_path = os.path.join(
            model_root, "{0:s}-e{1:d}-e2e.ann".format(opt.network_out, epoch))
        print('Saving snapshot of the network to {:s}.'.format(model_path))
        torch.save(network.state_dict(), model_path)

print('Done without errors. Time: {0:.1f} minutes.'.format(
    (time.time() - training_start) / 60))
train_iter_log.close()
train_epoch_log.close()
