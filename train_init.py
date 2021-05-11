import os
import sys
import time
import argparse
import math
from datetime import datetime

import torch
import torch.optim as optim
from torchvision import utils

from dataset import CamLocDataset, CamLocDatasetLite, JellyfishDataset
from network import Network

def parse_args():
    now = datetime.now()

    parser = argparse.ArgumentParser(
        description='Initialize a scene coordinate regression network.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('scene', help='name of a scene in the dataset folder')
    parser.add_argument('network', help='output file name for the network')
    parser.add_argument('--modelpath', '-mp', type=str, default='models',
                        help='path where the models will be saved')
    parser.add_argument('--checkpoint', '-cp', type=str, default=None,
                        help='path to saved model (*.ann file) for retraining')
    parser.add_argument('--startepoch', '-se', type=int, default=0,
                        help='the value of starting epoch, is useful if a model is to be '
                        + 'retrained from checkpoint, the new run will start from this epoch')
    parser.add_argument('--learningrate', '-lr', type=float,
                        default=0.0001, help='learning rate')
    parser.add_argument('--iterations', '-iter', type=int,
                        help='number of training iterations, i.e. numer of model updates')
    parser.add_argument('--epochs', '-e', type=int,
                        help='number of training epochs, i.e. |iterations / no. training images|')
    parser.add_argument('--ignorebadat', '-ibe', type=int, default=float('inf'),
                        help='the epoch at wich we start ignoring the bad images')
    parser.add_argument('--inittolerance', '-itol', type=float, default=0.1,
                        help='switch to reprojection error optimization when '
                        + 'predicted scene coordinate is within this tolerance '
                        + 'threshold to the ground truth scene coordinate, in meters')
    parser.add_argument('--mindepth', '-mind', type=float, default=0.1,
                        help='enforce predicted scene coordinates to be this far in front '
                        + 'of the camera plane, in meters')
    parser.add_argument('--maxdepth', '-maxd', type=float, default=1000,
                        help='enforce that scene coordinates are at most this far in front '
                        + 'of the camera plane, in meters')
    parser.add_argument('--targetdepth', '-td', type=float, default=10,
                        help='if ground truth scene coordinates are unknown, use a proxy '
                        + 'scene coordinate on the pixel ray with this distance from '
                        + 'the camera, in meters')
    parser.add_argument('--softclamp', '-sc', type=float, default=100,
                        help='robust square root loss after this threshold, in pixels')
    parser.add_argument('--hardclamp', '-hc', type=float, default=1000,
                        help='clamp loss with this threshold, in pixels')
    parser.add_argument('--mode', '-m', type=int, default=1, choices=range(3),
                        help='training mode: 0 = RGB only (no ground truth scene coordinates), '
                        + '1 = RGB + ground truth scene coordinates, 2 = RGB-D')
    parser.add_argument('--sparse', '-sparse', action='store_true',
                        help='for mode 1 (RGB + ground truth scene coordinates) use sparse scene '
                        + 'coordinate initialization targets (eg. for Cambridge) instead of '
                        + 'rendered depth maps (eg. for 7scenes and 12scenes).')
    parser.add_argument('--tiny', '-tiny', action='store_true',
                        help='Train a model with massively reduced capacity for a low memory footprint.')
    parser.add_argument('--session', '-sid', default=now.strftime("%d-%m-%y-%H-%M-%S"),
                        help='custom session name appended to output files, useful to '
                        + 'separate different runs of a script')
    opt = parser.parse_args()

    return opt


def decide_epochs_iterations(trainset, opt):
    n = len(trainset)
    if not opt.iterations and not opt.epochs:
        # iterations: 1000000, frames: 4000
        # iterations: 197250, frames: 789 ... etc.
        iterations = int((1000000 / 4000) * n)
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
    return epochs, iterations


def compute_mean_scene_coords(trainset_loader):
    mean = torch.zeros((3))
    count = 0
    for image, gt_pose, gt_coords, focal_length, _, _ in trainset_loader:

        if use_init:
            # use mean of ground truth scene coordinates

            gt_coords = gt_coords[0]
            gt_coords = gt_coords.view(3, -1)

            coord_mask = gt_coords.abs().sum(0) > 0
            if coord_mask.sum() > 0:
                gt_coords = gt_coords[:, coord_mask]

                mean += gt_coords.sum(1)
                count += coord_mask.sum()
        else:
            # use mean of camera position
            mean += gt_pose[0, 0:3, 3]
            count += 1
        if count % 100 == 0:
            print("Computed mean scene coordinate of {0:d} frames.".format(count))

    mean = mean / count
    return mean


def prepare_network(opt):
    if opt.checkpoint:
        # load network
        network = Network(mean, opt.tiny)
        network.load_state_dict(torch.load(opt.checkpoint))
        network = network.cuda()
        network.train()
        print("Successfully loaded {0:s}.".format(opt.checkpoint))
    else:
        # create network
        network = Network(mean, opt.tiny)
        network = network.cuda()
        network.train()
        print("Successfully created initial network.")
    return network


def generate_target_reprojection_pixel_positions(network):
    # generate grid of target reprojection pixel positions
    pixel_grid = torch.zeros((2,
                              # 5000px is max limit of image size, increase if needed
                              math.ceil(5000 / network.OUTPUT_SUBSAMPLE),
                              math.ceil(5000 / network.OUTPUT_SUBSAMPLE)))

    for x in range(0, pixel_grid.size(2)):
        for y in range(0, pixel_grid.size(1)):
            pixel_grid[0, y, x] = x * network.OUTPUT_SUBSAMPLE + \
                network.OUTPUT_SUBSAMPLE / 2
            pixel_grid[1, y, x] = y * network.OUTPUT_SUBSAMPLE + \
                network.OUTPUT_SUBSAMPLE / 2
    pixel_grid = pixel_grid.cuda()
    return pixel_grid


def save_sanitycheck_images(image, time_stamp, count):
    folder = 'image-sanitycheck'
    if not os.path.exists(folder):
        os.makedirs(folder)
    path = os.path.join(folder, '{0:d}-{1:s}-unfished.png'.format(count, time_stamp))
    print("Saving", path)
    utils.save_image(image, path)


def create_camera_calibration_matrix(focal_length, image):
    focal_length = float(focal_length[0])
    cam_mat = torch.eye(3)
    cam_mat[0, 0] = focal_length
    cam_mat[1, 1] = focal_length
    cam_mat[0, 2] = image.size(3) / 2
    cam_mat[1, 2] = image.size(2) / 2
    cam_mat = cam_mat.cuda()

    return cam_mat


def compute_loss_rgbd(scene_coords, gt_coords):
    scene_coords = scene_coords[0].view(3, -1)
    gt_coords = gt_coords[0].view(3, -1).cuda()

    # check for invalid ground truth scene coordinates
    gt_coords_mask = gt_coords.abs().sum(0) > 0

    loss = torch.norm(scene_coords - gt_coords,
                      dim=0, p=2)[gt_coords_mask]
    loss = loss.mean()
    num_valid_sc = gt_coords_mask.float().mean()

    return loss, num_valid_sc


def compute_loss_rgb(opt, pixel_grid, scene_coords, gt_pose, gt_coords, cam_mat, use_init):
    # crop ground truth pixel positions to prediction size
    pixel_grid_crop = pixel_grid[:, 0:scene_coords.size(
        2), 0:scene_coords.size(3)].clone()
    pixel_grid_crop = pixel_grid_crop.view(2, -1)

    # make 3D points homogeneous
    ones = torch.ones(
        (scene_coords.size(0), 1, scene_coords.size(2), scene_coords.size(3)))
    ones = ones.cuda()

    scene_coords = torch.cat((scene_coords, ones), 1)
    scene_coords = scene_coords.squeeze().view(4, -1)

    # prepare pose for projection operation
    gt_pose = gt_pose[0].inverse()[0:3, :]
    gt_pose = gt_pose.cuda()

    # scene coordinates to camera coordinate
    camera_coords = torch.mm(gt_pose, scene_coords)

    # re-project predicted scene coordinates
    reprojection_error = torch.mm(cam_mat, camera_coords)
    reprojection_error[2].clamp_(
        min=opt.mindepth)  # avoid division by zero
    reprojection_error = reprojection_error[0:2] / \
        reprojection_error[2]

    reprojection_error = reprojection_error - pixel_grid_crop
    reprojection_error = reprojection_error.norm(2, 0)

    # check predicted scene coordinate for various constraints
    # behind or too close to camera plane
    invalid_min_depth = camera_coords[2] < opt.mindepth
    # check for very large reprojection errors
    invalid_repro = reprojection_error > opt.hardclamp

    gt_coords_mask, target_camera_coords = None, None
    if use_init:
        # ground truth scene coordinates available, transform to uniform
        gt_coords = torch.cat((gt_coords.cuda(), ones), 1)
        gt_coords = gt_coords.squeeze().view(4, -1)

        # check for invalid/unknown ground truth scene coordinates (all zeros)
        gt_coords_mask = torch.abs(gt_coords[0:3]).sum(0) == 0

        # scene coordinates to camera coordinate
        target_camera_coords = torch.mm(gt_pose, gt_coords)

        # distance between predicted and ground truth coordinates
        gt_coord_dist = torch.norm(
            camera_coords - target_camera_coords, dim=0, p=2)

        # check for additional constraints regarding ground truth scene coordinates
        # too far from ground truth scene coordinates
        invalid_gt_distance = gt_coord_dist > opt.inittolerance
        # filter unknown ground truth scene coordinates
        invalid_gt_distance[gt_coords_mask] = 0

        # combine all constraints
        valid_scene_coordinates = (
            invalid_min_depth + invalid_gt_distance + invalid_repro) == 0
    else:
        # no ground truth scene coordinates available, 
        # enforce max distance of predicted coordinates
        invalid_max_depth = camera_coords[2] > opt.maxdepth

        # combine all constraints
        valid_scene_coordinates = (
            invalid_min_depth + invalid_max_depth + invalid_repro) == 0

    return (pixel_grid_crop, scene_coords, gt_pose, camera_coords, reprojection_error, \
            gt_coords, gt_coords_mask, target_camera_coords, gt_coord_dist, valid_scene_coordinates)


def assemble_loss(opt, use_init, pixel_grid_crop, scene_coords, gt_pose, camera_coords, \
        reprojection_error, gt_coords, gt_coords_mask, target_camera_coords, gt_coord_dist, \
        valid_scene_coordinates): 
    loss = 0
    
    if num_valid_sc > 0:
    
        # reprojection error for all valid scene coordinates
        reprojection_error = reprojection_error[valid_scene_coordinates]
    
        # calculate soft clamped l1 loss of reprojection error
        loss_l1 = reprojection_error[reprojection_error <=
                                     opt.softclamp]
        loss_sqrt = reprojection_error[reprojection_error >
                                       opt.softclamp]
        loss_sqrt = torch.sqrt(opt.softclamp*loss_sqrt)
    
        loss += (loss_l1.sum() + loss_sqrt.sum())
    
    if num_valid_sc < scene_coords.size(1):
    
        invalid_scene_coordinates = (valid_scene_coordinates == 0)
    
        if use_init:
            # 3D distance loss for all invalid scene coordinates 
            # where the ground truth is known
            invalid_scene_coordinates[gt_coords_mask] = 0
    
            loss += gt_coord_dist[invalid_scene_coordinates].sum()
        else:
            # generate proxy coordinate targets with constant depth assumption
            target_camera_coords = pixel_grid_crop
            target_camera_coords[0] -= image.size(3) / 2
            target_camera_coords[1] -= image.size(2) / 2
            target_camera_coords *= opt.targetdepth
            target_camera_coords /= focal_length
            # make homogeneous
            target_camera_coords = torch.cat(
                (target_camera_coords, torch.ones(
                    (1, target_camera_coords.size(1))).cuda()), 0)
    
            # distance
            loss += torch.abs(camera_coords[:, invalid_scene_coordinates] -
                              target_camera_coords[:, invalid_scene_coordinates]).sum()
    return loss


def save_bad_images(opt, bad_images):
    print('Saving bad images.')
    bad_images_log = open('log_init_bad_{0:s}_{1:s}.txt'.format(
        opt.network, opt.session), 'w', 1)
    for ts in bad_images.keys():
        bad_images_log.write("{0:.2f}\t{1:.2f}\t{2:s}\n"\
                .format(bad_images[ts][0], bad_images[ts][1], bad_images[ts][2]))
    bad_images_log.close()



if __name__ == "__main__":

    opt = parse_args()
    use_init = opt.mode > 0

    model_root = opt.modelpath
    if not os.path.exists(model_root):
        os.mkdir(model_root)

    # for RGB-D initialization, we utilize ground truth scene coordinates
    # as in mode 2 (RGB + ground truth scene coordinates)
    # trainset = CamLocDataset(opt.scene + "/train", mode=min(opt.mode, 1), 
    #                                               sparse=opt.sparse, augment=True)
    # trainset = CamLocDatasetLite(opt.scene, mode=min(opt.mode, 1), sparse=opt.sparse, augment=True)
    print("Preparing dataset, will take a while ...")
    trainset = JellyfishDataset(opt.scene, mode=min(
        opt.mode, 1), sparse=opt.sparse, augment=True)
    trainset_loader = torch.utils.data.DataLoader(
        trainset, shuffle=True, num_workers=6)

    print("Found {0:d} training images in {1:s}.".format(
        len(trainset), opt.scene))

    # decide iterations from the number of training data
    epochs, iterations = decide_epochs_iterations(trainset, opt)
    print("Total epochs: {0:d}, Total iterations: {1:d}".format(
        epochs, iterations))

    print("Calculating mean scene coordinates ...")
    mean = compute_mean_scene_coords(trainset_loader)
    print("Done. Mean: {0:.2f}, {1:.2f}, {2:.2f}\n".format(
        mean[0], mean[1], mean[2]))

    network = prepare_network(opt)
    optimizer = optim.Adam(network.parameters(), lr=opt.learningrate)

    # keep track of training progress
    # train_iter_log = open('log_init_%s_%s.txt' % (opt.scene, opt.session), 'w', 1)
    train_iter_log = open('log_init_iter_{0:s}_{1:s}.txt'.format(
        opt.network, opt.session), 'w', 1)
    train_epoch_log = open('log_init_epoch_{0:s}_{1:s}.txt'.format(
        opt.network, opt.session), 'w', 1)

    pixel_grid = generate_target_reprojection_pixel_positions(network)

    iteration = 0
    sanity_check = True
    min_epoch, max_epoch = opt.startepoch + 1, opt.startepoch + epochs + 1
    ignore_bad_at = opt.ignorebadat
    bad_images = {}
    for epoch in range(min_epoch, max_epoch):

        now = datetime.now()
        print("========== Stamp: {0:s} / Epoch: {1:d} =========="
              .format(now.strftime("%d/%m/%y [%H-%M-%S]"), epoch))

        count = 0
        mean_loss = 0.0
        mean_num_valid_sc = 0.0
        for image, gt_pose, gt_coords, focal_length, time_stamp, file_path in trainset_loader:
            if len(bad_images) == 0 or time_stamp[0] not in bad_images:
                
                if sanity_check and count < 10 and epoch < 2:
                    save_sanitycheck_images(image, time_stamp[0], count)

                start_time = time.time()

                # create camera calibartion matrix
                cam_mat = create_camera_calibration_matrix(focal_length, image)
                scene_coords = network(image.cuda())

                # calculate loss dependant on the mode

                if opt.mode == 2:
                    # === RGB-D mode, optimize 3D distance to ground truth scene coordinates ======
                    loss, num_valid_sc = compute_loss_rgbd(scene_coords, gt_coords)
                else:
                    # === RGB mode, optmize a variant of the reprojection error ===================
                    (pixel_grid_crop, scene_coords, gt_pose, camera_coords, reprojection_error, \
                            gt_coords, gt_coords_mask, target_camera_coords, gt_coord_dist, 
                            valid_scene_coordinates) = compute_loss_rgb(opt, pixel_grid, \
                                    scene_coords, gt_pose, gt_coords, cam_mat, use_init)
                    
                    num_valid_sc = int(valid_scene_coordinates.sum())
                    
                    # assemble loss
                    loss = assemble_loss(opt, use_init, pixel_grid_crop, scene_coords, gt_pose, \
                            camera_coords, reprojection_error, gt_coords, gt_coords_mask, \
                            target_camera_coords, gt_coord_dist, valid_scene_coordinates) 

                    loss /= scene_coords.size(1)
                    num_valid_sc /= scene_coords.size(1)

                if epoch > ignore_bad_at and num_valid_sc * 100 < 90.0:
                    bad_images[time_stamp[0]] = [loss, num_valid_sc * 100, epoch, file_path[0]]

                loss.backward()		# calculate gradients (pytorch autograd)
                optimizer.step()	# update all model parameters
                optimizer.zero_grad()

                print('Epoch: {0:d}/{1:d},\tIteration: {2:6d},\tLoss: {3:.1f},'\
                        .format(epoch, max_epoch-1, iteration, loss)
                        + '\tValid: {0:.1f}%,\tTime: {1:.2f}s\tTs: {2:s}'\
                                .format(num_valid_sc*100, time.time()-start_time, time_stamp[0]), \
                                flush=True)
                train_iter_log.write('{0:d}\t{1:f}\t{2:f}\t{3:s}\t{4:s}\n'.format(
                    iteration, loss, num_valid_sc, time_stamp[0], file_path[0]))

                mean_loss = mean_loss + loss
                mean_num_valid_sc = mean_num_valid_sc + (num_valid_sc * 100)

                iteration = iteration + 1
                count = count + 1

                del loss
            
            else:
                print("== {0:s}: bad image, skipping.".format(time_stamp[0]))

        mean_loss = mean_loss / count
        mean_num_valid_sc = mean_num_valid_sc / count
        train_epoch_log.write('{0:d}\t{1:f}\t{2:f}\n'.format(
            epoch, mean_loss, mean_num_valid_sc))

        if epoch % 25 == 0 or epoch == min_epoch or epoch == max_epoch:
            model_path = os.path.join(
                model_root, "{0:s}-e{1:d}-init.ann".format(opt.network, epoch))
            print('Saving snapshot of the network to {:s}.'.format(model_path))
            torch.save(network.state_dict(), model_path)
        
        if len(bad_images) > 0:
            save_bad_images(opt, bad_images)

    print('Done without errors.')
    train_iter_log.close()
    train_epoch_log.close()
