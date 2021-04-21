import os
import warnings


def mkdir(path):
    if not os.path.exists(path):
        print("Making {0:s}".format(path))
        os.makedirs(path)


def download_data(src_folder, ds, rm_zip=False):
    path = src_folder + '/' + ds
    if not os.path.exists(path + '.zip'):
        print("========== Downloading 7-scenes Data:", ds, "==========")

        os.system('wget http://download.microsoft.com/download/2/8/5/'
                  + '28564B23-0828-408F-8631-23B1EFF1DAC8/' + ds + '.zip')
        os.system('unzip ' + ds + '.zip')
        if rm_zip:
            os.system('rm ' + ds + '.zip')

        sequences = os.listdir(path)

        for file_ in sequences:
            if file_.endswith('.zip'):
                print("Unpacking", file_)
                os.system('unzip ' + path + '/' + file_ + ' -d ' + path)
                if rm_zip:
                    os.system('rm ' + path + '/' + file_)
    else:
        print("File \'" + path + '.zip' +
              "\' elready exists, skipping download.")


def link_frames(root, name, split_file, focal_length, size=None):
    kind = split_file.strip().split('Split')[0].lower()
    # create subfolders
    mkdir(os.path.join(root, name, kind, 'rgb'))
    mkdir(os.path.join(root, name, kind, 'poses'))
    mkdir(os.path.join(root, name, kind, 'calibration'))

    # read the split file
    with open(os.path.join(root, 'raw', name, split_file), 'r') as f:
        split = f.readlines()
    # map sequences to folder names
    split = ['seq-' + s.strip()[8:].zfill(2) for s in split]

    for seq in split:
        files = os.listdir(os.path.join(root, 'raw', name, seq))

        # link images
        images = [f for f in files if f.endswith('color.png')]
        count = 0
        for img in images[0:size]:
            os.system('ln -sf '
                      + os.path.join(root, 'raw', name, seq, img) + ' '
                      + os.path.join(root, name, kind, 'rgb', seq + '-' + img))
            count = count + 1
        print("Linked {:d} images.".format(count))

        # link folders
        poses = [f for f in files if f.endswith('pose.txt')]
        count = 0
        for pose in poses[0:size]:
            os.system('ln -sf '
                      + os.path.join(root, 'raw', name, seq, pose) + ' '
                      + os.path.join(root, name, kind, 'poses', seq + '-' + pose))
            count = count + 1
        print("Linked {:d} folders.".format(count))

        # create calibration files
        count = 0
        for i in range(len(images[0:size])):
            fn = '{0:s}-frame-{1:s}.calibration.txt'.format(
                seq, str(i).zfill(6))
            with open(os.path.join(root, name, kind, 'calibration', fn), 'w') as g:
                g.write(str(focal_length))
                count = count + 1
        print("Written {:d} calibration files.".format(count))


def make_frame_lists(root, name, prefix, **kwargs):
    """
    This function makes two files in the `path`. One file contains a list of routes to all
    train images (with poses, calibration etc.) and the other file contains a list of routes to all 
    test images (with poses). We will use this from now on because the `link_frames()` function 
    is not very space efficient, also very slow.
    """
    focal_length = kwargs['focal_length']
    include_rendered_depth = kwargs['include_rendered_depth']
    include_precomputed_cam_coord = kwargs['include_precomputed_cam_coord']
    fname = name + '-' + prefix + '-map.csv'
    path = os.path.join(root, fname)
    print("Saving in {0:s}".format(path))
    with open(path, 'w') as fp:
        scene_path = os.path.join(root, 'raw', name)
        split_file = os.path.join(scene_path, 'TrainSplit.txt') if prefix == 'train' \
            else os.path.join(scene_path, 'TestSplit.txt')
        with open(split_file, 'r') as sf:
            seqs = sorted([s.strip()[0:3] + '-' + '{:s}'.format(s.strip()[-1].zfill(2))
                           for s in sf.readlines()])
            for seq in seqs:
                print("Collating {0:s} in {1:s}".format(seq, path))
                seq_path = os.path.join(scene_path, seq)
                files = sorted(os.listdir(seq_path))
                images = [f for f in files if f.endswith('color.png')]
                poses = [f for f in files if f.endswith('pose.txt')]
                depths = [f for f in files if f.endswith('depth.png')]

                inits = []
                if include_rendered_depth:
                    init_path = os.path.join(root, 'rendered-depth')
                    if os.path.exists(init_path):
                        init_files = sorted(os.listdir(init_path))
                        inits = [f for f in init_files if f.startswith(seq)]
                    else:
                        warnings.warn("{0:s} not found, skipping.".format(init_path))

                eyes = []
                if include_precomputed_cam_coord:
                    eye_path = os.path.join(scene_path, 'precomputed-cam-coord')
                    if os.path.exists(eye_path):
                        eye_files = sorted(os.listdir(eye_path))
                        eyes = [f for f in files if f.startswith(seq)]
                    else:
                        warnings.warn("{0:s} not found, skipping.".format(eye_path))

                for i in range(len(images)):
                    image = os.path.join(seq_path, images[i])
                    pose = os.path.join(seq_path, poses[i])
                    depth = os.path.join(seq_path, depths[i])
                    init = os.path.join(seq_path, inits[i]) if len(inits) else ''
                    eye = os.path.join(seq_path, eyes[i]) if len(eyes) else ''
                    fp.write(image + ',' + pose + ',' + depth + ',' \
                            + init + ',' + eye + ',' \
                            + str(focal_length) + '\n')


if __name__ == "__main__":

    # name of the folder where we download the original 7scenes dataset to
    # we restructure the dataset by creating symbolic links to that folder
    data_home = os.environ['DATA_HOME']
    root = os.path.join(data_home, 'sevenscenes')
    raw_path = os.path.join(root, 'raw')
    focal_length = 525.0

    # download the original 7 scenes dataset for poses and images
    mkdir(raw_path)
    os.chdir(raw_path)

    # for ds in ['chess', 'fire', 'heads', 'office', 'pumpkin', 'redkitchen', 'stairs']:
    for name in ['chess']:
        download_data(raw_path, name)
        make_frame_lists(root, name, 'train', focal_length=focal_length, \
                include_rendered_depth=True, include_precomputed_cam_coord=True)
        make_frame_lists(root, name, 'test', focal_length=focal_length, \
                include_rendered_depth=True, include_precomputed_cam_coord=True)
        # print("Linking files in ... " + root)
        # link_frames(root, name, 'TrainSplit.txt', focal_length)
        # link_frames(root, name, 'TestSplit.txt', focal_length)
