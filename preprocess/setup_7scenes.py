import os


def mkdir(directory):
    """Checks whether the directory exists and creates it if necessacy."""
    if not os.path.exists(directory):
        os.makedirs(directory)


def download_data(src_folder, ds, rm_zip=False):
    path = src_folder + '/' + ds
    if not os.path.exists(path + '.zip'):
        print("========== Downloading 7scenes Data:", ds, "==========")

        os.system('cd ' + src_folder)
        os.system('wget http://download.microsoft.com/download/2/8/5/'
                  + '28564B23-0828-408F-8631-23B1EFF1DAC8/' + ds + '.zip')
        os.system('unzip ' + ds + '.zip')
        os.system('rm ' + ds + '.zip')

        sequences = os.listdir(path)

        for file_ in sequences:
            if file_.endswith('.zip'):
                print("Unpacking", file_)
                os.system('unzip ' + path + '/' + file_ + ' -d ' + path)
                if rm_zip:
                    os.system('rm ' + path + '/' + file_)
    else:
        print("File \'" + path + '.zip' + "\' elready exists, skipping download.")


def link_frames(target_folder, split_file, variant, size=None):
    # create subfolders
    mkdir(target_folder + variant + '/rgb/')
    mkdir(target_folder + variant + '/poses/')
    mkdir(target_folder + variant + '/calibration/')

    # read the split file
    with open(ds + '/' + split_file, 'r') as f:
        split = f.readlines()
    # map sequences to folder names
    split = ['seq-' + s.strip()[8:].zfill(2) for s in split]

    for seq in split:
        files = os.listdir(ds + '/' + seq)

        # link images
        images = [f for f in files if f.endswith('color.png')]
        count = 0
        for img in images[0:size]:
            os.system('ln -sf ' + src_folder + '/' + ds + '/' + seq + '/' + img + ' ' 
                    + target_folder + variant + '/rgb/' + seq + '-' + img)
            count = count + 1
        print("Linked {:d} images.".format(count))

        # link folders
        poses = [f for f in files if f.endswith('pose.txt')]
        count = 0
        for pose in poses[0:size]:
            os.system('ln -sf ' + src_folder + '/' + ds + '/' + seq + '/' + pose + ' ' 
                    + target_folder + variant + '/poses/' + seq + '-' + pose)
            count = count + 1
        print("Linked {:d} folders.".format(count))

        # create calibration files
        count = 0 
        for i in range(len(images[0:size])):
            with open(target_folder+variant + '/calibration/{0:s}-frame-{1:s}.calibration.txt'
                      .format(seq, str(i).zfill(6)), 'w') as f:
                f.write(str(focallength))
                count = count + 1
        print("Written {:d} calibration files.".format(count))


if __name__ == "__main__":

    # name of the folder where we download the original 7scenes dataset to
    # we restructure the dataset by creating symbolic links to that folder
    data_root = os.environ['DATA_HOME'] 
    src_folder = data_root + '/7scenes/raw'
    focallength = 525.0

    # download the original 7 scenes dataset for poses and images
    mkdir(src_folder)
    os.chdir(src_folder)

    # for ds in ['chess', 'fire', 'heads', 'office', 'pumpkin', 'redkitchen', 'stairs']:
    for ds in ['chess']:
        download_data(src_folder, ds, rm_zip=True)
        target_folder = data_root + '/7scenes/' + ds + '/'
        print("Linking files in ... " + target_folder)
        link_frames(target_folder, 'TrainSplit.txt', 'train')
        link_frames(target_folder, 'TestSplit.txt', 'test')
