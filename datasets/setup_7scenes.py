import os


def mkdir(directory):
    """Checks whether the directory exists and creates it if necessacy."""
    if not os.path.exists(directory):
        os.makedirs(directory)


def download_data(ds, rm_zip=False):
    if not os.path.exists(ds):
        print("=== Downloading 7scenes Data:", ds, "==================")

        os.system('wget http://download.microsoft.com/download/2/8/5/'
                  + '28564B23-0828-408F-8631-23B1EFF1DAC8/' + ds + '.zip')
        os.system('unzip ' + ds + '.zip')
        os.system('rm ' + ds + '.zip')

        sequences = os.listdir(ds)

        for file in sequences:
            if file.endswith('.zip'):
                print("Unpacking", file)
                os.system('unzip ' + ds + '/' + file + ' -d ' + ds)
                if rm_zip:
                    os.system('rm ' + ds + '/' + file)
    else:
        print("File " + ds + ".zip exists, skipping download.")


def link_frames(target_folder, split_file, variant, size=-1):
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
        for img in images[0:size]:
            os.system('ln -sf ' + src_folder + '/' + ds + '/' + seq + '/' + img + ' ' 
                    + target_folder + variant + '/rgb/' + seq + '-' + img)
        # link folders
        poses = [f for f in files if f.endswith('pose.txt')]
        for pose in poses[0:size]:
            os.system('ln -sf ' + src_folder + '/' + ds + '/' + seq + '/' + pose + ' ' 
                    + target_folder + variant + '/poses/' + seq + '-' + pose)

        # create calibration files
        for i in range(len(images)):
            with open(target_folder+variant + '/calibration/{0:s}-frame-{1:s}.calibration.txt'
                      .format(seq, str(i).zfill(6)), 'w') as f:
                f.write(str(focallength))


if __name__ == "__main__":

    # name of the folder where we download the original 7scenes dataset to
    # we restructure the dataset by creating symbolic links to that folder
    src_folder = os.environ['HOME'] + '/7scenes/raw_data'
    focallength = 525.0

    # download the original 7 scenes dataset for poses and images
    mkdir(src_folder)
    os.chdir(src_folder)

    # for ds in ['chess', 'fire', 'heads', 'office', 'pumpkin', 'redkitchen', 'stairs']:
    for ds in ['chess']:
        download_data(ds)
        print("Linking files...")
        target_folder = '../7scenes_' + ds + '/'
        link_frames(target_folder, 'TrainSplit.txt', 'train', size=10)
        link_frames(target_folder, 'TestSplit.txt', 'test', size=10)
