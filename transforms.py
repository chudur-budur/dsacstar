import numpy as np
import cv2
import torch
from skimage import transform, img_as_ubyte
from PIL import Image

__all__ = ["rotate", "unfish", "cambridgify", "compute_pose"]

def rotate(img, angle, order, mode='constant'):
    # rotate input image
    if isinstance(img, torch.Tensor): # tensor to numpy
        img_ = img.permute(1, 2, 0).numpy()
    elif isinstance(img, Image.Image):
        img_ = np.array(img)
    elif isinstance(img, np.ndarray):
        img_ = img
    else:
        raise TypeError("Couldn't recognize `img` data type", type(img))

    img_ = transform.rotate(img_, angle, order=order, mode=mode)

    if isinstance(img, torch.Tensor): # tensor to numpy
        img_ = torch.from_numpy(img_).permute(2, 0, 1).float()
    elif isinstance(img, Image.Image):
        img_ = Image.fromarray(img_)
    elif isinstance(img, np.ndarray): 
        if img.dtype == np.uint8: 
            img_ = img_as_ubyte(img_)
    return img_


def unfish(image,
           camera_intrinsics=np.array(
        [626.61271202815, 625.32362717293194,
         1336.0594825127226, 949.7379173445073]),
        distortion_coeffs=np.array(
        [0.20744323318046468, -0.07788465215624041,
         0.00530138015440915, 0.0032944874130648585])):
    """Undistort an fisheye image.

    In Jellyfish data, the images are from a fisheye camera.
    So we need to to undistort and rescale the image for the
    neural net input.
    """
    fx, fy, cx, cy = camera_intrinsics[0], camera_intrinsics[1], \
        camera_intrinsics[2], camera_intrinsics[3]
    cmat = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])

    # undistort
    mapx, mapy = cv2.fisheye.\
            initUndistortRectifyMap(cmat, distortion_coeffs, np.eye(3), \
                cmat, (image.shape[1], image.shape[0]), cv2.CV_16SC2)
    
    image_ = cv2.remap(image, mapx, mapy, cv2.INTER_LINEAR)

    return image_


def cambridgify(image):
    target_height = 480  # rescale images
    # sub sampling of our CNN architecture,
    # for size of the initalization targets
    nn_subsampling = 8
    img_aspect = image.shape[0] / image.shape[1]
    if img_aspect > 1:
        # portrait
        img_w = target_height
        img_h = int(np.ceil(target_height * img_aspect))
    else:
        # landscape
        img_w = int(np.ceil(target_height / img_aspect))
        img_h = target_height

    out_w = int(np.ceil(img_w / nn_subsampling))
    out_h = int(np.ceil(img_h / nn_subsampling))
    out_scale = out_w / image.shape[1]
    img_scale = img_w / image.shape[1]
    image_ = cv2.resize(image, (img_w, img_h))
    return image


def cannify(image, low_threshold=0, ratio=3, kernel_size=3):
    # image = cannify(image, low_threshold=20, ratio=2, kernel_size=4)
    src_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.blur(src_gray, (3, 3))
    detected_edges = cv2.Canny(
        img_blur, low_threshold, low_threshold * ratio, kernel_size)
    mask = detected_edges != 0
    image = image * (mask[:, :, None].astype(image.dtype))
    return image


def compute_pose(p, q):
    # quaternion to axis-angle
    angle = 2 * np.arccos(q[3])
    x = q[0] / np.sqrt(1 - q[3]**2)
    y = q[1] / np.sqrt(1 - q[3]**2)
    z = q[2] / np.sqrt(1 - q[3]**2)

    R, _ = cv2.Rodrigues(np.array([x * angle, y * angle, z * angle]))
    T = -np.matmul(R, p.T)[:, np.newaxis]

    pose = None
    if np.absolute(T).max() > 10000:
        warnings.warn(
            "A matrix with extremely large translation. Outlier?")
        warnings.warn(T)
    else:
        pose = np.hstack((R, T))
        pose = np.vstack((pose, [[0, 0, 0, 1]]))
        pose = np.linalg.inv(pose)
    return pose
