import torch
from torch.utils.data import Dataset

import os.path as osp
import json
import cv2
from skimage import io
from PIL import Image
import numpy as np
import random
from torch.utils.data.dataloader import default_collate
from torch.utils.data.dataloader import DataLoader
import matplotlib.pyplot as plt
from torchvision.transforms import functional as F
import copy



def add_shade(img, random_state=None, nb_ellipses=20,
              amplitude=[-0.5, 0.5], kernel_size_interval=(250, 350)):
    """ Overlay the image with several shades
    Parameters:
      nb_ellipses: number of shades
      amplitude: tuple containing the illumination bound (between -1 and 0) and the
        shawdow bound (between 0 and 1)
      kernel_size_interval: interval of the kernel used to blur the shades
    """
    if random_state is None:
        random_state = np.random.RandomState(None)
    transparency = random_state.uniform(*amplitude)

    min_dim = min(img.shape) / 4
    mask = np.zeros(img.shape[:2], np.uint8)
    for i in range(nb_ellipses):
        ax = int(max(random_state.rand() * min_dim, min_dim / 5))
        ay = int(max(random_state.rand() * min_dim, min_dim / 5))
        max_rad = max(ax, ay)
        x = random_state.randint(max_rad, img.shape[1] - max_rad)  # center
        y = random_state.randint(max_rad, img.shape[0] - max_rad)
        angle = random_state.rand() * 90
        cv2.ellipse(mask, (x, y), (ax, ay), angle, 0, 360, 255, -1)

    kernel_size = int(kernel_size_interval[0] + random_state.rand() *
                      (kernel_size_interval[1] - kernel_size_interval[0]))
    if (kernel_size % 2) == 0:  # kernel_size has to be odd
        kernel_size += 1
    mask = cv2.GaussianBlur(mask.astype(np.float64), (kernel_size, kernel_size), 0)
    shaded = img * (1 - transparency * mask/255.)
    shaded = np.clip(shaded, 0, 255)
    return shaded.astype(np.uint8)


def add_fog(img, random_state=None, max_nb_ellipses=20,
            transparency=0.6, kernel_size_interval=(150, 250)):
    """ Overlay the image with several shades
    Parameters:
      max_nb_ellipses: number max of shades
      transparency: level of transparency of the shades (1 = no shade)
      kernel_size_interval: interval of the kernel used to blur the shades
    """
    if random_state is None:
        random_state = np.random.RandomState(None)

    centers = np.empty((0, 2), dtype=np.int32)
    rads = np.empty((0, 1), dtype=np.int32)
    min_dim = min(img.shape) / 4
    shaded_img = img.copy()
    for i in range(max_nb_ellipses):
        ax = int(max(random_state.rand() * min_dim, min_dim / 5))
        ay = int(max(random_state.rand() * min_dim, min_dim / 5))
        max_rad = max(ax, ay)
        x = random_state.randint(max_rad, img.shape[1] - max_rad)  # center
        y = random_state.randint(max_rad, img.shape[0] - max_rad)
        new_center = np.array([[x, y]])

        # Check that the ellipsis will not overlap with pre-existing shapes
        diff = centers - new_center
        if np.any(max_rad > (np.sqrt(np.sum(diff * diff, axis=1)) - rads)):
            continue
        centers = np.concatenate([centers, new_center], axis=0)
        rads = np.concatenate([rads, np.array([[max_rad]])], axis=0)

        col = random_state.randint(256)  # color of the shade
        angle = random_state.rand() * 90
        cv2.ellipse(shaded_img, (x, y), (ax, ay), angle, 0, 360, col, -1)
    shaded_img = shaded_img.astype(float)
    kernel_size = int(kernel_size_interval[0] + random_state.rand() *
                      (kernel_size_interval[1] - kernel_size_interval[0]))
    if (kernel_size % 2) == 0:  # kernel_size has to be odd
        kernel_size += 1

    cv2.GaussianBlur(shaded_img, (kernel_size, kernel_size), 0, shaded_img)
    mask = np.where(shaded_img != img)
    shaded_img[mask] = (1 - transparency) * shaded_img[mask] + transparency * img[mask]
    shaded_img = np.clip(shaded_img, 0, 255)
    return shaded_img.astype(np.uint8)


# def motion_blur(img, max_ksize=5):
def motion_blur(img, max_ksize=8):
    # Either vertial, hozirontal or diagonal blur
    mode = np.random.choice(['h', 'v', 'diag_down', 'diag_up'])
    ksize = np.random.randint(0, (max_ksize+1)/2)*2 + 1  # make sure is odd

    center = int((ksize-1)/2)
    kernel = np.zeros((ksize, ksize))
    if mode == 'h':
        kernel[center, :] = 1.
    elif mode == 'v':
        kernel[:, center] = 1.
    elif mode == 'diag_down':
        kernel = np.eye(ksize)
    elif mode == 'diag_up':
        kernel = np.flip(np.eye(ksize), 0)
    var = ksize * ksize / 16.
    grid = np.repeat(np.arange(ksize)[:, np.newaxis], ksize, axis=-1)
    gaussian = np.exp(-(np.square(grid-center)+np.square(grid.T-center))/(2.*var))
    kernel *= gaussian
    kernel /= np.sum(kernel)
    img = cv2.filter2D(img.astype(np.uint8), -1, kernel)
    return img

# def additive_gaussian_noise(img, random_state=None, std=(0, 10)):
def additive_gaussian_noise(img, random_state=None, std=(0, 15)):
    """ Add gaussian noise to the current image pixel-wise
    Parameters:
      std: the standard deviation of the filter will be between std[0] and std[0]+std[1]
    """
    if random_state is None:
        random_state = np.random.RandomState(None)
    sigma = std[0] + random_state.rand() * std[1]
    gaussian_noise = random_state.randn(*img.shape) * sigma
    noisy_img = img + gaussian_noise
    noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)
    return noisy_img


# def additive_speckle_noise(img, intensity=1):
def additive_speckle_noise(img, intensity=2):
    """ Add salt and pepper noise to an image
    Parameters:
      intensity: the higher, the more speckles there will be
    """
    noise = np.zeros(img.shape, dtype=np.uint8)
    cv2.randu(noise, 0, 256)
    black = noise < intensity
    white = noise > 255 - intensity
    noisy_img = img.copy()
    noisy_img[white > 0] = 255
    noisy_img[black > 0] = 0
    return noisy_img


# def random_brightness(img, random_state=None, max_change=50):
def random_brightness(img, random_state=None, max_change=80):
    """ Change the brightness of img
    Parameters:
      max_change: max amount of brightness added/subtracted to the image
    """
    if random_state is None:
        random_state = np.random.RandomState(None)
    brightness = random_state.randint(-max_change, max_change)
    new_img = img.astype(np.int16) + brightness
    return np.clip(new_img, 0, 255).astype(np.uint8)


# def random_contrast(img, random_state=None, max_change=[0.5, 1.5]):
def random_contrast(img, random_state=None, max_change=[0.5, 2.0]):
    """ Change the contrast of img
    Parameters:
      max_change: the change in contrast will be between 1-max_change and 1+max_change
    """
    if random_state is None:
        random_state = np.random.RandomState(None)
    contrast = random_state.uniform(*max_change)
    mean = np.mean(img, axis=(0, 1))
    new_img = np.clip(mean + (img - mean) * contrast, 0, 255)
    return new_img.astype(np.uint8)



class TrainDataset(Dataset):
    def __init__(self, root, ann_file, transform = None, augmentation = 4):
        self.root = root
        with open(ann_file,'r') as _:
            self.annotations = json.load(_)
        self.transform = transform
        self.augmentation = augmentation
    
    def __getitem__(self, idx_):
        # print(idx_)

        idx = idx_%len(self.annotations)
        # random_prob = torch.rand(1)
        # reminder = torch.randint(0,4,(1,)).item()
        reminder = idx_//len(self.annotations)
        
        # idx = 0
        # reminder = 0
        ann = copy.deepcopy(self.annotations[idx])
        if len(ann['edges_negative']) == 0:
            ann['edges_negative'] = [[0,0]]
        ann['reminder'] = reminder
        # image = cv2.imread(osp.join(self.root,ann['filename']), cv2.IMREAD_GRAYSCALE).astype(float)

        image = cv2.imread(osp.join(self.root,ann['filename']), cv2.IMREAD_GRAYSCALE)
        data_aug_random = random.randint(1, 10)
        if data_aug_random == 1:
            image = add_shade(image)
        elif data_aug_random == 2:
            image = add_fog(image)
        elif data_aug_random == 3:
            image = motion_blur(image)
        elif data_aug_random == 4:
            image = additive_gaussian_noise(image)
        elif data_aug_random == 5:
            image = additive_speckle_noise(image)
        elif data_aug_random == 6:
            image = random_brightness(image)
        elif data_aug_random == 7:
            image = random_contrast(image)
        image = image.astype(float)

        if len(image.shape) == 2:
            image = np.concatenate([image[...,None],image[...,None],image[...,None]],axis=-1)
        else:
            image = image[:,:,:3]

        # if len(ann['junctions']) == 0:
        #     ann['junctions'] = [[0,0]]
        #     ann['edges_positive'] = [[0,0]]
        
        # image = Image.open(osp.join(self.root,ann['filename'])).convert('RGB')
        for key,_type in (['junctions',np.float32],
                          ['edges_positive',np.int32],
                          ['edges_negative',np.int32]):
            ann[key] = np.array(ann[key],dtype=_type)
        
        width = ann['width']
        height = ann['height']
        if reminder == 1:
            image = image[:,::-1,:]
            # image = F.hflip(image)
            ann['junctions'][:,0] = width-ann['junctions'][:,0]
        elif reminder == 2:
            # image = F.vflip(image)
            image = image[::-1,:,:]
            ann['junctions'][:,1] = height-ann['junctions'][:,1]
        elif reminder == 3:
            # image = F.vflip(F.hflip(image))
            image = image[::-1,::-1,:]
            ann['junctions'][:,0] = width-ann['junctions'][:,0]
            ann['junctions'][:,1] = height-ann['junctions'][:,1]
        elif reminder == 4:
            image_rotated = np.rot90(image)

            junctions = ann['junctions'] - np.array([image.shape[1],image.shape[0]]).reshape(1,-1)/2.0
            theta = 0.5*np.pi
            rot_mat = np.array([[np.cos(theta),np.sin(theta)],[-np.sin(theta),np.cos(theta)]])
            junctions_rotated = (rot_mat@junctions.transpose()).transpose()
            junctions_rotated = junctions_rotated + np.array([image_rotated.shape[1],image_rotated.shape[0]]).reshape(1,-1)/2.0

            ann['width'] = image_rotated.shape[1]
            ann['height'] = image_rotated.shape[0]
            ann['junctions'] = np.asarray(junctions_rotated,dtype=np.float32)
            image = image_rotated
        elif reminder == 5:
            image_rotated = np.rot90(np.rot90(np.rot90(image)))

            junctions = ann['junctions'] - np.array([image.shape[1],image.shape[0]]).reshape(1,-1)/2.0
            theta = 1.5*np.pi
            rot_mat = np.array([[np.cos(theta),np.sin(theta)],[-np.sin(theta),np.cos(theta)]])
            junctions_rotated = (rot_mat@junctions.transpose()).transpose()
            junctions_rotated = junctions_rotated + np.array([image_rotated.shape[1],image_rotated.shape[0]]).reshape(1,-1)/2.0

            ann['width'] = image_rotated.shape[1]
            ann['height'] = image_rotated.shape[0]
            ann['junctions'] = np.asarray(junctions_rotated,dtype=np.float32)
            image = image_rotated
        # elif reminder == 6:


        if self.transform is not None:
            return self.transform(image,ann)
        
        
        return image, ann

    def __len__(self):
        return len(self.annotations)*self.augmentation

def collate_fn(batch):
    return (default_collate([b[0] for b in batch]),
            [b[1] for b in batch])