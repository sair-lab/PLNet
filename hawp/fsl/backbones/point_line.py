import torch
import torch.nn as nn
import torch.nn.functional as F
import time

__all__ = ["PointLineNet", "hg"]

from pathlib import Path
import torch
from torch import nn

def simple_nms(scores, nms_radius: int):
    """ Fast Non-maximum suppression to remove nearby points """
    assert(nms_radius >= 0)

    def max_pool(x):
        return torch.nn.functional.max_pool2d(
            x, kernel_size=nms_radius*2+1, stride=1, padding=nms_radius)

    zeros = torch.zeros_like(scores)
    max_mask = scores == max_pool(scores)
    for _ in range(2):
        supp_mask = max_pool(max_mask.float()) > 0
        supp_scores = torch.where(supp_mask, zeros, scores)
        new_max_mask = supp_scores == max_pool(supp_scores)
        max_mask = max_mask | (new_max_mask & (~supp_mask))
    return torch.where(max_mask, scores, zeros)


def remove_borders(keypoints, scores, border: int, height: int, width: int):
    """ Removes keypoints too close to the border """
    mask_h = (keypoints[:, 0] >= border) & (keypoints[:, 0] < (height - border))
    mask_w = (keypoints[:, 1] >= border) & (keypoints[:, 1] < (width - border))
    mask = mask_h & mask_w
    return keypoints[mask], scores[mask]


def top_k_keypoints(keypoints, scores, k: int):
    if k >= len(keypoints):
        return keypoints, scores
    scores, indices = torch.topk(scores, k, dim=0)
    return keypoints[indices], scores


def sample_descriptors(keypoints, descriptors, s: int = 8):
    """ Interpolate descriptors at keypoint locations """
    b, c, h, w = descriptors.shape
    keypoints = keypoints - s / 2 + 0.5
    keypoints /= torch.tensor([(w*s - s/2 - 0.5), (h*s - s/2 - 0.5)],
                              ).to(keypoints)[None]
    keypoints = keypoints*2 - 1  # normalize to (-1, 1)
    args = {'align_corners': True} if torch.__version__ >= '1.3' else {}
    descriptors = torch.nn.functional.grid_sample(
        descriptors, keypoints.view(b, 1, -1, 2), mode='bilinear', **args)
    descriptors = torch.nn.functional.normalize(
        descriptors.reshape(b, c, -1), p=2, dim=1)
    return descriptors


class SuperPoint(nn.Module):
    """SuperPoint Convolutional Detector and Descriptor

    SuperPoint: Self-Supervised Interest Point Detection and
    Description. Daniel DeTone, Tomasz Malisiewicz, and Andrew
    Rabinovich. In CVPRW, 2019. https://arxiv.org/abs/1712.07629

    """
    default_config = {
        'descriptor_dim': 256,
        'nms_radius': 4,
        'keypoint_threshold': 0.005,
        'max_keypoints': -1,
        'remove_borders': 4,
    }

    def __init__(self, config):
        super().__init__()
        self.config = {**self.default_config, **config}

        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        c1, c2, c3, c4, c5 = 64, 64, 128, 128, 256

        self.conv1a = nn.Conv2d(1, c1, kernel_size=3, stride=1, padding=1)
        self.conv1b = nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1)
        self.conv2a = nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1)
        self.conv2b = nn.Conv2d(c2, c2, kernel_size=3, stride=1, padding=1)
        self.conv3a = nn.Conv2d(c2, c3, kernel_size=3, stride=1, padding=1)
        self.conv3b = nn.Conv2d(c3, c3, kernel_size=3, stride=1, padding=1)
        self.conv4a = nn.Conv2d(c3, c4, kernel_size=3, stride=1, padding=1)
        self.conv4b = nn.Conv2d(c4, c4, kernel_size=3, stride=1, padding=1)

        self.convPa = nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convPb = nn.Conv2d(c5, 65, kernel_size=1, stride=1, padding=0)

        self.convDa = nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convDb = nn.Conv2d(
            c5, self.config['descriptor_dim'],
            kernel_size=1, stride=1, padding=0)

        path = Path(__file__).parent.parent / 'point_model/point_model.pth'
        self.load_state_dict(torch.load(str(path)))

        mk = self.config['max_keypoints']
        if mk == 0 or mk < -1:
            raise ValueError('\"max_keypoints\" must be positive or \"-1\"')

        print('Loaded SuperPoint model')

    def forward(self, data):
        """ Compute keypoints, scores, descriptors for image """
        # Shared Encoder
        features = []
        x = self.relu(self.conv1a(data))
        x = self.relu(self.conv1b(x))
        features.append(x)                  # [B, 64, H, W]
        x = self.pool(x)
        x = self.relu(self.conv2a(x))
        x = self.relu(self.conv2b(x))
        features.append(x)                  # [B, 64, H/2, W/2]
        x = self.pool(x)
        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))
        features.append(x)                  # [B, 128, H/4, W/4]

        x = self.pool(x)
        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))

        # Compute the dense keypoint scores
        cPa = self.relu(self.convPa(x))
        scores = self.convPb(cPa)
        scores = torch.nn.functional.softmax(scores, 1)[:, :-1]
        b, _, h, w = scores.shape
        scores = scores.permute(0, 2, 3, 1).reshape(b, h, w, 8, 8)
        scores = scores.permute(0, 1, 3, 2, 4).reshape(b, h*8, w*8)
        scores = simple_nms(scores, self.config['nms_radius'])

        # Extract keypoints
        keypoints = [
            torch.nonzero(s > self.config['keypoint_threshold'])
            for s in scores]
        scores = [s[tuple(k.t())] for s, k in zip(scores, keypoints)]

        # Discard keypoints near the image borders
        keypoints, scores = list(zip(*[
            remove_borders(k, s, self.config['remove_borders'], h*8, w*8)
            for k, s in zip(keypoints, scores)]))

        # Keep the k keypoints with highest score
        if self.config['max_keypoints'] >= 0:
            keypoints, scores = list(zip(*[
                top_k_keypoints(k, s, self.config['max_keypoints'])
                for k, s in zip(keypoints, scores)]))

        # Convert (h, w) to (x, y)
        keypoints = [torch.flip(k, [1]).float() for k in keypoints]

        # Compute the dense descriptors
        cDa = self.relu(self.convDa(x))
        descriptors = self.convDb(cDa)
        descriptors = torch.nn.functional.normalize(descriptors, p=2, dim=1)

        # Extract descriptors
        descriptors = [sample_descriptors(k[None], d[None], 8)[0]
                       for k, d in zip(keypoints, descriptors)]

        return {
            'features': features,
            'keypoints': keypoints,
            'scores': scores,
            'descriptors': descriptors,
        }


class UNet(nn.Module):

    def __init__(self, input_channel, conv_channel, output_channel, layer_num):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        D0, D1 = conv_channel, int(conv_channel/2)
        D2 = D0 + D1

        # [H/4, W/4]
        self.conv1a = nn.Conv2d(input_channel, D0, kernel_size=3, stride=1, padding=1)
        self.bn1a = nn.BatchNorm2d(D0)
        self.conv1b = nn.Conv2d(D0, D0, kernel_size=3, stride=1, padding=1)
        self.bn1b = nn.BatchNorm2d(D0)

        # [H/8, W/8]
        self.conv2a = nn.Conv2d(D0, D0, kernel_size=3, stride=1, padding=1)
        self.bn2a = nn.BatchNorm2d(D0)
        self.conv2b = nn.Conv2d(D0, D0, kernel_size=3, stride=1, padding=1)
        self.bn2b = nn.BatchNorm2d(D0)

        # [H/16, W/16]
        self.conv3a = nn.Conv2d(D0, D0, kernel_size=3, stride=1, padding=1)
        self.bn3a = nn.BatchNorm2d(D0)
        self.conv3b = nn.Conv2d(D0, D0, kernel_size=3, stride=1, padding=1)
        self.bn3b = nn.BatchNorm2d(D0)

        # [H/32, W/32]
        self.conv4a = nn.Conv2d(D0, D0, kernel_size=3, stride=1, padding=1)
        self.bn4a = nn.BatchNorm2d(D0)
        self.conv4b = nn.Conv2d(D0, D0, kernel_size=3, stride=1, padding=1)
        self.bn4b = nn.BatchNorm2d(D0)

        # [H/64, W/64]
        self.conv5a = nn.Conv2d(D0, D0, kernel_size=3, stride=1, padding=1)
        self.bn5a = nn.BatchNorm2d(D0)
        self.conv5b = nn.Conv2d(D0, D0, kernel_size=3, stride=1, padding=1)
        self.bn5b = nn.BatchNorm2d(D0)

        # [H/64, W/64]
        # self.deconv1 = nn.ConvTranspose2d(D0, D1, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.deconv1 = nn.Conv2d(D0, D1, kernel_size=3, stride=1, padding=1)
        self.bn1_dec = nn.BatchNorm2d(D1)

        # [H/32, W/32]
        self.conv4a_up = nn.Conv2d(D0, D1, kernel_size=3, stride=1, padding=1)
        self.bn4a_up   = nn.BatchNorm2d(D1)
        self.conv4b_up = nn.Conv2d(D0, D0, kernel_size=3, stride=1, padding=1)
        self.bn4b_up   = nn.BatchNorm2d(D0)

        # [H/32, W/32]
        # self.deconv2 = nn.ConvTranspose2d(D0, D1, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.deconv2 = nn.Conv2d(D0, D1, kernel_size=3, stride=1, padding=1)
        self.bn2_dec = nn.BatchNorm2d(D1)

        # [H/16, W/16]
        self.conv3a_up = nn.Conv2d(D0, D1, kernel_size=3, stride=1, padding=1)
        self.bn3a_up   = nn.BatchNorm2d(D1)
        self.conv3b_up = nn.Conv2d(D0, D0, kernel_size=3, stride=1, padding=1)
        self.bn3b_up   = nn.BatchNorm2d(D0)

        # [H/16, W/16]
        # self.deconv3 = nn.ConvTranspose2d(D0, D1, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.deconv3 = nn.Conv2d(D0, D1, kernel_size=3, stride=1, padding=1)
        self.bn3_dec = nn.BatchNorm2d(D1)

        # [H/8, W/8]
        self.conv2a_up = nn.Conv2d(D0, D1, kernel_size=3, stride=1, padding=1)
        self.bn2a_up   = nn.BatchNorm2d(D1)
        self.conv2b_up = nn.Conv2d(D0, D0, kernel_size=3, stride=1, padding=1)
        self.bn2b_up   = nn.BatchNorm2d(D0)

        # [H/8, W/8]
        # self.deconv4 = nn.ConvTranspose2d(D0, D1, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.deconv4 = nn.Conv2d(D0, D1, kernel_size=3, stride=1, padding=1)
        self.bn4_dec = nn.BatchNorm2d(D1)

        # [H/4, W/4]
        self.conv1a_up = nn.Conv2d(D0, D1, kernel_size=3, stride=1, padding=1)
        self.bn1a_up   = nn.BatchNorm2d(D1)
        self.conv1b_up = nn.Conv2d(D0, output_channel, kernel_size=3, stride=1, padding=1)
        self.bn1b_up   = nn.BatchNorm2d(output_channel)


    def forward(self, x):
        x1 = self.relu(self.bn1a(self.conv1a(x)))          # [B, D0, H/4, W/4]
        x1 = self.relu(self.bn1b(self.conv1b(x1)))          # [B, D0, H/4, W/4]

        x2 = self.pool(x1)                                 # [B, D0, H/8, W/8]

        x2 = self.relu(self.bn2a(self.conv2a(x2)))          # [B, D0, H/8, W/8]
        x2 = self.relu(self.bn2b(self.conv2b(x2)))          # [B, D0, H/8, W/8]

        x3 = self.pool(x2)                                 # [B, D0, H/16, W/16]

        x3 = self.relu(self.bn3a(self.conv3a(x3)))          # [B, D0, H/16, W/16]
        x3 = self.relu(self.bn3b(self.conv3b(x3)))          # [B, D0, H/16, W/16]

        x4 = self.pool(x3)                                 # [B, D0, H/32, W/32]

        x4 = self.relu(self.bn4a(self.conv4a(x4)))          # [B, D0, H/32, W/32]
        x4 = self.relu(self.bn4b(self.conv4b(x4)))          # [B, D0, H/32, W/32]

        x5 = self.pool(x4)                                 # [B, D0, H/64, W/64]

        x5 = self.relu(self.bn5a(self.conv5a(x5)))          # [B, D0, H/64, W/64]
        x5 = self.relu(self.bn5b(self.conv5b(x5)))          # [B, D0, H/64, W/64]

        x = F.interpolate(x5, scale_factor=2)
        x = self.relu(self.bn1_dec(self.deconv1(x)))           # [B, D1, H/32, W/32]

        x4_up = self.relu(self.bn4a_up(self.conv4a_up(x4)))     # [B, D1, H/32, W/32]
        x = torch.cat([x, x4_up], -3)                           # [B, D0, H/32, W/32]
        x = self.relu(self.bn4b_up(self.conv4b_up(x)))          # [B, D0, H/32, W/32]

        x = F.interpolate(x, scale_factor=2)
        x = self.relu(self.bn2_dec(self.deconv2(x)))            # [B, D1, H/16, W/16]

        x3_up = self.relu(self.bn3a_up(self.conv3a_up(x3)))     # [B, D1, H/16, W/16]
        x = torch.cat([x, x3_up], -3)                           # [B, D0, H/16, W/16]
        x = self.relu(self.bn3b_up(self.conv3b_up(x)))          # [B, D0, H/16, W/16]

        x = F.interpolate(x, scale_factor=2)
        x = self.relu(self.bn3_dec(self.deconv3(x)))            # [B, D1, H/8, W/8]

        x2_up = self.relu(self.bn2a_up(self.conv2a_up(x2)))     # [B, D1, H/8, W/8]
        x = torch.cat([x, x2_up], -3)                           # [B, D0, H/8, W/8]
        x = self.relu(self.bn2b_up(self.conv2b_up(x)))          # [B, D0, H/8, W/8]

        x = F.interpolate(x, scale_factor=2)
        x = self.relu(self.bn4_dec(self.deconv4(x)))            # [B, D1, H/4, W/4]

        x1_up = self.relu(self.bn1a_up(self.conv1a_up(x1)))     # [B, D1, H/4, W/4]
        x = torch.cat([x, x1_up], -3)                           # [B, D0, H/4, W/4]
        x = self.relu(self.bn1b_up(self.conv1b_up(x)))          # [B, output_channel, H/4, W/4]

        return x


class PointLineNet(nn.Module):

    def __init__(self, head):
        super().__init__()
        self.point_detector = SuperPoint({})
        for param in self.point_detector.parameters():
            param.requires_grad = False

        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # [H, W]
        self.conv1a = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.bn1a = nn.BatchNorm2d(32)
        self.conv1b = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.bn1b = nn.BatchNorm2d(32)

        # [H/2, W/2]
        self.conv2a = nn.Conv2d(96, 128, kernel_size=3, stride=1, padding=1)
        self.bn2a = nn.BatchNorm2d(128)
        self.conv2b = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn2b = nn.BatchNorm2d(128)

        self.stack1 = UNet(256, 128, 128, 4)
        self.fc1 = nn.Conv2d(128, 256, kernel_size=1)
        self.score1 = head(256, 9)
        self.stack2 = UNet(128, 128, 128, 4)
        self.fc2 = nn.Conv2d(128, 256, kernel_size=1)
        self.score2 = head(256, 9)


    def forward(self, image):
        torch.cuda.synchronize()
        t0 = time.time()

        points = self.point_detector(image[:, :1, ...])
        torch.cuda.synchronize()
        t1 = time.time()

        features = points['features']

        x1 = self.relu(self.bn1a(self.conv1a(features[0])))
        x1 = self.relu(self.bn1b(self.conv1b(x1)))          # [B, 64, H, W]

        x2 = self.pool(x1)                                  # [B, 64, H/2, W/2]

        x2 = torch.cat([x2, features[1]], -3)               # [B, 128, H/2, W/2]
        x2 = self.relu(self.bn2a(self.conv2a(x2)))
        x2 = self.relu(self.bn2b(self.conv2b(x2)))          # [B, 128, H/2, W/2]

        x3 = self.pool(x2)                                  # [B, 128, H/4, W/4]
        x3 = torch.cat([x3, features[2]], -3)               # [B, 256, H/4, W/4]


        torch.cuda.synchronize()
        t2 = time.time()

        x_stack1 = self.stack1(x3)
        x_stack1_ = self.fc1(x_stack1)
        score1 = self.score1(x_stack1_)

        x_stack2 = self.stack2(x_stack1)
        x_stack2_ = self.fc2(x_stack2)
        score2 = self.score1(x_stack2_)


        torch.cuda.synchronize()
        t3 = time.time()

        dt1 = t1 - t0
        dt2 = t2 - t1
        dt3 = t3 - t2

        # print("dt1 = {}, dt2 = {}, dt3 = {}".format(dt1*1000, dt2*1000, dt3*1000))

        return [score2, score1], x_stack2_