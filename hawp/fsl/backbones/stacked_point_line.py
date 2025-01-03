"""
Hourglass network inserted in the pre-activated Resnet
Use lr=0.01 for current version
(c) Nan Xue (HAWP)
(c) Yichao Zhou (LCNN)
(c) YANG, Wei
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

__all__ = ["StackPointLine", "hg"]


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

        # path = Path(__file__).parent / 'weights/superpoint_v1.pth'
        path = "/media/code/ubuntu_files/public_projects/SuperGluePretrainedNetwork/models/weights/superpoint_v1.pth"
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
        features.append(x)                  # [B, 128, H/8, W/8]

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



class Bottleneck2D(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck2D, self).__init__()

        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 2, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        return out


class Hourglass(nn.Module):
    def __init__(self, block, num_blocks, planes, depth):
        super(Hourglass, self).__init__()
        self.depth = depth
        self.block = block
        self.hg = self._make_hour_glass(block, num_blocks, planes, depth)

    def _make_residual(self, block, num_blocks, planes):
        layers = []
        for i in range(0, num_blocks):
            layers.append(block(planes * block.expansion, planes))
        return nn.Sequential(*layers)

    def _make_hour_glass(self, block, num_blocks, planes, depth):
        hg = []
        for i in range(depth):
            res = []
            for j in range(3):
                res.append(self._make_residual(block, num_blocks, planes))
            if i == 0:
                res.append(self._make_residual(block, num_blocks, planes))
            hg.append(nn.ModuleList(res))
        return nn.ModuleList(hg)

    def _hour_glass_forward(self, n, x):
        up1 = self.hg[n - 1][0](x)
        low1 = F.max_pool2d(x, 2, stride=2)
        low1 = self.hg[n - 1][1](low1)

        if n > 1:
            low2 = self._hour_glass_forward(n - 1, low1)
        else:
            low2 = self.hg[n - 1][3](low1)
        low3 = self.hg[n - 1][2](low2)
        up2 = F.interpolate(low3, scale_factor=2)
        out = up1 + up2
        return out

    def forward(self, x):
        return self._hour_glass_forward(self.depth, x)


class StackPointLine(nn.Module):
    """Hourglass model from Newell et al ECCV 2016"""

    def __init__(self, input_channels, inplanes, num_feats, block, head, depth, num_stacks, num_blocks, num_classes):
        '''
            input_channels : 3 or 1
            inplanes       : 64    
            num_feats      : 256//2 = 128
            block          : Bottleneck2D
            head           : MultitaskHead(c_in, c_out, head_size=head_size)  head_size : [[3], [1], [1], [2], [2]]
            depth          : 4
            num_stacks     : 2
            num_blocks     : 1
            num_classes    : sum(sum(head_size, [])) = 9
        '''

        super(StackPointLine, self).__init__()

        self.inplanes = inplanes
        self.num_feats = num_feats
        self.num_stacks = num_stacks
        self.conv1 = nn.Conv2d(input_channels, self.inplanes, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_residual(block, self.inplanes, 1)
        self.layer2 = self._make_residual(block, self.inplanes, 1)
        self.layer3 = self._make_residual(block, self.num_feats, 1)
        self.maxpool = nn.MaxPool2d(2, stride=2)

        # build hourglass modules
        ch = self.num_feats * block.expansion
        # vpts = []
        hg, res, fc, score, fc_, score_ = [], [], [], [], [], []
        for i in range(num_stacks):
            hg.append(Hourglass(block, num_blocks, self.num_feats, depth))
            res.append(self._make_residual(block, self.num_feats, num_blocks))
            fc.append(self._make_fc(ch, ch))
            score.append(head(ch, num_classes))
            # vpts.append(VptsHead(ch))
            # vpts.append(nn.Linear(ch, 9))
            # score.append(nn.Conv2d(ch, num_classes, kernel_size=1))
            # score[i].bias.data[0] += 4.6
            # score[i].bias.data[2] += 4.6
            if i < num_stacks - 1:
                fc_.append(nn.Conv2d(ch, ch, kernel_size=1))
                score_.append(nn.Conv2d(num_classes, ch, kernel_size=1))
        self.hg = nn.ModuleList(hg)
        self.res = nn.ModuleList(res)
        self.fc = nn.ModuleList(fc)
        self.score = nn.ModuleList(score)
        # self.vpts = nn.ModuleList(vpts)
        self.fc_ = nn.ModuleList(fc_)
        self.score_ = nn.ModuleList(score_)

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
        self.conv2a = nn.Conv2d(96, 64, kernel_size=3, stride=1, padding=1)
        self.bn2a = nn.BatchNorm2d(64)
        self.conv2b = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn2b = nn.BatchNorm2d(64)

        # [H/4, W/4]
        self.conv3a = nn.Conv2d(192, 128, kernel_size=3, stride=1, padding=1)
        self.bn3a = nn.BatchNorm2d(128)
        self.conv3b = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn3b = nn.BatchNorm2d(256)


    def _make_residual(self, block, planes, blocks, stride=1):
        '''
            block          : Bottleneck2D
            planes         : layer1 and layer2 : 64, layer3 and res : 128
            blocks         : 1

            self.inplanes  : layer1 : 64
            block.expansion: 2
        '''
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                )
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _make_fc(self, inplanes, outplanes):
        bn = nn.BatchNorm2d(inplanes)
        conv = nn.Conv2d(inplanes, outplanes, kernel_size=1)
        return nn.Sequential(conv, bn, self.relu)

    def forward(self, x):
        out = []
        # # print("x0.shape = {}".format(x.shape))
        # x = self.conv1(x)
        # x = self.bn1(x)
        # x = self.relu(x)
        # # print("x1.shape = {}".format(x.shape))

        # x = self.layer1(x)
        # # print("x2.shape = {}".format(x.shape))
        # x = self.maxpool(x)
        # # print("x3.shape = {}".format(x.shape))
        # x = self.layer2(x)
        # # print("x4.shape = {}".format(x.shape))
        # x = self.layer3(x)
        # # print("x5.shape = {}".format(x.shape))

        points = self.point_detector(x[:, :1, ...])
        features = points['features']

        x1 = self.relu(self.bn1a(self.conv1a(features[0])))
        x1 = self.relu(self.bn1b(self.conv1b(x1)))          # [B, 32, H, W]

        x2 = self.pool(x1)                                  # [B, 32, H/2, W/2]

        x2 = torch.cat([x2, features[1]], -3)               # [B, 96, H/2, W/2]
        x2 = self.relu(self.bn2a(self.conv2a(x2)))
        x2 = self.relu(self.bn2b(self.conv2b(x2)))          # [B, 64, H/2, W/2]

        x3 = self.pool(x2)                                  # [B, 128, H/4, W/4]

        x3 = torch.cat([x3, features[2]], -3)               # [B, 192, H/4, W/4]
        x3 = self.relu(self.bn3a(self.conv3a(x3)))          # [B, 128, H/4, W/4]
        x3 = self.relu(self.bn3b(self.conv3b(x3)))          # [B, 256, H/4, W/4]
        x = x3

        for i in range(self.num_stacks):
            # print("i = {}".format(i))
            # print("x6.shape = {}".format(x.shape))
            y = self.hg[i](x)
            # print("y1.shape = {}".format(y.shape))
            y = self.res[i](y)
            # print("y2.shape = {}".format(y.shape))
            y = self.fc[i](y)
            # print("y3.shape = {}".format(y.shape))
            score = self.score[i](y)
            # print("score.shape = {}".format(score.shape))
            out.append(score)
            # print("out.shape = {}".format(len(out)))

            if i < self.num_stacks - 1:
                fc_ = self.fc_[i](y)
                # print("fc_.shape = {}".format(fc_.shape))
                score_ = self.score_[i](score)
                # print("score_.shape = {}".format(score_.shape))
                x = x + fc_ + score_
                # print("x_num_stacks.shape = {}".format(x.shape))

        return out[::-1], y 
