import torch
from collections import OrderedDict
from torch.nn import utils, functional as F
from torch.optim import Adam
from torch.autograd import Variable
from torch.backends import cudnn
import scipy.misc as sm
import numpy as np
from networks.deeplab_resnet import resnet50_locate
from networks.vgg import vgg16_locate
import os
import torchvision.utils as vutils
import cv2
import math
import time
import mobula
from attention_sampler.attsampler_th import AttSampler
from KRN_edge import *
from KRN import KRN
mobula.op.load('attention_sampler')



class Solver(object):
    def __init__(self, train_loader, test_loader, config):
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.config = config
        self.iter_size = config.iter_size
        self.show_every = config.show_every
        self.lr_decay_epoch = [9]
        self.build_model()

        # self.build_model1()
        if config.mode == 'test':
            print('Loading pre-trained model from %s...' % self.config.model)
            if self.config.cuda:
                self.net.load_state_dict(torch.load(self.config.clm_model, map_location={'cuda:2':'cuda:0'}))
                self.net_hou.load_state_dict(torch.load(self.config.fsm_model, map_location={'cuda:2':'cuda:0'}))
            else:
                self.net.load_state_dict(torch.load(self.config.model, map_location='cpu'))
            self.net.eval()

    # print the network information and parameter numbers
    def print_network(self, model, name):
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(name)
        print(model)
        print("The number of parameters: {}".format(num_params))

    # build the network
    def build_model(self):  # 训练好的模型
        self.net = KRN(self.config.arch, *extra_layer(self.config.arch, resnet50_locate()))
        if self.config.cuda:
            self.net = self.net.cuda()
        self.net.eval()
        if self.config.mode == 'train':
            self.net.load_state_dict(
                torch.load(self.config.clm_model))

        self.net_hou = KRN_edge(self.config.arch, *extra_layer(self.config.arch, resnet50_locate()))
        if self.config.cuda:
            self.net_hou = self.net_hou.cuda()
        self.net_hou.eval() # use_global_stats = True
        if self.config.mode == 'train':
            self.net_hou.load_state_dict(torch.load(self.config.fsm_model))
        self.lr = self.config.lr
        self.wd = self.config.wd
        self.optimizer = Adam([{'params': filter(lambda p: p.requires_grad, self.net.parameters())},
                               {'params': filter(lambda p: p.requires_grad, self.net_hou.parameters())}], lr=self.lr,
                              weight_decay=self.wd)

    def test(self):
        mode_name = 'sal_fuse'
        time_s = time.time()
        img_num = len(self.test_loader)
        for i, data_batch in enumerate(self.test_loader):
            images, name, im_size = data_batch['image'], data_batch['name'][0], np.asarray(data_batch['size'])
            with torch.no_grad():
                images = Variable(images)
                if self.config.cuda:
                    images = images.cuda()
                feasum_out, merge_solid, out_merge_solid1, out_merge_solid2, out_merge_solid3, out_merge_solid4 = self.net(
                    images)


                map_s = feasum_out

                map_sx = torch.unsqueeze(torch.max(map_s, 3)[0], dim=3)  # ([1, 400, 1])
                map_sx = torch.squeeze(map_sx, dim=1)
                map_sy = torch.unsqueeze(torch.max(map_s, 2)[0], dim=3)  # ([1, 342, 1])
                map_sy = torch.squeeze(map_sy, dim=1)
                sum_sx = torch.sum(map_sx, dim=(1, 2), keepdim=True)
                sum_sy = torch.sum(map_sy, dim=(1, 2), keepdim=True)
                map_sx /= sum_sx
                map_sy /= sum_sy

                semi_pred, grid = AttSampler(scale=1, dense=2)(images, map_sx, map_sy)
                mapsssss, grid5 = AttSampler(scale=1, dense=2)(map_s, map_sx, map_sy)
                # mapsssss,grid5 = AttSampler(scale=1, dense=2)(map_s, map_sx, map_sy)
                data_pred, merge_solid, out_merge_solid1, out_merge_contour1, out_merge_solid2, out_merge_contour2, out_merge_solid3, out_merge_contour3, out_merge_solid4, out_merge_contour4 = self.net_hou(
                    semi_pred)

                ##################################restore##############################################
                x_index = grid[0, 1, :, 0]  # 400
                y_index = grid[0, :, 1, 1]  # 300

                new_data_size = tuple(data_pred.shape[1:4])
                new_data = torch.empty(new_data_size[0], new_data_size[1], new_data_size[2],
                                       device=images.device)  # 创建新的图
                new_data_final = torch.empty(new_data_size[0], new_data_size[1], new_data_size[2],
                                             device=images.device)  # 创建新的图
                x_index = (x_index + 1) * new_data_size[2] / 2
                y_index = (y_index + 1) * new_data_size[1] / 2

                xl = 0
                grid_l = x_index[0]
                data_l = data_pred[:, :, :, 0]
                for num in range(1, len(x_index)):
                    grid_r = x_index[num]
                    xr = torch.ceil(grid_r) - 1
                    xr = xr.int()
                    data_r = data_pred[:, :, :, num]
                    for h in range(xl + 1, xr + 1):
                        if h == grid_r:
                            new_data[:, :, h] = data_r
                        else:
                            new_data[:, :, h] = ((h - grid_l) * data_r / (grid_r - grid_l)) + (
                                        (grid_r - h) * data_l / (grid_r - grid_l))
                    xl = xr
                    grid_l = grid_r
                    data_l = data_r
                new_data[:, :, 0] = new_data[:, :, 1]
                try:
                    for h in range(xr + 1, len(x_index)):
                        new_data[:, :, h] = new_data[:, :, xr]
                except:
                    print('h', h)
                    print('xr', xr)

                yl = 0
                grid1_l = y_index[0]
                data1_l = new_data[:, 0, :]
                for num in range(1, len(y_index)):
                    grid1_r = y_index[num]
                    yr = torch.ceil(grid1_r) - 1
                    yr = yr.int()
                    data1_r = new_data[:, num, :]
                    for h in range(yl + 1, yr + 1):
                        if h == grid1_r:
                            new_data_final[:, h, :] = data1_r
                        else:
                            new_data_final[:, h, :] = ((h - grid1_l) * data1_r / (grid1_r - grid1_l)) + (
                                        (grid1_r - h) * data1_l / (grid1_r - grid1_l))
                    yl = yr
                    grid1_l = grid1_r
                    data1_l = data1_r
                new_data_final[:, 0, :] = new_data_final[:, 1, :]
                try:
                    for h in range(yr + 1, len(y_index)):
                        new_data_final[:, h, :] = new_data_final[:, yr, :]
                except:
                    print('h', h)
                    print('yr', yr)
                preds = torch.unsqueeze(new_data_final, dim=1)

                pred = np.squeeze(preds).cpu().data.numpy()
                multi_fuse = 255 * pred
                cv2.imwrite(os.path.join(self.config.test_fold, name[:-4] + '_' + mode_name + '.png'), multi_fuse)

        time_e = time.time()
        print('Speed: %f FPS' % (img_num / (time_e - time_s)))
        print('Test Done!')

    def train(self):
        iter_num = len(self.train_loader.dataset) // self.config.batch_size
        aveGrad = 0
        x_showEvery = 0
        for epoch in range(self.config.epoch):
            r_sal_loss = 0
            r_sal_loss1 = 0
            self.net_hou.zero_grad()
            for i, data_batch in enumerate(self.train_loader):
                sal_image, sal_label, sal_edge, sal_saliency = data_batch['sal_image'], data_batch['sal_label'], \
                                                               data_batch['sal_edge'], data_batch['sal_saliency']
                if (sal_image.size(2) != sal_label.size(2)) or (sal_image.size(3) != sal_label.size(3)):
                    print('IMAGE ERROR, PASSING```')
                    continue
                sal_image, sal_label, sal_edge, sal_saliency = Variable(sal_image), Variable(sal_label), Variable(
                    sal_edge), Variable(sal_saliency)
                if self.config.cuda:
                    # cudnn.benchmark = True
                    sal_image, sal_label, sal_edge, sal_saliency = sal_image.cuda(), sal_label.cuda(), sal_edge.cuda(), sal_saliency.cuda()

                feasum_out, merge_solid, out_merge_solid1, out_merge_solid2, out_merge_solid3, out_merge_solid4 = self.net(
                    sal_image)

                high_score = torch.trunc(sal_saliency, out=None)

                sal_feasum_out_loss = mulloss(feasum_out, sal_saliency, high_score)
                sal_solid_loss = mulloss(merge_solid, sal_saliency, high_score)
                sal_solid_loss1 = mulloss(out_merge_solid1, sal_saliency, high_score)
                sal_solid_loss2 = mulloss(out_merge_solid2, sal_saliency, high_score)
                sal_solid_loss3 = mulloss(out_merge_solid3, sal_saliency, high_score)
                sal_solid_loss4 = mulloss(out_merge_solid4, sal_saliency, high_score)
                sal_sal_loss = (
                            2 * sal_feasum_out_loss + sal_solid_loss + sal_solid_loss1 + sal_solid_loss2 + sal_solid_loss3 + sal_solid_loss4)

                map_s = feasum_out

                map_sx = torch.unsqueeze(torch.max(map_s, 3)[0], dim=3)  # ([1, 400, 1])
                map_sx = torch.squeeze(map_sx, dim=1)
                map_sy = torch.unsqueeze(torch.max(map_s, 2)[0], dim=3)  # ([1, 342, 1])
                map_sy = torch.squeeze(map_sy, dim=1)
                sum_sx = torch.sum(map_sx, dim=(1, 2), keepdim=True)
                sum_sy = torch.sum(map_sy, dim=(1, 2), keepdim=True)
                map_sx /= sum_sx
                map_sy /= sum_sy

                #################################################################################################
                semi_pred, grid = AttSampler(scale=1, dense=2)(sal_image, map_sx, map_sy)  #
                edge, grid = AttSampler(scale=1, dense=2)(sal_edge, map_sx, map_sy)  #
                label, grid = AttSampler(scale=1, dense=2)(sal_label, map_sx, map_sy)  #
                feasum_out, merge_solid, out_merge_solid1, out_merge_contour1, out_merge_solid2, out_merge_contour2, out_merge_solid3, out_merge_contour3, out_merge_solid4, out_merge_contour4 = self.net_hou(
                    semi_pred)

                data_pred = feasum_out

                ########################### restore #####################
                x_index = grid[0, 1, :, 0]  # 400
                y_index = grid[0, :, 1, 1]  # 300
                new_data_size = tuple(data_pred.shape[1:4])
                new_data = torch.empty(new_data_size[0], new_data_size[1], new_data_size[2],
                                       device=sal_image.device)
                new_data_final = torch.empty(new_data_size[0], new_data_size[1], new_data_size[2],
                                             device=sal_image.device)
                x_index = (x_index + 1) * new_data_size[2] / 2
                y_index = (y_index + 1) * new_data_size[1] / 2

                xl = 0
                grid_l = x_index[0]
                data_l = data_pred[:, :, :, 0]
                for num in range(1, len(x_index)):
                    grid_r = x_index[num]
                    xr = torch.ceil(grid_r) - 1
                    xr = xr.int()
                    data_r = data_pred[:, :, :, num]
                    for h in range(xl + 1, xr + 1):
                        if h == grid_r:
                            new_data[:, :, h] = data_r
                        else:
                            new_data[:, :, h] = ((h - grid_l) * data_r / (grid_r - grid_l)) + (
                                        (grid_r - h) * data_l / (grid_r - grid_l))
                    xl = xr
                    grid_l = grid_r
                    data_l = data_r
                new_data[:, :, 0] = new_data[:, :, 1]
                try:
                    for h in range(xr + 1, len(x_index)):
                        new_data[:, :, h] = new_data[:, :, xr]
                except:
                    print('h', h)
                    print('xr', xr)

                yl = 0
                grid1_l = y_index[0]
                data1_l = new_data[:, 0, :]
                for num in range(1, len(y_index)):
                    grid1_r = y_index[num]
                    yr = torch.ceil(grid1_r) - 1
                    yr = yr.int()
                    data1_r = new_data[:, num, :]
                    for h in range(yl + 1, yr + 1):
                        if h == grid1_r:
                            new_data_final[:, h, :] = data1_r
                        else:
                            new_data_final[:, h, :] = ((h - grid1_l) * data1_r / (grid1_r - grid1_l)) + (
                                        (grid1_r - h) * data1_l / (grid1_r - grid1_l))
                    yl = yr
                    grid1_l = grid1_r
                    data1_l = data1_r
                new_data_final[:, 0, :] = new_data_final[:, 1, :]
                try:
                    for h in range(yr + 1, len(y_index)):
                        new_data_final[:, h, :] = new_data_final[:, yr, :]
                except:
                    print('h', h)
                    print('yr', yr)
                new_data_final = torch.unsqueeze(new_data_final, dim=1)


                ########################### loss function #####################
                solid_loss = F.binary_cross_entropy(new_data_final, sal_label, reduction='mean') + iou_loss(
                    new_data_final, sal_label)
                solid_loss0 = F.binary_cross_entropy(merge_solid, label, reduction='mean') + iou_loss(merge_solid,
                                                                                                      label)
                solid_loss1 = F.binary_cross_entropy(out_merge_solid1, label, reduction='mean') + iou_loss(
                    out_merge_solid1, label)
                edge_loss1 = bce2d(out_merge_contour1, edge, reduction='mean')
                solid_loss2 = F.binary_cross_entropy(out_merge_solid2, label, reduction='mean') + iou_loss(
                    out_merge_solid2, label)
                edge_loss2 = bce2d(out_merge_contour2, edge, reduction='mean')
                solid_loss3 = F.binary_cross_entropy(out_merge_solid3, label, reduction='mean') + iou_loss(
                    out_merge_solid3, label)
                edge_loss3 = bce2d(out_merge_contour3, edge, reduction='mean')
                solid_loss4 = F.binary_cross_entropy(out_merge_solid4, label, reduction='mean') + iou_loss(
                    out_merge_solid4, label)
                edge_loss4 = bce2d(out_merge_contour4, edge, reduction='mean')

                final_sal_loss = (
                            2 * solid_loss + solid_loss0 + edge_loss1 + solid_loss1 + edge_loss2 + solid_loss2 + edge_loss3 + solid_loss3 + edge_loss4 + solid_loss4)
                sal_loss = (sal_sal_loss + final_sal_loss) / (self.iter_size * self.config.batch_size)

                sal_loss_fuse = F.binary_cross_entropy(new_data_final, sal_label, reduction='sum')
                sal_loss_fuse1 = F.binary_cross_entropy(new_data_final, sal_label, reduction='sum')
                r_sal_loss += sal_loss_fuse.data
                r_sal_loss1 += sal_loss_fuse1.data
                x_showEvery += 1

                sal_loss.backward()

                aveGrad += 1

                if aveGrad % self.iter_size == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    aveGrad = 0

                if i % (self.show_every // self.config.batch_size) == 0:
                    #                     if i == 0:
                    #                         x_showEvery = 1
                    print('epoch: [%2d/%2d], iter: [%5d/%5d]  ||  Sal : %10.4f ||  Sal1 : %10.4f' % (
                        epoch, self.config.epoch, i, iter_num, r_sal_loss / x_showEvery, r_sal_loss1 / x_showEvery))
                    print('Learning rate: ' + str(self.lr))
                    r_sal_loss = 0
                    r_sal_loss1 = 0
                    x_showEvery = 0

            if (epoch + 1) % self.config.epoch_save == 0:
                torch.save(self.net.state_dict(), '%s/models/epoch_net_%d.pth' % (self.config.save_folder, epoch + 1))
                torch.save(self.net_hou.state_dict(), '%s/models/epoch_%d.pth' % (self.config.save_folder, epoch + 1))

            if epoch in self.lr_decay_epoch:
                self.lr = self.lr * 0.1
                self.optimizer = Adam([{'params': filter(lambda p: p.requires_grad, self.net.parameters())},
                                       {'params': filter(lambda p: p.requires_grad, self.net_hou.parameters())}],
                                      lr=self.lr,
                                      weight_decay=self.wd)
        torch.save(self.net.state_dict(), '%s/models/net_final.pth' % self.config.save_folder)
        torch.save(self.net_hou.state_dict(), '%s/models/final.pth' % self.config.save_folder)


def bce2d(input, target, reduction=None):
    assert (input.size() == target.size())
    pos = torch.eq(target, 1).float()
    neg = torch.eq(target, 0).float()

    num_pos = torch.sum(pos)
    num_neg = torch.sum(neg)
    num_total = num_pos + num_neg

    alpha = num_neg / num_total
    beta = 1.1 * num_pos / num_total
    # target pixel = 1 -> weight beta
    # target pixel = 0 -> weight 1-beta
    weights = alpha * pos + beta * neg

    return F.binary_cross_entropy_with_logits(input, target, weights, reduction=reduction)


def _iou(pred, target, size_average=True):
    b = pred.shape[0]
    IoU = 0.0
    for i in range(0, b):
        # compute the IoU of the foreground
        Iand1 = torch.sum(target[i, :, :, :] * pred[i, :, :, :])
        Ior1 = torch.sum(target[i, :, :, :]) + torch.sum(pred[i, :, :, :]) - Iand1
        IoU1 = Iand1 / Ior1

        # IoU loss is (1-IoU1)
        IoU = IoU + (1 - IoU1)

    return IoU / b


class IOU(torch.nn.Module):
    def __init__(self, size_average=True):
        super(IOU, self).__init__()
        self.size_average = size_average

    def forward(self, pred, target):
        return _iou(pred, target, self.size_average)


iou_loss = IOU(size_average=True)


class Mul_loss(nn.Module):
    """L1 Charbonnierloss."""

    def __init__(self):
        super(Mul_loss, self).__init__()
        self.eps = 1e-8

    def forward(self, x, y, gt):
        vx = x - torch.mean(x)
        vy = y - torch.mean(y)
        CCloss = torch.sum(vx * vy) / ((torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2))) + self.eps)

        x_map_norm = (x - torch.mean(x)) / (torch.std(x) + self.eps)
        y_map_norm = (y - torch.mean(y)) / (torch.std(y) + self.eps)
        diff = torch.abs(x_map_norm - y_map_norm)
        m = torch.sum(torch.mul(diff, gt))
        # print(m)
        num = torch.sum(gt) + self.eps  #
        NSSloss = torch.div(m, num)

        max_x = torch.max(x)
        x = x / max_x
        sum_x = torch.sum(x)
        sum_y = torch.sum(y)
        x = x / (sum_x + self.eps)
        y = y / (sum_y + self.eps)
        KLDloss = torch.sum(y * torch.log(self.eps + y / (x + self.eps)))
        return 1 - CCloss + NSSloss + KLDloss

mulloss = Mul_loss()
