# pylint: disable=all
import torch
import torch.nn as nn
import numpy as np
from torch.optim import AdamW
import torch.optim as optim
import itertools
import torch.nn.functional as F

try:
    from .loss import *
    from .warplayer import warp
    from .IFNet_HDv2 import *
except ImportError:
    from loss import *
    from warplayer import warp
    from IFNet_HDv2 import *

def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=True),
        nn.PReLU(out_planes)
    )


def deconv(in_planes, out_planes, kernel_size=4, stride=2, padding=1):
    return nn.Sequential(
        torch.nn.ConvTranspose2d(in_channels=in_planes, out_channels=out_planes,
                                 kernel_size=4, stride=2, padding=1, bias=True),
        nn.PReLU(out_planes)
    )

class Conv2(nn.Module):
    def __init__(self, in_planes, out_planes, stride=2):
        super(Conv2, self).__init__()
        self.conv1 = conv(in_planes, out_planes, 3, stride, 1)
        self.conv2 = conv(out_planes, out_planes, 3, 1, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class ContextNet(nn.Module):
    def __init__(self):
        c = 32
        super(ContextNet, self).__init__()
        self.conv0 = Conv2(3, c)
        self.conv1 = Conv2(c, c)
        self.conv2 = Conv2(c, 2*c)
        self.conv3 = Conv2(2*c, 4*c)
        self.conv4 = Conv2(4*c, 8*c)

    def forward(self, x, flow):
        x = self.conv0(x)
        x = self.conv1(x)
        flow = F.interpolate(flow, scale_factor=0.5, mode="bilinear", align_corners=False) * 0.5
        f1 = warp(x, flow)
        x = self.conv2(x)
        flow = F.interpolate(flow, scale_factor=0.5, mode="bilinear", align_corners=False) * 0.5
        f2 = warp(x, flow)
        x = self.conv3(x)
        flow = F.interpolate(flow, scale_factor=0.5, mode="bilinear", align_corners=False) * 0.5
        f3 = warp(x, flow)
        x = self.conv4(x)
        flow = F.interpolate(flow, scale_factor=0.5, mode="bilinear", align_corners=False) * 0.5
        f4 = warp(x, flow)
        return [f1, f2, f3, f4]


class FusionNet(nn.Module):
    def __init__(self):
        super(FusionNet, self).__init__()
        c = 32
        self.conv0 = Conv2(10, c)
        self.down0 = Conv2(c, 2*c)
        self.down1 = Conv2(4*c, 4*c)
        self.down2 = Conv2(8*c, 8*c)
        self.down3 = Conv2(16*c, 16*c)
        self.up0 = deconv(32*c, 8*c)
        self.up1 = deconv(16*c, 4*c)
        self.up2 = deconv(8*c, 2*c)
        self.up3 = deconv(4*c, c)
        self.conv = nn.ConvTranspose2d(c, 4, 4, 2, 1)

    def forward(self, img0, img1, flow, c0, c1, flow_gt):
        warped_img0 = warp(img0, flow[:, :2])
        warped_img1 = warp(img1, flow[:, 2:4])
        if flow_gt == None:
            warped_img0_gt, warped_img1_gt = None, None
        else:
            warped_img0_gt = warp(img0, flow_gt[:, :2])
            warped_img1_gt = warp(img1, flow_gt[:, 2:4])
        x = self.conv0(torch.cat((warped_img0, warped_img1, flow), 1))
        s0 = self.down0(x)
        s1 = self.down1(torch.cat((s0, c0[0], c1[0]), 1))
        s2 = self.down2(torch.cat((s1, c0[1], c1[1]), 1))
        s3 = self.down3(torch.cat((s2, c0[2], c1[2]), 1))
        x = self.up0(torch.cat((s3, c0[3], c1[3]), 1))
        x = self.up1(torch.cat((x, s2), 1))
        x = self.up2(torch.cat((x, s1), 1))
        x = self.up3(torch.cat((x, s0), 1))
        x = self.conv(x)
        return x, warped_img0, warped_img1, warped_img0_gt, warped_img1_gt


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.flownet = IFNet()
        self.contextnet = ContextNet()
        self.fusionnet = FusionNet()
        self.optimG = AdamW(itertools.chain(
            self.flownet.parameters(),
            self.contextnet.parameters(),
            self.fusionnet.parameters()), lr=1e-6, weight_decay=1e-5)
        self.schedulerG = optim.lr_scheduler.CyclicLR(
            self.optimG, base_lr=1e-6, max_lr=1e-3, step_size_up=8000, cycle_momentum=False)
        self.epe = EPE()
        self.ter = Ternary()
        self.sobel = SOBEL()

    def load_model(self, path):
        def convert(param):
            return {k.replace("module.", ""): v for k, v in param.items() if "module." in k}
        self.flownet.load_state_dict(convert(torch.load('{}/flownet.ckpt'.format(path), map_location="cpu")))
        self.contextnet.load_state_dict(convert(torch.load('{}/contextnet.ckpt'.format(path), map_location="cpu")))
        self.fusionnet.load_state_dict(convert(torch.load('{}/unet.ckpt'.format(path), map_location="cpu")))

    def save_model(self, path):
        torch.save(self.flownet.state_dict(), '{}/flownet.ckpt'.format(path))
        torch.save(self.contextnet.state_dict(), '{}/contextnet.ckpt'.format(path))
        torch.save(self.fusionnet.state_dict(), '{}/unet.ckpt'.format(path))

    def predict(self, imgs, flow, training=True, flow_gt=None, UHD=False):
        img0 = imgs[:, :3]
        img1 = imgs[:, 3:]
        if UHD:
            flow = F.interpolate(flow, scale_factor=2.0, mode="bilinear", align_corners=False) * 2.0
        c0 = self.contextnet(img0, flow[:, :2])
        c1 = self.contextnet(img1, flow[:, 2:4])
        flow = F.interpolate(flow, scale_factor=2.0, mode="bilinear",
                             align_corners=False) * 2.0
        """
        refine_output, warped_img0, warped_img1, warped_img0_gt, warped_img1_gt = self.fusionnet(
            img0, img1, flow, c0, c1, flow_gt)
        res = torch.sigmoid(refine_output[:, :3]) * 2 - 1
        mask = torch.sigmoid(refine_output[:, 3:4])
        merged_img = warped_img0 * mask + warped_img1 * (1 - mask)
        pred = merged_img + res
        pred = torch.clamp(pred, 0, 1)
        if training:
            return pred, mask, merged_img, warped_img0, warped_img1, warped_img0_gt, warped_img1_gt
        else:
            return pred, flow, mask, res
        """
        return flow

    def inference(self, img0, img1, UHD=False, no_backward=True):
        imgs = torch.cat((img0, img1), 1)
        flow, _ = self.flownet(imgs, UHD)
        #print(flow.shape)
        if no_backward:
            flow = flow[:,2:, :,:]
        #flow = F.interpolate(flow, scale_factor=2.0, mode="bilinear",
        #                     align_corners=False) * 2.0
        flow = flow * 2.0
        return flow #self.predict(imgs, flow, training=False, UHD=UHD)

    def update(self, imgs, gt, learning_rate=0, mul=1, training=True, flow_gt=None):
        for param_group in self.optimG.param_groups:
            param_group['lr'] = learning_rate
        if training:
            self.train()
        else:
            self.eval()
        flow, flow_list = self.flownet(imgs)
        pred, mask, merged_img, warped_img0, warped_img1, warped_img0_gt, warped_img1_gt = self.predict(
            imgs, flow, flow_gt=flow_gt)
        loss_ter = self.ter(pred, gt).mean()
        if training:
            with torch.no_grad():
                loss_flow = torch.abs(warped_img0_gt - gt).mean()
                loss_mask = torch.abs(
                    merged_img - gt).sum(1, True).float().detach()
                loss_mask = F.interpolate(loss_mask, scale_factor=0.5, mode="bilinear",
                                          align_corners=False).detach()
                flow_gt = (F.interpolate(flow_gt, scale_factor=0.5, mode="bilinear",
                                         align_corners=False) * 0.5).detach()
            loss_cons = 0
            for i in range(4):
                loss_cons += self.epe(flow_list[i][:, :2], flow_gt[:, :2], 1)
                loss_cons += self.epe(flow_list[i][:, 2:4], flow_gt[:, 2:4], 1)
            loss_cons = loss_cons.mean() * 0.01
        else:
            loss_cons = torch.tensor([0])
            loss_flow = torch.abs(warped_img0 - gt).mean()
            loss_mask = 1
        loss_l1 = (((pred - gt) ** 2 + 1e-6) ** 0.5).mean()
        if training:
            self.optimG.zero_grad()
            loss_G = loss_l1 + loss_cons + loss_ter
            loss_G.backward()
            self.optimG.step()
        return pred, merged_img, flow, loss_l1, loss_flow, loss_cons, loss_ter, loss_mask


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img0 = torch.zeros(3, 3, 256, 256).float().to(device)
    img1 = torch.tensor(np.random.normal(0, 1, (3, 3, 256, 256))).float().to(device)
    imgs = torch.cat((img0, img1), 1)
    model = Model().to(device)
    model.eval()
    print(model.inference(imgs).shape)
