# Residual Dense Network for Image Super-Resolution
# https://arxiv.org/abs/1802.08797

from model import common

import torch
import torch.nn as nn


def make_model(args, parent=False):
    return LRDN(args)

class RDB_Conv(nn.Module):
    def __init__(self, inChannels, growRate, kSize=3):
        super(RDB_Conv, self).__init__()
        Cin = inChannels
        G  = growRate
        self.conv = nn.Sequential(*[
            nn.Conv2d(Cin, G, kSize, padding=(kSize-1)//2, stride=1),
            nn.ReLU()
        ])

    def forward(self, x):
        out = self.conv(x)
        return torch.cat((x, out), 1)

class RDB(nn.Module):
    def __init__(self, growRate0, growRate, nConvLayers, kSize=3):
        super(RDB, self).__init__()
        #G0 = growRate0
        #G  = growRate
        #C  = nConvLayers
        G0 = 32
        G  = 16
        C  = 6
        #G0 = 16
        #G  = 8
        #C  = 2

        convs = []
        for c in range(C):
            convs.append(RDB_Conv(G0 + c*G, G))
        self.convs = nn.Sequential(*convs)
       
        self.LFF = nn.Conv2d(G0 + C*G, G0, 1, padding=0, stride=1)

    def forward(self, x):
        return self.LFF(self.convs(x)) + x

class LRDN(nn.Module):
    def __init__(self, args):
        super(LRDN, self).__init__()

#        G0 = args.G0
#        kSize = args.RDNkSize
        kSize = 3
         
        G0 = 32
        self.D = 4
        G = 16
        C = 6
        #C = 2
        # Shallow feature extraction net
        self.SFENet1 = nn.Conv2d(args.n_colors, G0, kSize, padding=(kSize-1)//2, stride=1)
        self.SFENet2 = nn.Conv2d(G0, G0, kSize, padding=(kSize-1)//2, stride=1)

        # Redidual dense blocks and dense feature fusion
        self.RDBs = nn.ModuleList()
        for i in range(self.D):
            self.RDBs.append(
                RDB(growRate0 = G0, growRate = G, nConvLayers = C)
            )

        self.RDBs2 = nn.ModuleList()
        for i in range(self.D):
            self.RDBs2.append(
                RDB(growRate0 = G0, growRate = G, nConvLayers = C)
            )

        self.RDBs4 = nn.ModuleList()
        for i in range(self.D):
            self.RDBs4.append(
                RDB(growRate0 = G0, growRate = G, nConvLayers = C)
            )

        self.RDBs8 = nn.ModuleList()
        for i in range(self.D):
            self.RDBs8.append(
                RDB(growRate0 = G0, growRate = G, nConvLayers = C)
            )

        self.compress8 = nn.Conv2d(self.D * G0, G0, 1, padding=0, stride=1)
        self.compress4 = nn.Conv2d(self.D * G0, G0, 1, padding=0, stride=1)
        self.compress2 = nn.Conv2d(self.D * G0, G0, 1, padding=0, stride=1)

        # Global Feature Fusion
        self.GFF = nn.Sequential(*[
            nn.Conv2d(self.D * G0, G0, 1, padding=0, stride=1),
            nn.Conv2d(G0, G0, kSize, padding=(kSize-1)//2, stride=1)
        ])

        # Up-sampling net
        self.UPNet = nn.Sequential(*[
                nn.Conv2d(G0, G, kSize, padding=(kSize-1)//2, stride=1),
                nn.Conv2d(G, args.n_colors, kSize, padding=(kSize-1)//2, stride=1)
            ]) 

        self.Down1 = nn.Conv2d(G0, G0, kSize, padding=(kSize-1)//2, stride=2)
        self.Down2 = nn.Conv2d(G0, G0, kSize, padding=(kSize-1)//2, stride=2)
        self.Down3 = nn.Conv2d(G0, G0, kSize, padding=(kSize-1)//2, stride=2)

#        self.Up1 = nn.ConvTranspose2d(G0, G0, kSize+1, stride=2, padding=1)
#        self.Up2 = nn.ConvTranspose2d(G0, G0, kSize+1, stride=2, padding=1)
#        self.Up3 = nn.ConvTranspose2d(G0, G0, kSize+1, stride=2, padding=1)

        self.Up1 = nn.Upsample(scale_factor=2, mode='bilinear',align_corners=True)
        self.Up2 = nn.Upsample(scale_factor=2, mode='bilinear',align_corners=True)
        self.Up3 = nn.Upsample(scale_factor=2, mode='bilinear',align_corners=True)
        self.Up12 = nn.Conv2d(G0, G0, kSize, padding=(kSize-1)//2, stride=1)
        self.Up22 = nn.Conv2d(G0, G0, kSize, padding=(kSize-1)//2, stride=1)
        self.Up32 = nn.Conv2d(G0, G0, kSize, padding=(kSize-1)//2, stride=1)

    def forward(self, x):
        f__1 = self.SFENet1(x)
        x  = self.SFENet2(f__1)
        x1 = self.Down1(x)
        x2 = self.Down2(x1)
        x3 = self.Down2(x2)

        x_ori = x
        RDBs_out8 = []
        x3_ori = x3
        for i in range(self.D):
            x3 = self.RDBs8[i](x3)
            RDBs_out8.append(x3)
        RDBs_out_feat8= torch.cat(RDBs_out8, 1)
        x3_out1 = self.compress8(RDBs_out_feat8) + x3_ori
        x3_out = self.Up32(self.Up3(x3_out1))

        x2 = x2 + x3_out 
        RDBs_out4 = []
        x2_ori = x2
        for i in range(self.D):
            x = self.RDBs4[i](x2)
            RDBs_out4.append(x2)
        RDBs_out_feat4= torch.cat(RDBs_out4, 1)
        x2_out = self.compress4(RDBs_out_feat4) + x2_ori
        x2_out = self.Up22(self.Up2(x2_out))

        x1 = x1 + x2_out
        RDBs_out2 = []
        x1_ori = x1
        for i in range(self.D):
            x1 = self.RDBs2[i](x1)
            RDBs_out2.append(x1)
        RDBs_out_feat2= torch.cat(RDBs_out2, 1)
        x1_out = self.compress2(RDBs_out_feat2) + x1_ori
        x1_out = self.Up12(self.Up1(x1_out))

        x = x_ori + x1_out        
        RDBs_out = []
        x_ori = x
        for i in range(self.D):
            x = self.RDBs[i](x)
            RDBs_out.append(x)
        RDBs_out_feat= torch.cat(RDBs_out,1)
        x = self.GFF(RDBs_out_feat) + x_ori
        return self.UPNet(x)
