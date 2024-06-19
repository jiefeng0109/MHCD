import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from CSANetMain import attontion
import torch




class FeatureFusionModule(nn.Module):
    def __init__(self, input_channels):
        super(FeatureFusionModule, self).__init__()

        # Step 1: Feature Summation
        # self.feature_sum = nn.Conv2d(input_channels, 1, kernel_size=1)

        # Step 2: Channel Compression

        self.conv_1 = nn.Conv2d(input_channels, 128, kernel_size=3,padding=1)
        self.conv_2 = nn.Conv2d(input_channels, 128, kernel_size=3,padding=1)
        self.conv_3 = nn.Conv2d(128, 64, kernel_size=3,padding=1)
        self.channel_compression = nn.Conv2d(1, 2, kernel_size=3, padding=1)

    def forward(self, feature1, feature2):
        # Step 1: Feature Summation

        # Step 2: Channel Compression
        f = self.conv_1(feature1+ feature2)+self.conv_2(feature1 + feature2)
        f = self.conv_3(f)
        f = f.mean(dim=1, keepdim=True)
        compressed_feature = self.channel_compression(f)

        # Step 3: Matrix Generation and Softmax Normalization

        weight = F.softmax(compressed_feature, dim=1)
        a = torch.unsqueeze(weight[:,0,:,:],1)
        b = torch.unsqueeze(weight[:,1,:,:],1)

        # Step 4: Element-wise Multiplication with Original Features
        output1 = (feature1+ feature2) * a + feature1
        output2 = (feature1+ feature2) * b + feature2

        return output1, output2


class FeatureMultiscale(nn.Module):
    def __init__(self, input_channels):
        super(FeatureMultiscale, self).__init__()

        self.conv_4 = nn.Conv2d(128, 64, kernel_size=3,padding=1)
        self.conv_5 = nn.Conv2d(64, 32, kernel_size=3,padding=1)

        self.conv_1 = nn.Conv2d(input_channels, 128, kernel_size=3,padding=1)
        self.conv_2 = nn.Conv2d(input_channels, 128, kernel_size=3,padding=1)
        self.conv_3 = nn.Conv2d(input_channels, 128, kernel_size=3,padding=1)
        self.GAP = nn.AdaptiveAvgPool2d((1,1))
        self.channel_compression = nn.Conv2d(128, 64, kernel_size=1)
        self.channel_expand = nn.Conv2d(64, input_channels*3, kernel_size=1)

    def forward(self, feature1, feature2,feature3):
        # Step 1: Feature Summation
        f = feature1 + feature2 + feature3
        f = self.conv_1(f)+self.conv_2(f)+self.conv_3(f)
        # f = self.conv_5(self.conv_4(f))
        f = F.adaptive_avg_pool2d(f, (1, 1))
        compressed_feature = self.channel_compression(f)
        expand_feature = self.channel_expand(compressed_feature)
        b,w,x,y = feature1.size()
        expand_feature = expand_feature.view(b,w,3,1,1)

        weight = F.softmax(expand_feature, dim=2)
        a =weight[:,:,0,:,:]
        b =weight[:,:,1,:,:]
        c =weight[:,:,2,:,:]

        output1 = feature1 * a
        output2 = feature2 * b
        output3 = feature2 * c

        return output1+output2+output3

class deeplab_V2(nn.Module):
    def __init__(self):
        super(deeplab_V2, self).__init__()
        self.fusion1 = FeatureFusionModule(128)
        self.fusion2= FeatureFusionModule(128)
        self.multiattentionfusion1= FeatureMultiscale(128)
        self.multiattentionfusion2= FeatureMultiscale(128)
        # self.fusion_add_1= FeatureFusionModule2(64)
        # self.fusion_add_2= FeatureFusionModule2(64)
        # self.fusion_add_3= FeatureFusionModule2(64)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=155, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True),

        )
        '''
        '''
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True),

        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),#![](classification_maps/IN_gt.png)
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True),

        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1, ceil_mode=True),

        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, dilation=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, dilation=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, dilation=2, padding=2),
            nn.ReLU(inplace=True),

        )  #进行卷积操作


        inter_channels = 512 // 4###################################################
        self.conv5a = nn.Sequential(nn.Conv2d(512, inter_channels, 3, padding=1, bias=False),

                                    nn.ReLU())

        self.conv5c = nn.Sequential(nn.Conv2d(512, inter_channels, 3, padding=1, bias=False),

                                    nn.ReLU())

        self.sa = attontion.PAM_Module(inter_channels)####
        self.sc = attontion.CAM_Module(inter_channels)
        self.sco = attontion.CoAM_Module(inter_channels)###
        self.conv51 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),

                                    nn.ReLU())
        self.conv52 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),

                                    nn.ReLU())


        self.conv8 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(128, 128,  3, padding=1))
        # self.conv9 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(128, 128,  3, padding=1))


    def forward(self, x1, x2):
        x1 = self.conv1(x1)
        x1 = self.conv2(x1)
        x2 = self.conv1(x2)
        x2 = self.conv2(x2)


        feat11 = self.conv5a(x1)
        feat21 = self.conv5a(x2)

        feat12 = self.conv5c(x1)
        feat22 = self.conv5c(x2)

        feat11, feat21 = self.fusion1(feat11, feat21)
        feat12, feat22 = self.fusion2(feat12, feat22)
######################################################################
        #生成sa attention
        sa_feat1 = self.sa(feat11)
        sa_feat2 = self.sa(feat21)

        sa_conv1 = self.conv51(sa_feat1)
        sa_conv2 = self.conv51(sa_feat2)

        # 生成sc attention
        sc_feat1 = self.sc(feat12)
        sc_feat2 = self.sc(feat22)

        sc_conv1 = self.conv52(sc_feat1)
        sc_conv2 = self.conv52(sc_feat2)


        #生成temporal
        sco_conv1 = self.sco(feat11,feat12)
        sco_conv1 = self.conv51(sco_conv1)
        sco_conv2 = self.sco(feat21,feat22)
        sco_conv2 = self.conv51(sco_conv2)


        # sco_conv1,sco_conv2 = self.fusion(sco_conv1,sco_conv2)
        feat_sum1 = self.multiattentionfusion1(sa_conv1,sc_conv1,sco_conv1)
        feat_sum2 = self.multiattentionfusion2(sa_conv2,sc_conv2,sco_conv2)
        # feat_sum1 = sa_conv1 + sc_conv1 + 0.3*sco_conv1
        # feat_sum2 = sa_conv2 + sc_conv2 + 0.3*sco_conv2


        sasc_output1 = self.conv8(feat_sum1)
        sasc_output2 = self.conv8(feat_sum2)



        return sasc_output1, sasc_output2


class SiameseNet(nn.Module):
    def __init__(self, norm_flag='l2'):
        super(SiameseNet, self).__init__()
        self.CNN = deeplab_V2()

        if norm_flag == 'l2':
            self.norm = F.normalize  #F.normalize对输入的数据（tensor）进行指定维度的L2_norm运算
        if norm_flag == 'exp':
            self.norm = nn.Softmax2d()

    def forward(self, t0, t1):
        t0 = t0.float()
        t1 = t1.float()
        out_t0_embedding, out_t1_embedding, = self.CNN(t0, t1)
        return out_t0_embedding, out_t1_embedding


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.conv1 = nn.Conv2d(128, 64, 3, padding=1, padding_mode='reflect', bias=False)  # [64, 24, 24]
        self.bat1 = nn.BatchNorm2d(64)#
        self.reli1 = nn.LeakyReLU(0.2)
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = nn.Conv2d(64, 32, 3, padding=1, padding_mode='reflect', bias=False)
        self.bat2 = nn.BatchNorm2d(32)
        self.reli2 = nn.LeakyReLU(0.2)

    def forward(self, x):
        con1 = self.conv1(x)
        ba1 = self.bat1(con1)
        re1 = self.reli1(ba1)
        po1 = self.pool1(re1)
        con2 = self.conv2(po1)
        ba2 = self.bat2(con2)
        re2 = self.reli2(ba2)

        return re2


class ChangeNet(nn.Module):
    def __init__(self):
        super(ChangeNet, self).__init__()
        self.singlebrach = Classifier()# re2

    def forward(self, t0, t1):

        indata = t0 - t1
        c3 = self.singlebrach(indata)

        return c3


class BSNET_Conv(nn.Module):

    def __init__(self, ):
        super(BSNET_Conv, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(155, 64, (3, 3), 1, 0),
            nn.ReLU(True))

        self.conv1_1 = nn.Sequential(
            nn.Conv2d(155, 128, (3, 3), 1, 0),
            nn.ReLU(True))
        self.conv1_2 = nn.Sequential(
            nn.Conv2d(128, 64, (3, 3), 1, 0),
            nn.ReLU(True))

        self.deconv1_2 = nn.Sequential(
            nn.ConvTranspose2d(64, 64, (3, 3), 1, 0),
            nn.ReLU(True))

        self.deconv1_1 = nn.Sequential(
            nn.ConvTranspose2d(64, 128, (3, 3), 1, 0),
            nn.ReLU(True))

        self.conv2_1 = nn.Sequential(
            nn.Conv2d(128, 155, (1, 1), 1, 0),
            nn.Sigmoid())

        self.fc1 = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(True))
        self.fc2 = nn.Sequential(
            nn.Linear(128, 155),
            nn.Sigmoid())

    def GlobalPool(self, feature_size):
        return nn.AvgPool2d(kernel_size=feature_size)

    def BAM(self, x):
        x = self.conv1(x)
        gp = self.GlobalPool(x.shape[2])
        x = gp(x)
        x = x.view(-1, 64)
        x = self.fc1(x)
        x = self.fc2(x)
        x = x.view(-1, 1, 1, 155)
        x = x.permute(0, 3, 1, 2)
        return x

    def RecNet(self, x):
        x = self.conv1_1(x)
        x = self.conv1_2(x)
        x = self.deconv1_2(x)
        x = self.deconv1_1(x)
        x = self.conv2_1(x)
        return x

    def forward(self, x):
        BRW = self.BAM(x)
        new_x = x * BRW
        ret = self.RecNet(new_x)

        return new_x,F.mse_loss(ret,x),BRW

class Finalmodel(nn.Module):#######################nn,Module################################################################
    def __init__(self):
        super(Finalmodel, self).__init__()
        self.siamesnet = SiameseNet()
        self.chnet = ChangeNet() #c3
        self.fc = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

        self.conv_con = nn.Sequential(
            nn.Conv2d(32, 64, (1, 1), 1, 0),
            nn.ReLU(True)
        )


        self.fc1 = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 2)
        )

        self.maxpool = nn.MaxPool2d(2)

        self.bf = BSNET_Conv()
        self.conv_re = nn.Sequential(
            nn.Conv2d(155, 128, (3, 3), 1, 0),
            nn.ReLU(True)
        )
        self.classifer = Classifier()

        self.fc_con = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU()
        )

    def forward(self, t0, t1):
        t0 = t0.permute(0, 3, 1, 2) #换个顺序 0123-----0312
        t1 = t1.permute(0, 3, 1, 2)

        t_diff = t1-t0
        t_diff, recon_error, BRW = self.bf(t_diff)

        out1 = self.classifer(self.conv_re(t_diff))
        out1 = self.maxpool(out1)
        out1 = out1.view(out1.size(0), -1)
        out1 = self.fc1(out1)
        out1 = F.sigmoid(out1)

        t0=t0*BRW
        t1=t1*BRW

        x1, x2 = self.siamesnet(t0, t1)
        out = self.chnet(x1, x2) ###转化为32通道的2-D特征
        out = self.maxpool(out)

        # contrastive_feature = self.fc_con(out.view(out.size(0), -1))
        # contrastive_feature = contrastive_feature.view(contrastive_feature.size(0),2, -1)

        feature = self.conv_con(out)
        feature = feature.view(feature.size(0), -1)
        contrastive_feature = self.fc_con(feature)
        #
        # contrastive_feature = self.fc(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        out = F.sigmoid(out)


        return out,t0,t1,recon_error,out1,contrastive_feature