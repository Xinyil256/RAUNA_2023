import torch
torch.cuda.current_device()
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
import matplotlib
#matplotlib.use('agg')
#修改了padding方式 reflect



kernel_y = torch.FloatTensor([[[[-1.0, 0, 1.0]]]]).expand(3, 1, -1, -1)
kernel_x = torch.transpose(kernel_y, dim0=2, dim1=3)
#gradient kernel
kernel_y_g = torch.FloatTensor([[[[-1.0, 0, 1.0]]]]).expand(3, 1, -1, -1)
kernel_x_g = torch.transpose(kernel_y_g, dim0=2, dim1=3)

kernel_r = 1/25 * torch.ones(size=(5,5)).unsqueeze(dim=0).unsqueeze(dim=0).expand(3,1,-1,-1)
kernel_z = 1/9*torch.ones(size=(3,3)).unsqueeze(dim=0).unsqueeze(dim=0)


def make_model(args, parent=False):
    return Mainnet(args)

class Mainnet(nn.Module):
    def __init__(self, args):
        super(Mainnet, self).__init__()
        self.S = args.stage
        self.iter = self.S-1
        self.conv_channels = args.conv_channels
        self.device = torch.device('cpu' if args.cpu else 'cuda')


        #parameters
        #TODO: True or False
        # lam=30
        self.lam = nn.Parameter(torch.FloatTensor([50]), requires_grad=True)
        self.sigma = nn.Parameter(torch.FloatTensor([10]), requires_grad=True)
        self.gamma = nn.Parameter(torch.FloatTensor([args.gam]), requires_grad=True)
        self.eps = nn.Parameter(torch.FloatTensor([0.02]), requires_grad=False)
        self.gam1 = self.make_eta(self.iter, self.gamma)  #gamma在每个阶段更新

        # # para of R0
        # # TODO: Update
        # self.lam0 = nn.Parameter(torch.FloatTensor([9]), requires_grad=False)
        # self.sigma0 = nn.Parameter(torch.FloatTensor([0.1]), requires_grad=False)

        # Stepsize
        #TODO: etaL
        self.etaL = torch.Tensor([.1])  # initialization
        self.etaR = torch.Tensor([.5])  # initialization
        self.eta1 = nn.Parameter(self.etaL, requires_grad=True)  # usd in initialization process
        self.eta2 = nn.Parameter(self.etaR, requires_grad=True)  # usd in initialization process
        self.eta11 = self.make_eta(self.iter, self.etaL)  # usd in iterative process
        self.eta12 = self.make_eta(self.iter, self.etaR)

        #  kernel
        #TODO: May update
        self.weight0_x = nn.Parameter(data=kernel_x, requires_grad=False)  # used in initialization process
        self.conv_x = self.make_weight(self.iter, kernel_x)
        self.weight0_y = nn.Parameter(data=kernel_y, requires_grad=False)  # used in initialization process
        self.conv_y = self.make_weight(self.iter, kernel_y)
        #TODO：update
        self.fuz = nn.Parameter(data=kernel_r, requires_grad=False)  # used in L0 cannot update

        # gradient kernel
        self.weight0_x_g = nn.Parameter(data=kernel_x_g, requires_grad=False)
        self.weight0_y_g = nn.Parameter(data=kernel_y_g, requires_grad=False)

        #proxnet in initialization
        self.num_L = args.num_L
        self.num_R = args.num_R
        self.lnet = Lnet(self.num_L+3)
        self.rnet = Rnet(self.num_R+3)

        self.l_stage = self.make_lnet(self.S, self.num_L+3)
        self.r_stage = self.make_rnet(self.S, self.num_R+3)

# TODO:finetune
        self.flnet = Lnet(self.num_L+3) #finetune
        self.frnet = Rnet(self.num_R+3) #finetune

        self.kernel_z_l = kernel_z.expand(self.num_L,3,-1,-1)
        self.w_l_f = nn.Parameter(self.kernel_z_l, requires_grad=True)
        self.kernel_z_r = kernel_z.expand(self.num_R,3,-1,-1)
        self.w_r_f = nn.Parameter(self.kernel_z_r, requires_grad=True)
        self.cnt = 0
        self.conv_l = nn.Conv2d(in_channels=3,out_channels=32 , kernel_size=3, stride=1, padding=1,padding_mode='reflect')
        self.conv_r = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1,
                                padding_mode='reflect')
        # self.kernel_z = kernel_z.expand(self.num_R, 3, -1, -1)
        # self.w_f = nn.Parameter(self.kernel_z, requires_grad=True)

    def make_lnet(self, iters, channel):
        layers = []
        for i in range(iters):
            layers.append(Lnet(channel))
        return nn.Sequential(*layers)

    def make_rnet(self, iters, channel):
        layers = []
        for i in range(iters):
            layers.append(Rnet(channel))
        return nn.Sequential(*layers)

    def make_eta(self, iters,const):
        const_dimadd = const.unsqueeze(dim=0)
        const_f = const_dimadd.expand(iters, -1)
        eta = nn.Parameter(data=const_f, requires_grad=True)
        return eta

    def make_weight(self, iters, const):
        const_dimadd = const.unsqueeze(dim=0)
        const_f = const_dimadd.expand(iters, -1, -1, -1, -1)
        weight = nn.Parameter(data=const_f, requires_grad=True)
        return weight


    def forward(self, input):
        # save mid-updating results
        ListR = []
        ListL = []
        ListLG_x = []
        ListLG_y = []
        padding_x = (0, 0,
                     1, 1)
        padding_y = (1, 1,
                     0, 0)
        padding0 = (2, 2,
                   2, 2)

        #initialization
        #TODO: Para of R
        R0 = torch.zeros(input.shape).to(self.device)
        R0[:,0, :, :] = (input[:,0, :, :] - torch.mean(input[:, 0, :, :]) + 0.0001) / (
                torch.std(input[:,0, :, :]) + 0.0001) * 0.1830 + 0.4717
        R0[:,1, :, :] = (input[:,1, :, :] - torch.mean(input[:,1, :, :]) + 0.0001) / (
                torch.std(input[:,1, :, :]) + 0.0001) * 0.1838 + 0.4558
        R0[:,2, :, :] = (input[:,2, :, :] - torch.mean(input[:,2, :, :]) + 0.0001) / (
                torch.std(input[:, 2, :, :]) + 0.0001) * 0.1915 + 0.4379
        #L0
        R0_inv =(input+0.0001)/(R0+0.0001)
        R0_pad = F.pad(R0_inv, padding0, mode='reflect')
        L0 = F.conv2d(R0_pad, weight=self.fuz, stride=1, padding=0, groups=3)
###########################################

        # 1st iteration: updating: R0 > L1

        U0 = R0*(R0*L0-input)
        H0 = R0 * R0
        dL0 = torch.div(U0+0.0001, H0+0.0001) #avoid div 0
        Z0_L = self.conv_l(L0)
        L = L0 - self.eta1 * dL0
        L_cat = torch.cat((L, Z0_L), dim=1)
        L_cat_new = self.l_stage[0](L_cat)
        L = L_cat_new[:, :3, :, :]
        Z_L = L_cat_new[:, 3:, :, :]

        # self.cnt +=1
        # if self.cnt%50 == 0:
        #     print(dL0)
         #control updaing speed
        # L0_pad = F.pad(L0, padding, mode='reflect')
        # Z0_L= self.conv_l(L0)
        # L_cat = torch.cat((dL0, Z0_L), dim=1)
        # L_cat_new = self.l_stage[0](L_cat)
        # dL = L_cat_new[:, :3, :, :]
        # L = L0 - self.eta1 * dL
        # Z_L = L_cat_new[:, 3:, :, :]



        ListL.append(L)

        #1st iteration: updating: L1 > R1
        A = R0 * L * L - input * L
        input_std = torch.std(input)
        # caculate G
        input_x = F.pad(input, padding_x, mode='reflect')
        input_y = F.pad(input, padding_y, mode='reflect')
        DIx = F.conv2d(input_x, weight=self.weight0_x, stride=1, padding=0,groups=3)

        DIx = DIx * (torch.abs(DIx) >= self.eps)
        DIx = DIx/(input_std * 2)
        Gx = (1 + self.lam * torch.exp(-torch.abs(DIx) / self.sigma)) * DIx
        DIy = F.conv2d(input_y, weight=self.weight0_y, stride=1, padding=0,groups=3)
        DIy = DIy * (torch.abs(DIy) >= self.eps)
        DIy = DIy/(input_std * 2)
        Gy = (1 + self.lam * torch.exp(-torch.abs(DIy) / self.sigma)) * DIy

        R0_x = F.pad(R0, padding_x, mode='reflect')
        R0_y = F.pad(R0, padding_y, mode='reflect')
        A_tilda_x = F.conv2d(R0_x, weight=self.weight0_x, stride=1, padding=0,groups=3) - Gx
        A_tilda_y = F.conv2d(R0_y, weight=self.weight0_y, stride=1, padding=0,groups=3) - Gy
        A_hat_x = F.conv_transpose2d(A_tilda_x, self.weight0_x, stride=1, padding=(1, 0),groups=3)
        A_hat_y = F.conv_transpose2d(A_tilda_y, self.weight0_y, stride=1, padding=(0, 1),groups=3)
        A_hat = self.gamma*(A_hat_x + A_hat_y)

        dR0 = (A + A_hat) / (L*L+4*self.gamma)

        R0_update = R0 - self.eta2 * dR0
        # R0_pad = F.pad(R0, padding, mode='reflect')
        # Z0_R = F.conv2d(R0_pad, weight=self.w_r_f, stride=1, padding=0)
        Z0_R = self.conv_r(R0)
        R_cat = torch.cat((R0_update, Z0_R), dim=1)
        R_cat_new = self.r_stage[0](R_cat)
        R = R_cat_new[:, :3, :, :]
        Z_R = R_cat_new[:, 3:, :, :]

        ListR.append(R)




        for i in range(self.iter):#self.iter = args.stage-1
            # Lnet

            U = R * (R * L - input)
            H = R * R
            dL = torch.div(U+0.0001, H+0.0001)
            L = L - self.eta11[i, :]/10 * dL
            L_cat = torch.cat((L, Z_L), dim=1)
            L_cat_new = self.l_stage[i + 1](L_cat)
            L = L_cat_new[:, :3, :, :]
            Z_L = L_cat_new[:, 3:, :, :]

            ListL.append(L)



             #Rnet
            A = R * L * L - input * L
            #TODO:
            # calculate G
            input_x = F.pad(input, padding_x, mode='reflect')
            input_y = F.pad(input, padding_y, mode='reflect')
            DIx = F.conv2d(input_x, weight=self.weight0_x, stride=1, padding=0,groups=3)
            DIx = DIx * (torch.abs(DIx) >= self.eps)
            DIx = DIx / (input_std * 2)
            Gx = (1 + self.lam * torch.exp(-torch.abs(DIx) / self.sigma)) * DIx
            DIy = F.conv2d(input_y, weight=self.weight0_y, stride=1, padding=0, groups=3)
            DIy = DIy * (torch.abs(DIy) >= self.eps)
            DIy = DIy / (input_std * 2)
            Gy = (1 + self.lam * torch.exp(-torch.abs(DIy) / self.sigma)) * DIy

            R_x = F.pad(R, padding_x, mode='reflect')
            R_y = F.pad(R, padding_y, mode='reflect')
            A_tilda_x = F.conv2d(R_x, weight=self.weight0_x, stride=1, padding=0,groups=3) - Gx
            A_tilda_y = F.conv2d(R_y, weight=self.weight0_y, stride=1, padding=0,groups=3) - Gy
            A_hat_x = F.conv_transpose2d(A_tilda_x, self.weight0_x, stride=1, padding=(1, 0),groups=3)
            A_hat_y = F.conv_transpose2d(A_tilda_y, self.weight0_y, stride=1, padding=(0, 1),groups=3)
            A_hat = self.gam1[i,:] * (A_hat_x + A_hat_y)

            dR = (A + A_hat)/(L*L+4*self.gamma)


            R_update = R - self.eta12[i,:]/10 * dR
            R_cat = torch.cat((R_update, Z_R), dim=1)
            R_cat_new = self.r_stage[i+1](R_cat)
            R = R_cat_new[:, :3, :, :]
            Z_R = R_cat_new[:, 3:, :, :]

            ListR.append(R)

        return L0, R0, ListL, ListR


class Rnet(nn.Module):  #####
    def __init__(self, channels):#,args):
        super(Rnet, self).__init__()
        self.channels = channels
        #self.stage = args.stage
        self.f = nn.ReLU(inplace=True)
        self.resm1 = nn.Sequential(nn.ReflectionPad2d(padding=(1, 1, 1, 1)),
                                  nn.Conv2d(self.channels, self.channels, kernel_size=3, stride = 1, padding= 0, dilation =1),
                                  nn.ReLU(),
                                  nn.ReflectionPad2d(padding=(1, 1, 1, 1)),
                                  nn.Conv2d(self.channels, self.channels, kernel_size=3, stride = 1, padding= 0, dilation = 1),

                                   )
        self.resm2 = nn.Sequential(nn.ReflectionPad2d(padding=(1, 1, 1, 1)),
                                  nn.Conv2d(self.channels, self.channels, kernel_size=3, stride = 1, padding=0, dilation = 1),

                                  nn.ReLU(),
                                  nn.ReflectionPad2d(padding=(1, 1, 1, 1)),
                                  nn.Conv2d(self.channels, self.channels, kernel_size=3, stride = 1, padding= 0, dilation = 1),

                                  )
        self.resm3 = nn.Sequential(nn.ReflectionPad2d(padding=(1, 1, 1, 1)),
                                   nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=0, dilation=1),
                                   nn.ReLU(),
                                   nn.ReflectionPad2d(padding=(1, 1, 1, 1)),
                                   nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=0, dilation=1),

                                   )
        self.resm4 = nn.Sequential(nn.ReflectionPad2d(padding=(1, 1, 1, 1)),
                                   nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=0, dilation=1),
                                   nn.ReLU(),
                                   nn.ReflectionPad2d(padding=(1, 1, 1, 1)),
                                   nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=0, dilation=1),
                                   )
#TODO: sigmoid of last stage?
    def forward(self, input):
        m1 = F.leaky_relu(input + self.resm1(input))
        m2 = F.leaky_relu(m1 + self.resm2(m1))
        m3 = F.leaky_relu(m2 + self.resm3(m2))
        m4 = F.leaky_relu(m3 + self.resm4(m3))
        return m4




class Lnet(nn.Module):
    def __init__(self, n_channels):
        super(Lnet, self).__init__()
        self.channels = n_channels

        def add_conv_stage(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=True):
            return nn.Sequential(
                nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size, stride=stride, padding=padding, padding_mode='reflect'),
                nn.ReLU(),
                nn.Conv2d(dim_out, dim_out, kernel_size=kernel_size, stride=stride, padding=padding, padding_mode='reflect'),
                nn.ReLU()
            )

        # Up sampling module
        def upsample(ch_coarse, ch_fine):
            return nn.Sequential(
                nn.ConvTranspose2d(ch_coarse, ch_fine, 4, 2, 1, bias=False),
                nn.ReLU()
            )

        self.conv1 = add_conv_stage(self.channels, 32)
        self.conv2 = add_conv_stage(32, 64)
        self.conv3 = add_conv_stage(64, 128)
        self.conv4 = add_conv_stage(128, 256)
        self.conv3m = add_conv_stage(256, 128)
        self.conv2m = add_conv_stage(128, 64)
        self.conv1m = add_conv_stage(64, 32)

        self.conv0 = nn.Sequential(
            nn.Conv2d(32, self.channels, 3, 1, 1),
            nn.Sigmoid()
        )

        self.max_pool = nn.MaxPool2d(2,ceil_mode=True)


        self.upsample43 = upsample(256, 128)
        self.upsample32 = upsample(128, 64)
        self.upsample21 = upsample(64, 32)

    def forward(self, x):
        conv1_out = self.conv1(x)
        conv2_out = self.conv2(self.max_pool(conv1_out))
        conv3_out = self.conv3(self.max_pool(conv2_out))
        conv4_out = self.conv4(self.max_pool(conv3_out))
        conv4m_out_ = torch.cat((self.upsample43(conv4_out), conv3_out), 1)
        conv3m_out = self.conv3m(conv4m_out_)
        conv3m_out_ = torch.cat((self.upsample32(conv3m_out), conv2_out), 1)
        conv2m_out = self.conv2m(conv3m_out_)
        conv2m_out_ = torch.cat((self.upsample21(conv2m_out), conv1_out), 1)
        conv1m_out = self.conv1m(conv2m_out_)
        conv0_out = self.conv0(conv1m_out)

        return conv0_out








