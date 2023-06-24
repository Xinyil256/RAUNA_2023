import torch
torch.cuda.current_device()
import torch.nn as nn
import torch.nn.functional as F




def make_model(args, parent=False):
    return Mainnet(args)

class Mainnet(nn.Module):
    def __init__(self, args):
        super(Mainnet, self).__init__()
        self.renet = Renet()
        self.lenet = Lenet()
        self.anet = Anet()
        self.cnt = 0
        self.device = torch.device('cpu' if args.cpu else 'cuda')


    def forward(self, input_r, input_l, a):
        b, _,_,_ = input_r.shape
        al = a.unsqueeze(dim=1).unsqueeze(dim=2).unsqueeze(dim=3).expand(b, 1, 1, 1).to(self.device)
        al = self.anet(al)
        R = self.renet(input_r,al)
        L = self.lenet(input_l, al)
        # if self.cnt %5 ==0:
        #     print(L)
        #     self.cnt += 1
        return R, L, al
# class Lenet(nn.Module):
#     def __init__(self):
#         super(Lenet, self).__init__()
#         self.channels = 4
#         self.conv1 = nn.Sequential(nn.Conv2d(in_channels=self.channels, out_channels=32,kernel_size=3,
#                                             padding_mode='reflect', padding=1),
#                                   nn.LeakyReLU(),
#         )
#         self.conv2 = nn.Sequential(nn.Conv2d(in_channels=32, out_channels=32,kernel_size=3,
#                                             padding_mode='reflect', padding=1),
#                                   nn.LeakyReLU(),
#                                    )
#         self.conv3 = nn.Sequential(nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3,
#                                              padding_mode='reflect', padding=1),
#                                    nn.LeakyReLU(),
#                                    )
#
#     def forward(self,input):
#         conv1 = self.conv1(input)
#         conv2 = self.conv2(conv1)
#         conv3 = self.conv3(conv2)
#         # conv4 = self.conv4(conv3)
#         out = F.sigmoid(conv3)
#         return out

class Lenet(nn.Module):
    def __init__(self):
        super(Lenet, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.res1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1, padding_mode='reflect', bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, padding=1, padding_mode='reflect', bias=True),
        )
        self.res2 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1, padding_mode='reflect', bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, padding=1, padding_mode='reflect', bias=True),
        )
        self.scale = nn.Sequential(
            nn.Linear(1, 100, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(100, 1, bias=True),
            nn.Sigmoid()
        )
    def forward(self, input, a):
        scale = self.scale(a)
        # print('.......................')
        # print(scale)
        # print('.......................')
        res1 = self.res1(input)
        out1 = self.relu(input + scale*res1)
        out2 = self.relu(out1 + scale * self.res2(out1))

        return out2
# class Lenet(nn.Module):
#     def __init__(self):
#         super(Lenet, self).__init__()
#         self.conv = nn.Sequential(
#             nn.Conv2d(2, 32, kernel_size=3, stride=1, padding=1,
#                                              dilation=1, padding_mode='reflect'),
#             nn.ReLU(),
#             nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0,
#                       dilation=1, padding_mode='reflect'),
#                                    )
#
#     def forward(self, input):
#         conv = input[:,:1,:,:] + 1.*self.conv(input)
#         out = F.sigmoid(conv)
#         return out
# class Lenet(nn.Module):
#     def __init__(self):
#         super(Lenet, self).__init__()
#         self.conv = nn.Sequential(
#             nn.Conv2d(2, 32, kernel_size=3, stride=1, padding=1,
#                       dilation=1, padding_mode='reflect'),
#             nn.LeakyReLU(),
#             nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1,
#                       dilation=1, padding_mode='reflect'),
#             nn.LeakyReLU(),
#             nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1,
#                       dilation=1, padding_mode='reflect'),
#             nn.LeakyReLU(),
#             nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0,
#                       dilation=1, padding_mode='reflect'),
#             # nn.LeakyReLU(),
#                                    )
#
#     def forward(self, x):
#         conv = self.conv(x)
#         return F.sigmoid(conv)

class Renet(nn.Module):
    def __init__(self):
        super(Renet, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.channels = 3
        self.res1 = nn.Sequential(
                                   nn.Conv2d(self.channels, 32, kernel_size=3, stride=1, padding=1,
                                             dilation=1, padding_mode='reflect'),
                                   nn.ReLU(),
                                   nn.Conv2d(32, self.channels, kernel_size=3, stride=1, padding=1,
                                             dilation=1, padding_mode='reflect'),
                                   )
        self.res2 = nn.Sequential(
            nn.Conv2d(self.channels, 32, kernel_size=3, stride=1, padding=1,
                      dilation=1, padding_mode='reflect'),
            nn.ReLU(),
            nn.Conv2d(32, self.channels, kernel_size=3, stride=1, padding=1,
                      dilation=1, padding_mode='reflect'),
        )
        self.res3 = nn.Sequential(
            nn.Conv2d(self.channels, 32, kernel_size=3, stride=1, padding=1,
                      dilation=1, padding_mode='reflect'),
            nn.ReLU(),
            nn.Conv2d(32, self.channels, kernel_size=3, stride=1, padding=1,
                      dilation=1, padding_mode='reflect'),
        )
        self.res4 = nn.Sequential(
            nn.Conv2d(self.channels, 32, kernel_size=3, stride=1, padding=1,
                      dilation=1, padding_mode='reflect'),
            nn.ReLU(),
            nn.Conv2d(32, self.channels, kernel_size=3, stride=1, padding=1,
                      dilation=1, padding_mode='reflect'),
        )

        self.scale = nn.Sequential(
            nn.Linear(1, 100, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(100, 1, bias=True),
            nn.Sigmoid()
        )


    def forward(self, input,a):
        scale = self.scale(a)
        out1 = self.relu(input + 1.* self.res1(input))
        out2 = self.relu(out1 + 1.*self.res2(out1))
        out3 = self.relu(out2 + 1. * self.res3(out2))
        out4 = self.relu(out3 + scale * self.res4(out3))
        return out4



class Anet(nn.Module):
    def __init__(self):
        super(Anet, self).__init__()
        self.relu = nn.ReLU(inplace=True)

        self.scale = nn.Sequential(
            nn.Linear(1, 100, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(100, 1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, a):
        scale = self.scale(a)

        return scale


# class Renet(nn.Module):
#     def __init__(self):
#         super(Renet, self).__init__()
#         self.channels = 4
#
#         def add_conv_stage(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=True):
#             return nn.Sequential(
#                 nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size, stride=stride, padding=padding, padding_mode='reflect'),
#                 nn.ReLU(),
#                 nn.Conv2d(dim_out, dim_out, kernel_size=kernel_size, stride=stride, padding=padding, padding_mode='reflect'),
#                 nn.ReLU()
#             )
#
#         # Up sampling module
#         def upsample(ch_coarse, ch_fine):
#             return nn.Sequential(
#                 nn.ConvTranspose2d(ch_coarse, ch_fine, 4, 2, 1, bias=False),
#                 nn.ReLU()
#             )
#
#         self.conv1 = add_conv_stage(self.channels, 32)
#         self.conv2 = add_conv_stage(32, 64)
#         self.conv3 = add_conv_stage(64, 128)
#         self.conv4 = add_conv_stage(128, 256)
#         self.conv3m = add_conv_stage(256, 128)
#         self.conv2m = add_conv_stage(128, 64)
#         self.conv1m = add_conv_stage(64, 32)
#
#         self.conv0 = nn.Sequential(
#             nn.Conv2d(32, 3, 3, 1, 1),
#             # nn.Sigmoid()
#         )
#
#         self.max_pool = nn.MaxPool2d(2, ceil_mode=True)
#
#
#         self.upsample43 = upsample(256, 128)
#         self.upsample32 = upsample(128, 64)
#         self.upsample21 = upsample(64, 32)
#
#     def forward(self, x):
#         conv1_out = self.conv1(x)
#         conv2_out = self.conv2(self.max_pool(conv1_out))
#         conv3_out = self.conv3(self.max_pool(conv2_out))
#         conv4_out = self.conv4(self.max_pool(conv3_out))
#         conv4m_out_ = torch.cat((self.upsample43(conv4_out), conv3_out), 1)
#         conv3m_out = self.conv3m(conv4m_out_)
#         conv3m_out_ = torch.cat((self.upsample32(conv3m_out), conv2_out), 1)
#         conv2m_out = self.conv2m(conv3m_out_)
#         conv2m_out_ = torch.cat((self.upsample21(conv2m_out), conv1_out), 1)
#         conv1m_out = self.conv1m(conv2m_out_)
#         conv0_out = self.conv0(conv1m_out)
#
#         return conv0_out














