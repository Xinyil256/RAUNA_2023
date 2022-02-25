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

    def forward(self, input_r, input_l, a):
        R = self.renet(input_r)
        l = torch.cat((input_l,a.unsqueeze(dim=1)), dim=1)
        L = self.lenet(l)
        return R, L


class Lenet(nn.Module):
    def __init__(self):
        super(Lenet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, stride=1, padding=2,
                                             dilation=2, padding_mode='reflect'),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1,
                      dilation=1, padding_mode='reflect'),
                                   )

    def forward(self, input):
        out = input[:, :1, :, :] + self.conv(input)
        return F.sigmoid(out)
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
        self.res1 = nn.Sequential(
                                   nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1,
                                             dilation=1, padding_mode='reflect'),
                                   nn.ReLU(inplace=False),
                                   nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1,
                                             dilation=1, padding_mode='reflect'),
                                   )

    def forward(self, input):
        res = self.res1(input)
        out = nn.ReLU()(input+res)
        return out














