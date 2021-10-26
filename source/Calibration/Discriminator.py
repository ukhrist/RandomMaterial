
from numpy.core.fromnumeric import squeeze
import torch
from torch import nn


from ..StatisticalDescriptors import autocorrelation, spec_area, interface, correlation_curvature, curvature, num_curvature
from ..StatisticalDescriptors.common import gradient, interpolate


# class Discriminator(nn.Module):
#     def __init__(self, input_length: int):
#         super(Discriminator, self).__init__()
#         # self.lin1 = nn.Linear(1, 1, bias=True)
#         # self.lin2 = nn.Linear(1, 1, bias=True)
#         self.vf = nn.Parameter(torch.tensor([0.]))
#         self.sa = nn.Parameter(torch.tensor([0.]))

#     def forward(self, X):        
#         vf = X.mean() #.unsqueeze(-1)
#         sa = spec_area(X)  #.unsqueeze(-1)
#         y = (vf/self.vf.exp()-1)**2 + (sa-self.sa.exp())**2
#         y = torch.exp(-y)
#         return y



# class Discriminator(nn.Module):
#     def __init__(self, input_length: int):
#         super(Discriminator, self).__init__()
#         self.kernel_size = 3
#         # self.conv1 = nn.Conv3d(int(input_length), int(input_length/2), self.kernel_size)
#         # self.conv2 = nn.Conv3d(int(input_length/2), int(input_length/4), self.kernel_size)
#         # self.conv3 = nn.Conv3d(int(input_length/4), int(input_length/8), self.kernel_size)
#         self.conv1 = nn.Conv3d(1, 2, 4, 2, 1, bias=False)
#         self.conv2 = nn.Conv3d(2, 2, 4, 2, 1, bias=False)
#         self.conv3 = nn.Conv3d(2, 1, 4, 2, 1, bias=False)
#         # self.assemble = nn.Linear(int(input_length), 1, bias=False)

#         # self.n = 10
#         # self.layer1 = nn.Linear(self.n, self.n, dtype=torch.double)
#         # self.layer2 = nn.Linear(self.n, self.n, dtype=torch.double)
#         # self.layer3 = nn.Linear(self.n, 1, dtype=torch.double)

#         self.actfc1 = nn.LeakyReLU(negative_slope=0.2)
#         self.actfc2 = nn.Sigmoid()
#         self.actfc3 = nn.Softplus()

#     def forward(self, X):
#         # y = autocorrelation(x) #-x.mean()) / x.mean()
#         # z = interface(x)
#         # z = autocorrelation(z) #-z.mean()) / z.mean()
#         # y = torch.stack([y, z])
#         # y = y.unsqueeze(0)
#         # y = torch.fft.rfftn(X, s=[n for n in X.shape]).abs()
#         # y = y.unsqueeze(-1) * torch.arange(self.n)
#         # y = self.actfc3(self.layer1(y))
#         # y = self.actfc3(self.layer2(y))
#         # y = self.actfc3(self.layer3(y))
#         # y = y.squeeze()
#         # y = torch.fft.irfftn(y, s=[n for n in X.shape])
#         y = X.unsqueeze(0).unsqueeze(0)
#         y = self.actfc1(self.conv1(y))
#         y = self.actfc1(self.conv2(y))
#         y = self.actfc1(self.conv3(y))
#         y = self.actfc2(y).mean()
#         return y




# Batch size during training
batch_size = 1

# Spatial size of training images. All images will be resized to this
#   size using a transformer.
image_size = 64

# Number of channels in the training images. For color images this is 3
nc = 1


# Size of feature maps in discriminator
ndf = 1




    
class Discriminator(nn.Module):
    def __init__(self, input_length):
        super(Discriminator, self).__init__()
        # self.main = nn.Sequential(
        #     # input is (nc) x 64 x 64
        #     nn.Conv3d(nc, ndf, 4, 2, 1, bias=True),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     # state size. (ndf) x 32 x 32
        #     nn.Conv3d(ndf, ndf * 2, 4, 2, 1, bias=True),
        #     nn.BatchNorm3d(ndf * 2),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     # state size. (ndf*2) x 16 x 16
        #     nn.Conv3d(ndf * 2, ndf * 4, 4, 2, 1, bias=True),
        #     nn.BatchNorm3d(ndf * 4),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     # state size. (ndf*4) x 8 x 8
        #     nn.Conv3d(ndf * 4, ndf * 8, 4, 2, 1, bias=True),
        #     nn.BatchNorm3d(ndf * 8),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     # state size. (ndf*8) x 4 x 4
        #     nn.Conv3d(ndf * 8, 1, 4, 1, 0, bias=True),
        #     nn.Sigmoid()
        # )

        # self.conv1 = nn.Conv3d(nc, ndf, 4, 2, 0, bias=True)
        # self.conv2 = nn.Conv3d(ndf, 1, 4, 2, 0, bias=True)
        # self.actfc = nn.LeakyReLU(0.2, inplace=True)
        # # self.BN = nn.BatchNorm3d(1)
        # self.actfc_final = nn.Sigmoid()

        self.conv11 = nn.Conv3d(nc, 1, 2, 1, 0, bias=False)
        self.conv12 = nn.Conv3d(nc, 1, 3, 1, 0, bias=False)
        self.conv13 = nn.Conv3d(nc, 1, 4, 1, 0, bias=False)

        self.lin = nn.Linear(3, 1, bias=False)

        self.bias = nn.Parameter(torch.zeros([3], dtype=float))

        # self.conv21 = nn.Conv3d(nc, 1, 2, 0, 0, bias=False)
        # self.conv22 = nn.Conv3d(nc, 1, 3, 0, 0, bias=False)
        # self.conv23 = nn.Conv3d(nc, 1, 4, 0, 0, bias=False)

        # self.actfc = nn.LeakyReLU(0.2, inplace=True)
        # self.BN = nn.BatchNorm3d(1)
        self.actfc_final = nn.Sigmoid()

    def forward(self, X):
        y1 = self.conv11(X.unsqueeze(0).unsqueeze(0)).mean()
        y2 = self.conv12(X.unsqueeze(0).unsqueeze(0)).mean()
        y3 = self.conv13(X.unsqueeze(0).unsqueeze(0)).mean()
        y = torch.stack([y1, y2, y3]).squeeze() - self.bias
        y = self.lin(y**2)
        # y = self.actfc_final(y)
        y = torch.exp(-y**2)
        return y