import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, bias=True, norm=False, relu=True, transpose=False):
        super(BasicConv, self).__init__()
        if bias and norm:
            bias = False

        padding = kernel_size // 2
        layers = list()
        if transpose:
            padding = kernel_size // 2 -1
            layers.append(nn.ConvTranspose2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        else:
            layers.append(
                nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        if norm:
            layers.append(nn.BatchNorm2d(out_channel))
        if relu:
            layers.append(nn.LeakyReLU())
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)

class MultiAttn(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.norm = nn.BatchNorm2d(dim)
        # Simple Channel Attention
        self.Wv = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            nn.Conv2d(dim, dim, kernel_size=3, padding=3 // 2, groups=dim, padding_mode='reflect')
        )
        self.Wg = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim, 1),
            nn.Sigmoid()
        )

        # Channel Attention
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim, 1, padding=0, bias=True),
            nn.LeakyReLU(),
            # nn.ReLU(True),
            nn.Conv2d(dim, dim, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

        # Pixel Attention
        self.pa = nn.Sequential(
            nn.Conv2d(dim, dim // 8, 1, padding=0, bias=True),
            nn.LeakyReLU(),
            # nn.ReLU(True),
            nn.Conv2d(dim // 8, 1, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

        self.mlp = nn.Sequential(
            nn.Conv2d(dim * 3, dim * 4, 1),
            nn.LeakyReLU(),
            # nn.ReLU(True),
            nn.Conv2d(dim * 4, dim, 1)
        )

    def forward(self, x):
        identity = x
        x = self.norm(x)
        x = torch.cat([self.Wv(x) * self.Wg(x), self.ca(x) * x, self.pa(x) * x], dim=1)
        x = self.mlp(x)
        x = identity + x
        return x

class FusionIn(nn.Module):
    def __init__(self, channel):
        super(FusionIn, self).__init__()
        self.merge = BasicConv(channel*2, channel, kernel_size=3, stride=1, relu=False)

    def forward(self, x1, x2):
        return self.merge(torch.cat([x1, x2], dim=1))
    
class ConvIn(nn.Module):
    def __init__(self, out_plane):
        super(ConvIn, self).__init__()
        self.main = nn.Sequential(
            BasicConv(3, out_plane//4, kernel_size=3, stride=1, relu=True),
            BasicConv(out_plane // 4, out_plane // 2, kernel_size=1, stride=1, relu=True),
            BasicConv(out_plane // 2, out_plane // 2, kernel_size=3, stride=1, relu=True),
            BasicConv(out_plane // 2, out_plane, kernel_size=1, stride=1, relu=False),
            nn.InstanceNorm2d(out_plane, affine=True)
        )

    def forward(self, x):
        x = self.main(x)
        return x
           
class MultiscaleConv(nn.Module):
    def __init__(self, in_channels,outchannel,dilation=3, res=True,group=False):
        super(MultiscaleConv, self).__init__()
        self.res = res
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels,outchannel,kernel_size=3,dilation=1,padding=1),
            nn.PReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels,outchannel,kernel_size=3,dilation=3,padding=3),
            nn.PReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels,outchannel,kernel_size=3,dilation=5,padding=5),
            nn.PReLU(),
        )
        self.merge = nn.Sequential(
            nn.Conv2d(outchannel*4,outchannel*2,kernel_size=1),
            nn.GELU(),
            nn.Conv2d(outchannel*2,outchannel,kernel_size=1),
        )
    def forward(self,x):
        x1 = self.conv1(x) + x
        x2 = self.conv2(x1) + x1
        x3 = self.conv3(x2) + x2
        out = torch.cat([x,x1,x2,x3],dim=1)
        out = self.merge(out)
        return x+out

class ConvBlock_In(nn.Module):
    def __init__(self, in_channels,outchannel,dilation=3, res=True,group=False):
        super(ConvBlock_In, self).__init__()
        self.res = res
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels,outchannel,kernel_size=3,dilation=4,padding=4),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels,outchannel,kernel_size=3,padding=1),
        )
        self.merge = nn.Sequential(
            nn.Conv2d(outchannel,outchannel*2,kernel_size=1),
            nn.GELU(),
            nn.Conv2d(outchannel*2,outchannel,kernel_size=1),
        )
    def forward(self,x):
        out = self.conv1(x) + self.conv2(x)
        out = self.merge(out)
        return x + out if self.res else out
    
class DynamicConv(nn.Module):
    def __init__(self, inchannels, dilation=0, kernel_size=3, stride=1):
        super(DynamicConv, self).__init__()
        self.stride = stride
        self.kernel_size = kernel_size
        self.kernelNumber = inchannels
        self.conv = nn.Conv2d(inchannels, self.kernelNumber*kernel_size**2, kernel_size=1, stride=1, bias=False)
        self.bn = nn.BatchNorm2d(self.kernelNumber*kernel_size**2)
        self.act = nn.Softmax(dim=-2)
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')
        self.ap = nn.AdaptiveAvgPool2d((1, 1))
        self.unfoldMask = []
        self.unfoldSize = kernel_size + dilation * (kernel_size - 1)
        self.pad = nn.ReflectionPad2d(self.unfoldSize//2)
        for i in range(self.unfoldSize):
            for j in range(self.unfoldSize):
                if (i % (dilation + 1) == 0) and (j % (dilation + 1) == 0):
                    self.unfoldMask.append(i * self.unfoldSize + j)

        
    def forward(self, x):
        copy = x
        filter = self.ap(x)
        filter = self.conv(filter)
        filter = self.bn(filter)
        n, c, h, w = x.shape
        x = F.unfold(self.pad(x), kernel_size=self.unfoldSize).reshape(n, self.kernelNumber, c//self.kernelNumber, self.unfoldSize**2, h*w)
        n,c1,p,q = filter.shape
        filter = filter.reshape(n, c1//self.kernel_size**2, self.kernel_size**2, p*q).unsqueeze(2)
        filter = self.act(filter)
        out = torch.sum(x * filter, dim=3).reshape(n, c, h, w)
        return out, copy - out
    
class BranchAttn(nn.Module):  
    def __init__(self, in_channels):  
        super(BranchAttn, self).__init__() 
        self.conv1 = nn.ModuleList([
            DynamicCA(in_channels) for i in range(8)
        ])

    def forward(self,x0,x1,x2,x3,x4,x5,x6,x7):
        (x0,x1,x2,x3,x4,x5,x6,x7) =  (self.conv1[0](x0,x1,x2,x3,x4,x5,x6,x7)+x0,self.conv1[1](x1,x0,x2,x3,x4,x5,x6,x7)+x1,
                                      self.conv1[2](x2,x0,x1,x3,x4,x5,x6,x7)+x2,self.conv1[3](x3,x0,x1,x2,x4,x5,x6,x7)+x3,
                                      self.conv1[4](x4,x0,x1,x2,x3,x5,x6,x7)+x4,self.conv1[5](x5,x0,x1,x2,x3,x4,x6,x7)+x5,
                                      self.conv1[6](x6,x0,x1,x2,x3,x4,x5,x7)+x6,self.conv1[7](x7,x0,x1,x2,x3,x4,x5,x6)+x7
                                      )

        out = torch.cat([x0,x1,x2,x3,x4,x5,x6,x7],dim=1)
        return out

class DynamicCA(nn.Module):
    def __init__(self,inchannels):
        super(DynamicCA, self).__init__()
        self.inchannels=inchannels
        self.fc = nn.Linear(inchannels*8, inchannels*8*inchannels)
        self.conv1d = nn.Conv1d(1, 1, kernel_size=3, padding=1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, y1, y2, y3, y4, y5, y6, y7):
        m = torch.cat((x, y1, y2, y3, y4, y5, y6, y7), dim=1)  # n, 32, h, w
        gap = F.adaptive_avg_pool2d(m, (1, 1))  # n, 32, 1, 1
        gap = gap.view(m.size(0), self.inchannels*8)  # n, 32
        fc_out = self.fc(gap)  # n, 128
        conv1d_input = fc_out.unsqueeze(1)  # n, 1, 128
        conv1d_out = self.conv1d(conv1d_input)  # n, 1, 128
        conv1d_out = conv1d_out.view(m.size(0), self.inchannels*8, self.inchannels)  # n, 32, 4
        softmax_out = self.softmax(conv1d_out)  # n, 32, 4
        out = torch.einsum('nchw,ncm->nmhw', (m, softmax_out))  # n, 4, h, w
        
        return out

class DynamicSplit(nn.Module):
    def __init__(self, in_channels,outchannel,basechannel):
        super(DynamicSplit, self).__init__()
        self.dynamic_filter = DynamicConv(in_channels)
        self.out_block_1 = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels,outchannel,kernel_size=3,padding=1)
        )
        self.out_block_2 = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels,basechannel//8,kernel_size=1),
            MultiscaleConv(basechannel//8,basechannel//8,res=True),
            MultiscaleConv(basechannel//8,basechannel//8,res=True),
        )
    def forward(self,x):
        low,high = self.dynamic_filter(x)
        low = self.out_block_2(low)
        high = self.out_block_1(high)
        return high,low

class NetBlock(nn.Module):  
    def __init__(self, in_channels):  
        super(NetBlock, self).__init__()  

        self.frequency_enlarge = ConvBlock_In(in_channels,in_channels,res=True)

        self.blocks = nn.ModuleList([
            DynamicSplit(in_channels,in_channels*7//8,in_channels),
            DynamicSplit(in_channels*7//8,in_channels*6//8,in_channels),
            DynamicSplit(in_channels*6//8,in_channels*5//8,in_channels),
            DynamicSplit(in_channels*5//8,in_channels*4//8,in_channels),
            DynamicSplit(in_channels*4//8,in_channels*3//8,in_channels),
            DynamicSplit(in_channels*3//8,in_channels*2//8,in_channels),
            DynamicSplit(in_channels*2//8,in_channels*1//8,in_channels),
        ])

        self.local = BranchAttn(in_channels//8)

        self.synthesizer = nn.Sequential(
            MultiAttn(in_channels),
        )
        self.merger = nn.Sequential(
            nn.Conv2d(in_channels,in_channels*2,kernel_size=1),
            nn.GELU(),
            nn.Conv2d(in_channels*2,in_channels,kernel_size=1),
        )
        self.merger2 = nn.Sequential(
            nn.Conv2d(in_channels,in_channels*2,kernel_size=1),
            nn.GELU(),
            nn.Conv2d(in_channels*2,in_channels,kernel_size=1),
        )
          
    def forward(self, m):  
        m0 = self.frequency_enlarge(m)

        m1,x0 = self.blocks[0](m0)
        m1,x1 = self.blocks[1](m1)
        m1,x2 = self.blocks[2](m1)
        m1,x3 = self.blocks[3](m1)
        m1,x4 = self.blocks[4](m1)
        m1,x5 = self.blocks[5](m1)
        x7,x6 = self.blocks[6](m1)

        m2 = self.local(x0,x1,x2,x3,x4,x5,x6,x7)
        m2 = self.merger(m2)
        m2 = m2 + m0
        out = self.synthesizer(m2)
        out = self.merger2(out)
        out = out + m2
        return out

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        base_channel = 32

        self.Encoder = nn.ModuleList([
            NetBlock(base_channel),
            NetBlock(base_channel*2),
            NetBlock(base_channel*4),
        ])

        self.feat_extract = nn.ModuleList([
            BasicConv(3, base_channel, kernel_size=3, relu=True, stride=1),
            BasicConv(base_channel, base_channel*2, kernel_size=3, relu=True, stride=2),
            BasicConv(base_channel*2, base_channel*4, kernel_size=3, relu=True, stride=2),
            BasicConv(base_channel*4, base_channel*2, kernel_size=4, relu=True, stride=2, transpose=True),
            BasicConv(base_channel*2, base_channel, kernel_size=4, relu=True, stride=2, transpose=True),
            BasicConv(base_channel, 3, kernel_size=3, relu=False, stride=1)
        ])

        self.Decoder = nn.ModuleList([
            NetBlock(base_channel * 4),
            NetBlock(base_channel * 2),
            NetBlock(base_channel)
        ])

        self.conv_res = nn.ModuleList([
            BasicConv(base_channel * 4, base_channel * 2, kernel_size=1, relu=True, stride=1),
            BasicConv(base_channel * 2, base_channel, kernel_size=1, relu=True, stride=1),
        ])

        self.conv_out = nn.ModuleList(
            [
                BasicConv(base_channel * 4, 3, kernel_size=3, relu=False, stride=1),
                BasicConv(base_channel * 2, 3, kernel_size=3, relu=False, stride=1),
            ]
        )

        self.fusion_in_1 = FusionIn(base_channel * 4)
        self.conv_in_1 = ConvIn(base_channel * 4)
        self.fusin_in_2 = FusionIn(base_channel * 2)
        self.conv_in_2 = ConvIn(base_channel * 2)

    def forward(self, x):
        x_2 = F.interpolate(x, scale_factor=0.5)
        x_4 = F.interpolate(x_2, scale_factor=0.5)
        z2 = self.conv_in_2(x_2)
        z4 = self.conv_in_1(x_4)

        outputs = list()

        x_ = self.feat_extract[0](x)
        res1 = self.Encoder[0](x_)

        z = self.feat_extract[1](res1)
        z = self.fusin_in_2(z, z2)
        res2 = self.Encoder[1](z)

        z = self.feat_extract[2](res2)
        z = self.fusion_in_1(z, z4)
        z = self.Encoder[2](z)

        z = self.Decoder[0](z)
        z_ = self.conv_out[0](z)

        z = self.feat_extract[3](z)
        outputs.append(z_+x_4)

        z = torch.cat([z, res2], dim=1)
        z = self.conv_res[0](z)
        z = self.Decoder[1](z)
        z_ = self.conv_out[1](z)
        
        z = self.feat_extract[4](z)
        outputs.append(z_+x_2)

        z = torch.cat([z, res1], dim=1)
        z = self.conv_res[1](z)
        z = self.Decoder[2](z)
        z = self.feat_extract[5](z)
        outputs.append(z+x)

        return outputs
    
def build_net():
    return Model()

