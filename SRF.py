
import torch
import torch.nn.functional as F
import torch.nn as nn 


class SFR(nn.Module):
    '''
    alpha: 0<alpha<1
    '''
    def __init__(self, 
                 op_channel:int,
                 alpha:float = 1/2,
                 squeeze_radio:int = 2 ,
                 group_size:int = 2,
                 group_kernel_size: int = 3,
                 ):
        super().__init__()
        self.up_channel     = up_channel   =   int(alpha*op_channel)
        self.low_channel    = low_channel  =   op_channel-up_channel

        self.squeeze1 = nn.Conv2d(up_channel, up_channel // squeeze_radio, kernel_size=1, bias=False)
        self.squeeze2 = nn.Conv2d(low_channel, low_channel // squeeze_radio, kernel_size=1, bias=False)
        self.squeeze3 = nn.Conv2d(up_channel, up_channel // squeeze_radio, kernel_size=1, bias=False)
        self.squeeze4 = nn.Conv2d(low_channel, low_channel // squeeze_radio, kernel_size=1, bias=False)

        # up
        self.GWCr = nn.Conv2d(up_channel // squeeze_radio, op_channel, kernel_size=group_kernel_size, stride=1,
                              padding=group_kernel_size // 2, groups=group_size)
        self.PWCr1 = nn.Conv2d(up_channel // squeeze_radio, op_channel, kernel_size=1, bias=False)
        # low
        self.PWCr2 = nn.Conv2d(low_channel // squeeze_radio, op_channel - low_channel // squeeze_radio, kernel_size=1,
                              bias=False)

        # up
        self.GWCd = nn.Conv2d(up_channel // squeeze_radio, op_channel, kernel_size=group_kernel_size, stride=1,
                              padding=group_kernel_size // 2, groups=group_size)
        self.PWCd1 = nn.Conv2d(up_channel // squeeze_radio, op_channel, kernel_size=1, bias=False)
        # low
        self.PWCd2 = nn.Conv2d(low_channel // squeeze_radio, op_channel - low_channel // squeeze_radio, kernel_size=1,
                              bias=False)

        self.advavg1 = nn.AdaptiveAvgPool2d(1)
        self.advavg2 = nn.AdaptiveAvgPool2d(1)


        self.param = torch.nn.Parameter(torch.rand([1, op_channel, 1, 1]), requires_grad=True)
        self.out = nn.Sequential(
            nn.Conv2d(op_channel, op_channel, 3, 1, 1),
            nn.BatchNorm2d(op_channel),
            nn.ReLU()
        )

    def forward(self, r, d):
        # Split
        r_up, r_low = torch.split(r, [self.up_channel, self.low_channel],dim=1)

        d_up, d_low = torch.split(d, [self.up_channel, self.low_channel], dim=1)

        r_up, r_low = self.squeeze1(r_up), self.squeeze2(r_low)
        d_up, d_low = self.squeeze3(d_up), self.squeeze4(d_low)


        # Transform
        rY1      = self.GWCr(r_up) + self.PWCr1(r_up)
        rY2      = torch.cat( [self.PWCr2(r_low), r_low], dim= 1 )

        dY1 = self.GWCd(d_up) + self.PWCd1(d_up)
        dY2 = torch.cat([self.PWCd2(d_low), d_low], dim=1)


        # Fuse
        out1     = torch.cat( [rY1,dY1], dim= 1 )
        out1     = F.softmax( self.advavg1(out1), dim=1 ) * out1 + out1

        out1a, out1b = torch.split(out1, out1.size(1)//2, dim=1)
        out1 = out1a + out1b
        # print(out1.shape)

        out2 = torch.cat([rY2, dY2], dim=1)
        out2 = F.softmax(self.advavg1(out2), dim=1) * out2 + out2


        out2a, out2b = torch.split(out2, out2.size(1) // 2, dim=1)
        out2 = out2a + out2b
        # print(out2.shape)

        p = torch.sigmoid(self.param)
        out_last = out1 * p + out2 * (1 - p)
        out = self.out(out_last)

        return out




if __name__ == '__main__':
    x       = torch.randn(1,32,16,16)
    y = torch.randn(1, 32, 16, 16)
    model   = SFR(op_channel=32)
    fusion = model(x,y)
    print(fusion.shape)
