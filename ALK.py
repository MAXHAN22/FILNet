import torch
import torch.nn as nn


class ALK(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv1 = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        self.conv2 = nn.Conv2d(dim, dim, 9, stride=1, padding=((9 // 2) * 4), groups=dim, dilation=4)
        self.conv_reduce1 = nn.Conv2d(dim, dim//2, 1)
        self.conv_reduce2 = nn.Conv2d(dim, dim//2, 1)
        self.conv_reduce3 = nn.Conv2d(dim, dim // 2, 1)
        self.conv_squeeze = nn.Conv2d(2, 3, 7, padding=3)
        self.conv = nn.Conv2d(dim//2, dim, 1)

    def forward(self, x):   
        attn1 = self.conv0(x)
        attn2 = self.conv1(attn1)
        attn3 = self.conv2(attn2)

        attn1 = self.conv_reduce1(attn1)
        attn2 = self.conv_reduce2(attn2)
        attn3 = self.conv_reduce3(attn3)
        
        attn = torch.cat([attn1, attn2, attn3], dim=1)
        # print(attn. shape)
        avg_attn = torch.mean(attn, dim=1, keepdim=True)
        max_attn, _ = torch.max(attn, dim=1, keepdim=True)
        # print(max_attn.shape)
        agg = torch.cat([avg_attn, max_attn], dim=1)
        # print(agg.shpe)
        sig = self.conv_squeeze(agg).sigmoid()
        attn = attn1 * sig[:,0,:,:].unsqueeze(1) + attn2 * sig[:,1,:,:].unsqueeze(1) + attn3 * sig[:,2,:,:].unsqueeze(1)
        attn = self.conv(attn)
        return x * attn + x




