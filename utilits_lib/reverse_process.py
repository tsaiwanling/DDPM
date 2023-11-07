import torch.nn as nn
import torch

'''
Ref: https://hackmd.io/@Tu32/Bkq3fQi0s
'''

class DoubleConv(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(int(in_c), int(out_c), kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(1, int(out_c)), #equivalent with LayerNorm
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(int(out_c), int(out_c), kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(1, int(out_c)), #equivalent with LayerNorm
            nn.ReLU(True)
        )
  
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class Down(nn.Module):
    def __init__(self, in_c, out_c, emb_dim=128):
        super().__init__()
        self.down = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            DoubleConv(in_c,out_c),
        )

        self.emb_layer = nn.Sequential(
            nn.ReLU(),
            nn.Linear(emb_dim, int(out_c)),
        )

    def forward(self, x, t):
        x = self.down(x)
        #擴充兩個dimension，然後使用repeat填滿成和圖片相同(如同numpy.tile)
        t_emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1]) 
        return x + t_emb

class Up(nn.Module):
    def __init__(self, in_c, out_c, emb_dim=128):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = DoubleConv(in_c,out_c)
        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, int(out_c)),
        )

    def forward(self, x, skip_x, t):
        x = self.up(x)
        x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb

class SelfAttention(nn.Module):
    def __init__(self, channels, size):
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.size = size
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        x = x.view(-1, self.channels, self.size * self.size).swapaxes(1, 2)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.swapaxes(2, 1).view(-1, self.channels, self.size, self.size)
    
class UNet(nn.Module):
    def __init__(self, c_in=3, c_out=3, time_dim=128, img_size=64, device="cuda"):
        super().__init__()
        self.device = device
        self.time_dim = time_dim

        self.inc = DoubleConv(c_in, 64) #(b,3,64,64) -> (b,64,64,64)

        self.down1 = Down(64, 128, emb_dim = time_dim) #(b,64,64,64) -> (b,128,32,32)
        self.sa1 = SelfAttention(128, img_size//2) #(b,128,32,32) -> (b,128,32,32)
        self.down2 = Down(128, 256, emb_dim = time_dim) #(b,128,32,32) -> (b,256,16,16)
        self.sa2 = SelfAttention(256, img_size//4) #(b,256,16,16) -> (b,256,16,16)
        self.down3 = Down(256, 256, emb_dim = time_dim) #(b,256,16,16) -> (b,256,8,8)
        self.sa3 = SelfAttention(256, img_size//8) #(b,256,8,8) -> (b,256,8,8)

        self.bot1 = DoubleConv(256, 512) #(b,256,8,8) -> (b,512,8,8)
        self.bot2 = DoubleConv(512, 512) #(b,512,8,8) -> (b,512,8,8)
        self.bot3 = DoubleConv(512, 256) #(b,512,8,8) -> (b,256,8,8)

        self.up1 = Up(512, 128, emb_dim = time_dim) #(b,512,8,8) -> (b,128,16,16) because the skip_x
        self.sa4 = SelfAttention(128, img_size//4) #(b,128,16,16) -> (b,128,16,16)
        self.up2 = Up(256, 64, emb_dim = time_dim) #(b,256,16,16) -> (b,64,32,32)
        self.sa5 = SelfAttention(64, img_size//2) #(b,64,32,32) -> (b,64,32,32)
        self.up3 = Up(128, 64, emb_dim = time_dim) #(b,128,32,32) -> (b,64,64,64)
        self.sa6 = SelfAttention(64, img_size) #(b,64,64,64) -> (b,64,64,64)

        self.outc = nn.Conv2d(64, c_out, kernel_size=1) #(b,64,64,64) -> (b,3,64,64)

    def pos_encoding(self, t, channels):
        # t = torch.tensor([t])
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq.to(self.device))
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq.to(self.device))
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def forward(self, x, t):
        t = self.pos_encoding(t, self.time_dim)
        #initial conv
        x1 = self.inc(x)
        
        #Down
        x2 = self.down1(x1, t)
        x2 = self.sa1(x2)
        x3 = self.down2(x2, t)
        x3 = self.sa2(x3)
        x4 = self.down3(x3, t)
        x4 = self.sa3(x4)

        #Bottle neck
        x4 = self.bot1(x4)
        x4 = self.bot2(x4)
        x4 = self.bot3(x4)

        #Up
        x = self.up1(x4, x3, t)
        x = self.sa4(x)
        x = self.up2(x, x2, t)
        x = self.sa5(x)
        x = self.up3(x, x1, t)
        x = self.sa6(x)

        #Output
        output = self.outc(x)
        return output