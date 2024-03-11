import torch

from torch import nn
import torch.nn.functional as F

from timm.models.layers import DropPath, to_2tuple, trunc_normal_

def upsample(src, tar):
    src = F.interpolate(src, size=tar.shape[2:], mode='bilinear', align_corners=True)
    return src
class Conv1(nn.Module):
    def __init__(self,  in_channel, out_channel):
        super(Conv1, self).__init__()

        self.conv=nn.Conv2d(in_channel,out_channel,kernel_size=1,padding=0,stride=1)
        self.bn=nn.BatchNorm2d(out_channel)
        self.relu=nn.LeakyReLU(inplace=True)

    def forward(self, input):

        out=self.conv(input)
        out=self.bn(out)
        out=self.relu(out)
        return out
class Convlutioanl(nn.Module):
    def __init__(self,  in_channel, out_channel):
        super(Convlutioanl, self).__init__()
        self.padding=(1,1,1,1)
        self.conv=nn.Conv2d(in_channel,out_channel,kernel_size=3,padding=0,stride=1)
        self.bn=nn.BatchNorm2d(out_channel)
        self.relu=nn.LeakyReLU(inplace=True)

    def forward(self, input):
        out=F.pad(input,self.padding,'replicate')
        out=self.conv(out)
        out=self.bn(out)
        out=self.relu(out)
        return out

class Conv5_5(nn.Module):
    def __init__(self,  in_channel, out_channel):
        super(Conv5_5, self).__init__()
        self.padding=(2,2,2,2)
        self.conv=nn.Conv2d(in_channel,out_channel,kernel_size=5,padding=0,stride=1)
        self.bn=nn.BatchNorm2d(out_channel)
        self.relu=nn.LeakyReLU(inplace=True)

    def forward(self, input):
        out=F.pad(input,self.padding,'replicate')
        out=self.conv(out)
        out=self.bn(out)
        out=self.relu(out)
        return out

class Conv7_7(nn.Module):
    def __init__(self,  in_channel, out_channel):
        super(Conv7_7, self).__init__()
        self.padding=(3,3,3,3)
        self.conv=nn.Conv2d(in_channel,out_channel,kernel_size=7,padding=0,stride=1)
        self.bn=nn.BatchNorm2d(out_channel)
        self.relu=nn.LeakyReLU(inplace=True)

    def forward(self, input):
        out=F.pad(input,self.padding,'replicate')
        out=self.conv(out)
        out=self.bn(out)
        out=self.relu(out)
        return out

class Convlutioanl_out(nn.Module):
    def __init__(self,  in_channel, out_channel):
        super(Convlutioanl_out, self).__init__()
        # self.padding=(2,2,2,2)
        self.conv=nn.Conv2d(in_channel,out_channel,kernel_size=1,padding=0,stride=1)
        # self.bn=nn.BatchNorm2d(out_channel)
        self.sigmoid=nn.Sigmoid()

    def forward(self, input):
        # out=F.pad(input,self.padding,'replicate')
        out=self.conv(input)
        # out=self.bn(out)
        out=self.sigmoid(out)
        return out

class WindowAttention(nn.Module):


    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)

        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):

        B_, N, C = x.shape

        A=self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads)
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):

        flops = 0

        flops += N * self.dim * 3 * self.dim

        flops += self.num_heads * N * (self.dim // self.num_heads) * N

        flops += self.num_heads * N * N * (self.dim // self.num_heads)

        flops += N * self.dim * self.dim
        return flops


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x



def window_partition(x, window_size):

    B, H, W, C = x.shape

    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):

    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class SwinTransformerBlock(nn.Module):


    def __init__(self, dim, input_resolution, num_heads, window_size=1, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:

            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            attn_mask = self.calculate_mask(self.input_resolution)
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def calculate_mask(self, x_size):

        H, W = x_size
        img_mask = torch.zeros((1, H, W, 1))
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        return attn_mask

    def forward(self, x, x_size):
        H, W = x_size

        B,C,H,W= x.shape

        x=x.view(B,H,W,C)
        shortcut = x
        shape=x.view(H*W*B,C)
        x = self.norm1(shape)
        x = x.view(B, H, W, C)

        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
        if self.input_resolution == x_size:
            attn_windows = self.attn(x_windows, mask=self.attn_mask)
        else:
            attn_windows = self.attn(x_windows, mask=self.calculate_mask(x_size).to(x.device))

        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x


        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        B,H,W,C=x.shape
        x=x.view(B,C,H,W)


        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution

        flops += self.dim * H * W

        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)

        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio

        flops += self.dim * H * W
        return flops



class PatchEmbed(nn.Module):


    def __init__(self, img_size=120, patch_size=4, in_chans=6, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        flops = 0
        H, W = self.img_size
        if self.norm is not None:
            flops += H * W * self.embed_dim
        return flops

class BasicLayer(nn.Module):


    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint


        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer)
            for i in range(depth)])

        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, x_size):
        for blk in self.blocks:

            x = blk(x, x_size)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops

class Channel_attention(nn.Module):
    def __init__(self,  channel, reduction):
        super(Channel_attention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc=nn.Sequential(
            nn.Conv2d(channel,channel//reduction,1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel//reduction,channel,1))
        self.sigmoid=nn.Sigmoid()
    def forward(self, input):
        out=self.avg_pool(input)
        out=self.fc(out)
        out=self.sigmoid(out)
        return out

class MODEL(nn.Module):
    def __init__(self, in_channel=1,out_channel_16=16,out_channel_256=256, out_channel_32=32,out_channel=64,out_channel_512=512,output_channel=1,out_channel_128=128,
                 img_size=120, patch_size=4, embed_dim=96, num_heads=8, window_size=1,out_channel_448=448,out_channel_896=896,out_channel_336=336,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., patch_norm=True, depth=2,
                 downsample=None,
                 drop_path=0., norm_layer=nn.LayerNorm, use_checkpoint=False
                ):
        super(MODEL, self).__init__()

        self.convInput= Convlutioanl(in_channel, out_channel_16)
        self.conv5 = Conv5_5(out_channel_16,  out_channel)
        self.conv7 = Conv7_7( out_channel, out_channel_256)
        self.conv64 = Convlutioanl(out_channel, out_channel_256)
        self.conv = Convlutioanl(out_channel*2, out_channel)
        self.convolutional_out =Convlutioanl_out(out_channel_32, output_channel)
        self.conv16_16= Convlutioanl(out_channel_16, out_channel_16)
        self.conv64_16 = Convlutioanl(out_channel, out_channel_16)
        self.conv256_16 = Convlutioanl(out_channel_256, out_channel_16)

        self.cam1 = Channel_attention(out_channel_16,4)
        self.cam2 = Channel_attention(out_channel,8)
        self.cam3 = Channel_attention(out_channel_256,16)


        self.conv1=Conv1( out_channel_32, out_channel_16)

        self.conv2=Conv1( out_channel_128, out_channel)
        self.conv3 = Conv1(out_channel_512, out_channel_256)
        self.conv4 = Conv1(out_channel_336, out_channel_16)
        self.patch_norm = patch_norm

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution
        self.basicLayer1 = BasicLayer(dim=out_channel_16,
                                     input_resolution=(patches_resolution[0], patches_resolution[1]),
                                     depth=depth,
                                     num_heads=num_heads,
                                     window_size=window_size,
                                     mlp_ratio=mlp_ratio,
                                     qkv_bias=qkv_bias, qk_scale=qk_scale,
                                     drop=drop, attn_drop=attn_drop,
                                     drop_path=drop_path,
                                     norm_layer=norm_layer,
                                     downsample=downsample,
                                     use_checkpoint=use_checkpoint)

        self.basicLayer2 = BasicLayer(dim=out_channel,
                                      input_resolution=(patches_resolution[0], patches_resolution[1]),
                                      depth=depth,
                                      num_heads=num_heads,
                                      window_size=window_size,
                                      mlp_ratio=mlp_ratio,
                                      qkv_bias=qkv_bias, qk_scale=qk_scale,
                                      drop=drop, attn_drop=attn_drop,
                                      drop_path=drop_path,
                                      norm_layer=norm_layer,
                                      downsample=downsample,
                                      use_checkpoint=use_checkpoint)

        self.basicLayer3 = BasicLayer(dim=out_channel_256,
                                      input_resolution=(patches_resolution[0], patches_resolution[1]),
                                      depth=depth,
                                      num_heads=num_heads,
                                      window_size=window_size,
                                      mlp_ratio=mlp_ratio,
                                      qkv_bias=qkv_bias, qk_scale=qk_scale,
                                      drop=drop, attn_drop=attn_drop,
                                      drop_path=drop_path,
                                      norm_layer=norm_layer,
                                      downsample=downsample,
                                      use_checkpoint=use_checkpoint)

        self.basicLayer4 = BasicLayer(dim=out_channel_448,
                                      input_resolution=(patches_resolution[0], patches_resolution[1]),
                                      depth=depth,
                                      num_heads=num_heads,
                                      window_size=window_size,
                                      mlp_ratio=mlp_ratio,
                                      qkv_bias=qkv_bias, qk_scale=qk_scale,
                                      drop=drop, attn_drop=attn_drop,
                                      drop_path=drop_path,
                                      norm_layer=norm_layer,
                                      downsample=downsample,
                                      use_checkpoint=use_checkpoint)

    def forward(self, ir,vi):

        convInput_A1 = self.convInput(ir)
        convInput_A2 = self.conv5(convInput_A1 )
        convInput_A3 = self.conv7(convInput_A2)



        convInput_B1 = self.convInput(vi)
        convInput_B2 = self.conv5(convInput_B1)
        convInput_B3 = self.conv7(convInput_B2)

        camA1=self.cam1(convInput_A1)
        camA1_1 = torch.cat((camA1*convInput_A1, convInput_B1),1)
        convA1=self.conv1( camA1_1)
        camA1_2=self.cam1(convA1)* convA1

        encode_sizeA1 = ( camA1_2.shape[2],  camA1_2.shape[3])
        TransformerA1 = self.basicLayer1( camA1_2, encode_sizeA1)

        camB1 = self.cam1(convInput_B1)
        camB1_1 = torch.cat((camB1 * convInput_B1, convInput_A1), 1)
        convB1 = self.conv1(camB1_1)
        camB1_2 = self.cam1(convB1) * convB1

        encode_sizeB1 = (camB1_2.shape[2], camB1_2.shape[3])
        TransformerB1 = self.basicLayer1(camB1_2, encode_sizeB1)

        camA2= self.cam2(convInput_A2)

        camA2_1 = torch.cat((camA2 * convInput_A2, convInput_B2), 1)
        convA2 = self.conv2(camA2_1)
        camA2_2 = self.cam2(convA2) * convA2

        encode_sizeA2 = (camA2_2.shape[2], camA2_2.shape[3])
        TransformerA2 = self.basicLayer2(camA2_2, encode_sizeA2)

        camB2= self.cam2(convInput_B2)
        camB2_1 = torch.cat((camB2* convInput_B2, convInput_A2), 1)
        convB2= self.conv2(camB2_1)
        camB2_2 = self.cam2(convB2) * convB2

        encode_sizeB2= (camB2_2.shape[2], camB2_2.shape[3])
        TransformerB2 = self.basicLayer2(camB2_2, encode_sizeB2)

        camA3=self.cam3(convInput_A3)
        camA3_1 = torch.cat(( camA3* convInput_A3, convInput_B3), 1)
        convA3 = self.conv3(camA3_1)
        camA3_2 = self.cam3(convA3) * convA3

        encode_sizeA3 = (camA3_2.shape[2], camA3_2.shape[3])
        TransformerA3 = self.basicLayer3(camA3_2, encode_sizeA3)

        camB3=self.cam3(convInput_B3)
        camB3_1 = torch.cat((camB3 * convInput_B3, convInput_A3), 1)
        convB3 = self.conv3(camB3_1)
        camB3_2 = self.cam3(convB3) * convB3

        encode_sizeB3 = (camB3_2.shape[2], camB3_2.shape[3])
        TransformerB3 = self.basicLayer3(camB3_2, encode_sizeB3)


        catA=torch.cat(( TransformerA1, TransformerA2, TransformerA3),1)
        convA4=self.conv4( catA)

        catB = torch.cat((TransformerB1, TransformerB2, TransformerB3), 1)

        convB4 = self.conv4(catB)

        camA4=self.cam1( convA4)
        camA4_1 = torch.cat((camA4 * convA4, convB4 ), 1)
        convA5 = self.conv1(camA4_1)
        camA4_2 = self.cam1(convA5) * convA5

        encode_sizeA4 = (camA4_2.shape[2], camA4_2.shape[3])
        TransformerA4 = self.basicLayer1(camA4_2, encode_sizeA4)

        camB4 =self.cam1(convB4)
        camB4_1 = torch.cat(( camB4* convB4, convA4), 1)
        convB5 = self.conv1(camB4_1)
        camB4_2 = self.cam1(convB5) * convB5

        encode_sizeB4 = (camB4_2.shape[2], camB4_2.shape[3])
        TransformerB4 = self.basicLayer1(camB4_2, encode_sizeB4)



        cat=torch.cat(( TransformerA4 ,  TransformerB4 ), 1)


        out = self.convolutional_out(cat )
        return out



