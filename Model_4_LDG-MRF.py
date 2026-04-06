import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions.normal import Normal
import einops
from torch_kmeans import KMeans

class SpatialTransformer_block(nn.Module):
    def __init__(self, mode='bilinear'):
        super().__init__()
        self.mode = mode
    def forward(self, src, flow):
        shape = flow.shape[2:]
        vectors = [torch.arange(0, s) for s in shape]
        grids = torch.meshgrid(vectors, indexing='ij')
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)
        grid = grid.to(flow.device)
        new_locs = grid + flow
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)
        new_locs = new_locs.permute(0, 2, 3, 4, 1)
        new_locs = new_locs[..., [2, 1, 0]]
        return F.grid_sample(src, new_locs, align_corners=True, mode=self.mode)

class VecInt(nn.Module):
    def __init__(self, inshape, nsteps=7):
        super().__init__()
        assert nsteps >= 0, 'nsteps should be >= 0, found: %d' % nsteps
        self.nsteps = nsteps
        self.scale = 1.0 / (2 ** self.nsteps)
        self.transformer = SpatialTransformer_block(inshape)
    def forward(self, vec):
        vec = vec * self.scale
        for _ in range(self.nsteps):
            vec = vec + self.transformer(vec, vec)
        return vec

class ResizeTransformer_block(nn.Module):
    def __init__(self, resize_factor, mode='trilinear'):
        super().__init__()
        self.factor = resize_factor
        self.mode = mode
    def forward(self, x):
        if self.factor < 1:
            # resize first to save memory
            x = F.interpolate(x, align_corners=True, scale_factor=self.factor, mode=self.mode)
            x = self.factor * x
        elif self.factor > 1:
            # multiply first to save memory
            x = self.factor * x
            x = F.interpolate(x, align_corners=True, scale_factor=self.factor, mode=self.mode)
        return x

def window_partition(x, window_size: tuple, mode: str):  # n, h, w, d, c
    """Image to windows."""
    _, height, width, depth, _ = x.shape
    fh, fw, fd = window_size[0], window_size[1], window_size[2]
    pad_l = pad_t = pad_d0 = 0
    pad_d1 = (fh - height % fh) % fh
    pad_b = (fw - width % fw) % fw
    pad_r = (fd - depth % fd) % fd
    x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b, pad_d0, pad_d1))
    _, height_pad, width_pad, depth_pad, _ = x.shape
    gh, gw, gd = height_pad // fh, width_pad // fw, depth_pad // fd
    # x = einops.rearrange(x, "n (gh fh) (gw fw) (gd fd) c -> n (gh gw gd) (fh fw fd) c", gh=grid_height, gw=grid_width, gd=grid_depth, fh=fh, fw=fw, fd=fd)
    if mode == 'mlp':
        x = einops.rearrange(x, "n (gh fh) (gw fw) (gd fd) c -> n gh gw gd fh fw fd c", gh=gh, gw=gw, gd=gd, fh=fh, fw=fw, fd=fd)
    elif mode == 'conv':
        x = einops.rearrange(x, "n (gh fh) (gw fw) (gd fd) c -> (n gh gw gd) c fh fw fd", gh=gh, gw=gw, gd=gd, fh=fh, fw=fw, fd=fd)
    grid_size = (gh, gw, gd)
    padding = (pad_d1, pad_b, pad_r)
    original_size = (height, width, depth)
    return x, grid_size, padding, original_size

def reversed_window_partition(x, grid_size: tuple, padding: tuple, original_size: tuple, window_size: tuple, mode: str):
    """Windows to image."""
    # x = einops.rearrange(x, "n (gh gw gd) (fh fw fd) c -> n (gh fh) (gw fw) (gd fd) c", gh=grid_size[0], gw=grid_size[1], gd=grid_size[2], fh=window_size[0], fw=window_size[1], fd=window_size[2])
    if mode == 'mlp':
        x = einops.rearrange(x, "n (gh gw gd) (fh fw fd) c -> n (gh fh) (gw fw) (gd fd) c", n=1, gh=grid_size[0], gw=grid_size[1], gd=grid_size[2], fh=window_size[0], fw=window_size[1], fd=window_size[2])
    elif mode == 'conv':
        x = einops.rearrange(x, "(n gh gw gd) c fh fw fd -> n (gh fh) (gw fw) (gd fd) c", n=1, gh=grid_size[0], gw=grid_size[1], gd=grid_size[2], fh=window_size[0], fw=window_size[1], fd=window_size[2])
    if padding[0] > 0 or padding[1] > 0 or padding[2] > 0:
        x = x[:, :original_size[0], :original_size[1], :original_size[2], :].permute(0, 4, 1, 2, 3).contiguous()
    return x

def correlation(mov_features, fix_features, max_disp=1):
    fix_features = nn.ConstantPad3d(max_disp, 0)(fix_features)
    offsetx, offsety, offsetz = torch.meshgrid([torch.arange(0, 2 * max_disp + 1),
                                                torch.arange(0, 2 * max_disp + 1),
                                                torch.arange(0, 2 * max_disp + 1)], indexing='ij')
    h, w, d = mov_features.shape[2], mov_features.shape[3], mov_features.shape[4]
    out_features = torch.cat([torch.mean(mov_features * fix_features[:, :, dx:dx+h, dy:dy+w, dz:dz+d], 1, keepdim=True)
                              for dx, dy, dz in zip(offsetx.reshape(-1), offsety.reshape(-1), offsetz.reshape(-1))], 1)
    return out_features

class Conv_block(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, pre_fn=None):
        super(Conv_block, self).__init__()
        self.double_conv_block = nn.Sequential(pre_fn if pre_fn else nn.Identity(),
                                    nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
                                    nn.LeakyReLU(0.2),
                                    nn.InstanceNorm3d(out_channels),
                                    nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
                                    nn.LeakyReLU(0.2),
                                    nn.InstanceNorm3d(out_channels)
                                    )
    def forward(self, x):
        return self.double_conv_block(x)

class Encoder(nn.Module):
    def __init__(self, in_channels: int, channel_num: int):
        super(Encoder, self).__init__()
        self.Convblock_1 = Conv_block(channel_num, channel_num//2, pre_fn=nn.Sequential(nn.Conv3d(in_channels, channel_num, kernel_size=3, stride=1, padding=1),
                                                                                        nn.LeakyReLU(0.2),
                                                                                        nn.InstanceNorm3d(channel_num)
                                                                                        ))
        self.Convblock_2 = Conv_block(channel_num//2, channel_num, pre_fn=nn.AvgPool3d(2))
        self.Convblock_3 = Conv_block(channel_num, channel_num*2, pre_fn=nn.AvgPool3d(2))
        self.Convblock_4 = Conv_block(channel_num*2, channel_num*4, pre_fn=nn.AvgPool3d(2))
        self.Convblock_5 = Conv_block(channel_num*4, channel_num*8, pre_fn=nn.AvgPool3d(2))
    def forward(self, x):
        x_1 = self.Convblock_1(x)
        x_2 = self.Convblock_2(x_1)
        x_3 = self.Convblock_3(x_2)
        x_4 = self.Convblock_4(x_3)
        x_5 = self.Convblock_5(x_4)
        return x_1, x_2, x_3, x_4, x_5

class Adjacent_matrix(nn.Module):
    def __init__(self, in_channels: int, channel_ratios: tuple, dropout_ratio: float, weight_range: str, laplace: bool):
        super(Adjacent_matrix, self).__init__()
        self.weight_range = weight_range
        self.laplace = laplace
        self.conv2d_1 = nn.Conv2d(in_channels, in_channels*channel_ratios[0], kernel_size=1, stride=1)
        self.norm_1 = nn.BatchNorm2d(in_channels*channel_ratios[0])
        self.conv2d_2 = nn.Conv2d(in_channels*channel_ratios[0], in_channels*channel_ratios[1], kernel_size=1, stride=1)
        self.norm_2 = nn.BatchNorm2d(in_channels*channel_ratios[1])
        self.conv2d_3 = nn.Conv2d(in_channels*channel_ratios[1], 1, kernel_size=1, stride=1)
        self.dropout = nn.Dropout(dropout_ratio)
    def forward(self, x, adj_diag):
        adj1 = x.unsqueeze(2)
        adj2 = torch.transpose(adj1, 1, 2)
        adj_new = torch.abs(adj1 - adj2).permute(0, 3, 1, 2)
        adj_new = self.conv2d_1(adj_new)
        adj_new = self.norm_1(adj_new)
        adj_new = F.leaky_relu(adj_new, negative_slope=0.2)
        adj_new = self.dropout(adj_new)
        adj_new = self.conv2d_2(adj_new)
        adj_new = self.norm_2(adj_new)
        adj_new = F.leaky_relu(adj_new, negative_slope=0.2)
        adj_new = self.dropout(adj_new)
        adj_new = self.conv2d_3(adj_new)
        adj_new = torch.transpose(adj_new, 1, 3)

        if self.weight_range == 'softmax':
            adj_new = adj_new - adj_diag.expand_as(adj_new) * 1e8  # (210,8,8,1)
            adj_new = torch.transpose(adj_new, 2, 3)  # (210,8,1,8)
            adj_new = adj_new.contiguous()
            adj_new_size = adj_new.size()
            adj_new = adj_new.view(-1, adj_new.size(3))
            adj_new = F.softmax(adj_new, dim=-1)
            adj_new = adj_new.view(adj_new_size)
            adj_new = torch.transpose(adj_new, 2, 3)
        elif self.weight_range == 'sigmoid':
            adj_new = F.sigmoid(adj_new)
            adj_new *= (1 - adj_diag)
        elif self.weight_range == 'none':
            adj_new *= (1 - adj_diag)
        else:
            raise (NotImplementedError)
        if self.laplace:
            adj_new = adj_diag - adj_new
        else:
            adj_new = torch.cat([adj_diag, adj_new], 3)
        return adj_new

class Adjacent_matrix_LocalGNN(nn.Module):
    def __init__(self, in_channels: int, channel_ratios: tuple, dropout_ratio: float, weight_range: str, laplace: bool):
        super(Adjacent_matrix_LocalGNN, self).__init__()
        self.weight_range = weight_range
        self.laplace = laplace
        self.conv2d_1 = nn.Conv2d(in_channels, in_channels*channel_ratios[0], kernel_size=1, stride=1)
        self.norm_1 = nn.BatchNorm2d(in_channels*channel_ratios[0])
        self.conv2d_2 = nn.Conv2d(in_channels*channel_ratios[0], in_channels*channel_ratios[1], kernel_size=1, stride=1)
        self.norm_2 = nn.BatchNorm2d(in_channels*channel_ratios[1])
        self.conv2d_3 = nn.Conv2d(in_channels*channel_ratios[1], 1, kernel_size=1, stride=1)
        self.dropout = nn.Dropout(dropout_ratio)
    def forward(self, x, adj_diag):
        adj1 = x.unsqueeze(2)
        adj2 = torch.transpose(adj1, 1, 2)
        adj_new = torch.abs(adj1 - adj2).permute(0, 3, 1, 2)
        adj_new = self.conv2d_1(adj_new)
        adj_new = self.norm_1(adj_new)
        adj_new = F.leaky_relu(adj_new, negative_slope=0.2)
        adj_new = self.dropout(adj_new)
        adj_new = self.conv2d_2(adj_new)
        adj_new = self.norm_2(adj_new)
        adj_new = F.leaky_relu(adj_new, negative_slope=0.2)
        adj_new = self.dropout(adj_new)
        adj_new = self.conv2d_3(adj_new)
        adj_new = torch.transpose(adj_new, 1, 3)

        if self.weight_range == 'softmax':
            adj_new = adj_new - adj_diag.expand_as(adj_new) * 1e8  # (210,8,8,1)
            adj_new = torch.transpose(adj_new, 2, 3)  # (210,8,1,8)
            adj_new = adj_new.contiguous()
            adj_new_size = adj_new.size()
            adj_new = adj_new.view(-1, adj_new.size(3))
            adj_new = F.softmax(adj_new, dim=-1)
            adj_new = adj_new.view(adj_new_size)
            adj_new = torch.transpose(adj_new, 2, 3)
        elif self.weight_range == 'sigmoid':
            adj_new = F.sigmoid(adj_new)
            adj_new *= (1 - adj_diag)
        elif self.weight_range == 'none':
            adj_new *= (1 - adj_diag)
        else:
            raise (NotImplementedError)
        if self.laplace:
            adj_new = adj_diag - adj_new
        else:
            adj_new = torch.cat([adj_diag, adj_new], 3)
        return adj_new

def gmul_GCN(edge_weight, node_features):
    edge_weight_size = edge_weight.size()
    N = edge_weight_size[-2]
    edge_weight = edge_weight.split(1, 3)
    edge_weight = torch.cat(edge_weight, 1).squeeze(3)
    output = torch.bmm(edge_weight, node_features)
    output = output.split(N, 1)
    output = torch.cat(output, 2)
    return output

class GCN_Global(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout_ratio: float):
        super(GCN_Global, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.gcn_weight = nn.Parameter(torch.zeros(in_channels*2, out_channels), requires_grad=True)
        nn.init.kaiming_uniform_(self.gcn_weight, a=math.sqrt(5))
        self.dropout = nn.Dropout(dropout_ratio)
        self.norm = nn.BatchNorm1d(out_channels)
    def forward(self, edge_weight, node_features):
        node_features = gmul_GCN(edge_weight, node_features)
        node_features = node_features.contiguous().view(-1, self.in_channels*2)
        node_features = self.dropout(F.leaky_relu(torch.matmul(node_features, self.gcn_weight), 0.2))
        node_features = self.norm(node_features).unsqueeze(0)
        return node_features

class GCN_Cluster(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout_ratio: float):
        super(GCN_Cluster, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.gcn_weight = nn.Parameter(torch.zeros(in_channels*2, out_channels), requires_grad=True)
        nn.init.kaiming_uniform_(self.gcn_weight, a=math.sqrt(5))
        self.dropout = nn.Dropout(dropout_ratio)
        self.norm_1 = nn.BatchNorm1d(out_channels)
    def forward(self, edge_weight, node_features, spatial_weight):
        node_features = gmul_GCN(edge_weight, node_features)
        weight = torch.matmul(self.gcn_weight, spatial_weight)
        node_features = node_features.contiguous().view(-1, self.in_channels*2)
        node_features = self.dropout(F.leaky_relu(torch.matmul(node_features, weight), 0.2))
        node_features = self.norm_1(node_features).unsqueeze(0)
        return node_features

class GCN_Local(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout_ratio: float):
        super(GCN_Local, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.gcn_weight = nn.Parameter(torch.zeros(in_channels*2, out_channels), requires_grad=True)
        nn.init.kaiming_uniform_(self.gcn_weight, a=math.sqrt(5))
        self.dropout = nn.Dropout(dropout_ratio)
        self.norm = nn.BatchNorm1d(out_channels)
    def forward(self, edge_weight, node_features):
        node_features = gmul_GCN(edge_weight, node_features)
        node_features_size = node_features.size()
        node_features = node_features.contiguous().view(-1, self.in_channels*2)
        node_features = self.dropout(F.leaky_relu(torch.matmul(node_features, self.gcn_weight), 0.2))
        node_features = node_features.view(*node_features_size[:-1], self.out_channels)
        return node_features

class MLP(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(MLP, self).__init__()
        self.mlp = nn.Sequential(nn.Linear(in_channels, out_channels//8),
                                 nn.ReLU(),
                                 nn.Linear(out_channels//8, out_channels//4),
                                 nn.ReLU(),
                                 nn.Linear(out_channels//4, out_channels),
                                 nn.Sigmoid())
    def forward(self, x):
        return self.mlp(x)

class KMeans_Get_Cluster_Index_and_Centroids(nn.Module):
    def __init__(self, cluster_nums: int):
        super(KMeans_Get_Cluster_Index_and_Centroids, self).__init__()
        self.cluster_nums = cluster_nums
        self.kmeans_hard = KMeans(n_clusters=cluster_nums)
        # self.kmeans_soft = SoftKMeans()
        # self.kmeans_constrained = ConstrainedKMeans()
    def forward(self, x):
        result = self.kmeans_hard(x)
        # result = self.kmeans_soft(x, k=self.cluster_nums)
        # result = self.kmeans_constrained(x, k=self.cluster_nums, weights=nn.Parameter(torch.ones(x.shape[0], x.shape[1])).cuda())
        cluster_index = result.labels
        cluster_centroids = result.centers
        return cluster_index, cluster_centroids

def scatter_restore(cluster_index, cluster_outputs, N, C):
    restored = torch.zeros((N, C), device=cluster_index.device, dtype=cluster_outputs[0].dtype)
    for i, cluster_tensor in enumerate(cluster_outputs):
        idx = (cluster_index == i).nonzero(as_tuple=True)[1]
        if idx.numel() == 0:
            continue
        restored[idx] = cluster_tensor
    return restored.view(1, N, C)

class Proposed_Module_ClusterGNN(nn.Module):
    def __init__(self, in_channels: int, window_size: tuple, cluster_nums: int):
        super(Proposed_Module_ClusterGNN, self).__init__()
        self.in_channels = in_channels
        self.window_size = window_size
        self.adj = nn.ModuleList([Adjacent_matrix(in_channels, (2, 3), 0.2, weight_range='softmax', laplace=False) for i in range(cluster_nums)])
        self.get_weight = MLP(in_channels, in_channels*in_channels)
        self.GCN = nn.ModuleList([GCN_Cluster(in_channels, in_channels, 0.2) for i in range(cluster_nums)])
        self.cluster_nums = cluster_nums
        self.kmeans = KMeans_Get_Cluster_Index_and_Centroids(cluster_nums=cluster_nums)
    def forward(self, x_concat):
        x, grid_size, padding, original_size = window_partition(x_concat, self.window_size, mode='mlp')
        x = einops.reduce(x, 'b gh gw gd fh fw fd c -> b (gh gw gd) c', 'mean')
        cluster_outputs = []
        cluster_index, cluster_centroids = self.kmeans(x)
        for i in range(self.cluster_nums):
            cluster = x[cluster_index == i].unsqueeze(0)
            if cluster.shape[1] == 1:
                cluster_outputs.append(cluster)
            else:
                adj_diag = Variable(torch.eye(cluster.shape[1]).unsqueeze(0).repeat(1, 1, 1).unsqueeze(3)).cuda()
                adj_new = self.adj[i](cluster, adj_diag)
                spatial_weight = self.get_weight(cluster_centroids[:, i, :])
                spatial_weight = spatial_weight.contiguous().view(self.in_channels, self.in_channels)
                cluster_new = self.GCN[i](adj_new, cluster, spatial_weight)
                cluster_outputs.append(cluster_new)
        x_new = scatter_restore(cluster_index, cluster_outputs, N=x.size(1), C=self.in_channels)
        x_new = einops.rearrange(x_new, 'b (gh gw gd) c -> b c gh gw gd', gh=grid_size[0], gw=grid_size[1], gd=grid_size[2])
        if self.window_size[0] == 2:
            x_new_pad = F.pad(x_new, (0, 1, 0, 1, 0, 1))
            x_unfold = x_new_pad.unfold(2, 2, 1).unfold(3, 2, 1).unfold(4, 2, 1)
        elif self.window_size[0] == 4:
            x_new_pad = F.pad(x_new, ((1, 2, 1, 2, 1, 2)))
            x_unfold = x_new_pad.unfold(2, 4, 1).unfold(3, 4, 1).unfold(4, 4, 1)
        elif self.window_size[0] == 8:
            x_new_pad = F.pad(x_new, (3, 4, 3, 4, 3, 4))
            x_unfold = x_new_pad.unfold(2, 8, 1).unfold(3, 8, 1).unfold(4, 8, 1)
        elif self.window_size[0] == 16:
            x_new_pad = F.pad(x_new, (7, 8, 7, 8, 7, 8))
            x_unfold = x_new_pad.unfold(2, 16, 1).unfold(3, 16, 1).unfold(4, 16, 1)
        else:
            return 0
        x_unfold = einops.rearrange(x_unfold, 'b c gh gw gd fh fw fd -> b (gh fh) (gw fw) (gd fd) c')
        out = x_unfold + x_concat
        return out

class Proposed_Module_GlobalGNN(nn.Module):
    def __init__(self, in_channels: int, window_size: tuple):
        super(Proposed_Module_GlobalGNN, self).__init__()
        self.window_size = window_size
        self.adj = Adjacent_matrix(in_channels, channel_ratios=(2, 3), dropout_ratio=0.2, weight_range='softmax', laplace=False)
        self.GCN = GCN_Global(in_channels, in_channels, dropout_ratio=0.2)
    def forward(self, x_concat):
        x, grid_size, padding, original_size = window_partition(x_concat, self.window_size, mode='mlp')
        x = einops.reduce(x, 'b gh gw gd fh fw fd c -> b (gh gw gd) c', 'mean')
        adj_diag = Variable(torch.eye(x.shape[1]).unsqueeze(0).repeat(1, 1, 1).unsqueeze(3)).cuda()
        adj_new = self.adj(x, adj_diag)
        x_new = self.GCN(adj_new, x)
        x_new = einops.rearrange(x_new, 'b (gh gw gd) c -> b c gh gw gd', gh=grid_size[0], gw=grid_size[1], gd=grid_size[2])
        if self.window_size[0] == 2:
            x_new_pad = F.pad(x_new, (0, 1, 0, 1, 0, 1))
            x_unfold = x_new_pad.unfold(2, 2, 1).unfold(3, 2, 1).unfold(4, 2, 1)
        elif self.window_size[0] == 4:
            x_new_pad = F.pad(x_new, ((1, 2, 1, 2, 1, 2)))
            x_unfold = x_new_pad.unfold(2, 4, 1).unfold(3, 4, 1).unfold(4, 4, 1)
        elif self.window_size[0] == 8:
            x_new_pad = F.pad(x_new, (3, 4, 3, 4, 3, 4))
            x_unfold = x_new_pad.unfold(2, 8, 1).unfold(3, 8, 1).unfold(4, 8, 1)
        elif self.window_size[0] == 16:
            x_new_pad = F.pad(x_new, (7, 8, 7, 8, 7, 8))
            x_unfold = x_new_pad.unfold(2, 16, 1).unfold(3, 16, 1).unfold(4, 16, 1)
        else:
            return 0
        x_unfold = einops.rearrange(x_unfold, 'b c gh gw gd fh fw fd -> b (gh fh) (gw fw) (gd fd) c')
        out = x_unfold + x_concat
        return out

class Proposed_Module_LocalGNN(nn.Module):
    def __init__(self, in_channels: int, window_size: tuple):
        super(Proposed_Module_LocalGNN, self).__init__()
        self.window_size = window_size
        self.conv_change_channels = nn.Sequential(nn.Conv3d(in_channels, in_channels*2, 3, 1, 1),
                                                  nn.InstanceNorm3d(in_channels*2),
                                                  nn.LeakyReLU(0.2),
                                                  nn.Conv3d(in_channels*2, in_channels, 3, 1, 1),
                                                  nn.InstanceNorm3d(in_channels),
                                                  nn.LeakyReLU(0.2))
        self.conv_downsample = nn.Sequential(nn.Conv3d(in_channels, in_channels, window_size[0]//2, window_size[0]//2, 0),
                                             nn.BatchNorm3d(in_channels),
                                             nn.LeakyReLU(0.2))
        self.conv_upsample = nn.Sequential(nn.ConvTranspose3d(in_channels, in_channels, window_size[0]//2, window_size[0]//2, 0),
                                           nn.BatchNorm3d(in_channels),
                                           nn.LeakyReLU(0.2))
        self.adj = Adjacent_matrix_LocalGNN(in_channels, (2, 3), 0.2, weight_range='softmax', laplace=False)
        self.GCN = GCN_Local(in_channels, in_channels, 0.2)
    def forward(self, x_concat):
        x_concat = self.conv_change_channels(x_concat)
        x, grid_size, padding, original_size = window_partition(x_concat.permute(0, 2, 3, 4, 1), self.window_size, mode='conv')
        x = self.conv_downsample(x)
        x = einops.rearrange(x, 'b c d h w -> b (d h w) c')
        adj_diag = Variable(torch.eye(x.size(1)).unsqueeze(0).repeat(x.size(0), 1, 1).unsqueeze(3)).cuda()
        adj_new = self.adj(x, adj_diag)
        x_new = self.GCN(adj_new, x)
        x_new = einops.rearrange(x_new, 'b (d h w) c -> b c d h w', d=2, h=2, w=2)
        x_new = self.conv_upsample(x_new)
        x_new = reversed_window_partition(x_new, grid_size, padding, original_size, self.window_size, 'conv')
        return x_new

class context_attention(nn.Module):
    def __init__(self, in_channels: int):
        super(context_attention, self).__init__()
        self.in_channels = in_channels
        self.reweight = nn.Sequential(nn.Linear(in_channels, in_channels//4),
                                      nn.GELU(),
                                      nn.Linear(in_channels//4, in_channels*2),
                                      nn.Dropout(0.2))
    def forward(self, x_1, x_2):
        a = (x_1 + x_2).permute(0, 4, 1, 2, 3).flatten(2).mean(2)
        a = self.reweight(a).reshape(1, self.in_channels, 2).permute(2, 0, 1).softmax(dim=0).unsqueeze(2).unsqueeze(2).unsqueeze(2)
        sum = x_1 * a[0] + x_2 * a[1]
        return sum

class Proposed_Module(nn.Module):
    def __init__(self, in_channels: int, window_size: tuple, cluster_nums: int):
        super(Proposed_Module, self).__init__()
        self.conv_block = nn.Sequential(nn.Conv3d(in_channels*2+27, in_channels, kernel_size=3, stride=1, padding=1),
                                  nn.LeakyReLU(0.2),
                                  nn.InstanceNorm3d(in_channels))
        self.module_1 = Proposed_Module_GlobalGNN(in_channels=in_channels, window_size=window_size)
        self.module_2 = Proposed_Module_ClusterGNN(in_channels=in_channels, window_size=window_size, cluster_nums=cluster_nums)
        self.module_3 = Proposed_Module_LocalGNN(in_channels=in_channels, window_size=window_size)
        self.context_attention_1 = context_attention(in_channels=in_channels)
        self.context_attention_2 = context_attention(in_channels=in_channels)
    def forward(self, x_1, x_2):
        x_corr = correlation(x_1, x_2, 1)
        x_concat = torch.concat([x_1, x_corr, x_2], dim=1)
        x_concat = self.conv_block(x_concat).permute(0, 2, 3, 4, 1)
        x_global = self.module_1(x_concat)
        x_cluster = self.module_2(x_concat)
        x_new_1 = self.context_attention_1(x_global, x_cluster)
        x_new_2 = x_concat + x_new_1
        x_new_2 = x_new_2.permute(0, 4, 1, 2, 3)
        x_new_3 = self.module_3(x_new_2)
        x_new_4 = self.context_attention_2(x_new_1, x_new_3)
        result = x_concat + x_new_4
        result = result.permute(0, 4, 1, 2, 3)
        return result

class Flow_Estimator(nn.Module):
    def __init__(self, in_channels: int, if_corr: bool):
        super(Flow_Estimator, self).__init__()
        if if_corr:
            self.conv1 = nn.Conv3d(in_channels*2+27, in_channels, 3, 1, 1)
            self.norm1 = nn.InstanceNorm3d(in_channels)
            self.flow_conv = nn.Conv3d(in_channels, 3, 3, 1, 1)
            self.flow_conv.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow_conv.weight.shape))
            self.flow_conv.bias = nn.Parameter(torch.zeros(self.flow_conv.bias.shape))
        else:
            self.conv1 = nn.Conv3d(in_channels, in_channels//2, 3, 1, 1)
            self.norm1 = nn.InstanceNorm3d(in_channels//2)
            self.flow_conv = nn.Conv3d(in_channels//2, 3, 3, 1, 1)
            self.flow_conv.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow_conv.weight.shape))
            self.flow_conv.bias = nn.Parameter(torch.zeros(self.flow_conv.bias.shape))
    def forward(self, x):
        x = F.leaky_relu(self.norm1(self.conv1(x)), 0.2)
        x = self.flow_conv(x)
        return x

class Model(nn.Module):
    def __init__(self, channel_num: int, cluster_num: int):
        super(Model, self).__init__()
        self.encoder = Encoder(in_channels=1, channel_num=channel_num)

        self.proposed_module_2 = Proposed_Module(in_channels=channel_num, window_size=(16, 16, 16), cluster_nums=cluster_num)
        self.proposed_module_3 = Proposed_Module(in_channels=channel_num*2, window_size=(8, 8, 8), cluster_nums=cluster_num)
        self.proposed_module_4 = Proposed_Module(in_channels=channel_num*4, window_size=(4, 4, 4), cluster_nums=cluster_num)
        self.proposed_module_5 = Proposed_Module(in_channels=channel_num*8, window_size=(2, 2, 2), cluster_nums=cluster_num)

        self.reghead_5 = Flow_Estimator(in_channels=channel_num*8, if_corr=False)
        self.reghead_4 = Flow_Estimator(in_channels=channel_num*4, if_corr=False)
        self.reghead_3 = Flow_Estimator(in_channels=channel_num*2, if_corr=False)
        self.reghead_2 = Flow_Estimator(in_channels=channel_num, if_corr=False)
        self.reghead_1 = Flow_Estimator(in_channels=channel_num//2, if_corr=True)

        self.ResizeTransformer = ResizeTransformer_block(resize_factor=2, mode='trilinear')
        self.SpatialTransformer = SpatialTransformer_block(mode='bilinear')
    def forward(self, mov, fix):
        m1, m2, m3, m4, m5 = self.encoder(mov)
        f1, f2, f3, f4, f5 = self.encoder(fix)

        # Step 1
        x_5 = self.proposed_module_5(m5, f5)
        flow_5 = self.reghead_5(x_5)

        # Step 2
        flow_5_up = self.ResizeTransformer(2 * flow_5)
        m4 = self.SpatialTransformer(m4, flow_5_up)
        x_4 = self.proposed_module_4(m4, f4)
        x = self.reghead_4(x_4)
        flow_4 = self.SpatialTransformer(flow_5_up, x) + x

        # Step 3
        flow_4_up = self.ResizeTransformer(2 * flow_4)
        m3 = self.SpatialTransformer(m3, flow_4_up)
        x_3 = self.proposed_module_3(m3, f3)
        x = self.reghead_3(x_3)
        flow_3 = self.SpatialTransformer(flow_4_up, x) + x

        # Step 4
        flow_3_up = self.ResizeTransformer(2 * flow_3)
        m2 = self.SpatialTransformer(m2, flow_3_up)
        x_2 = self.proposed_module_2(m2, f2)
        x = self.reghead_2(x_2)
        flow_2 = self.SpatialTransformer(flow_3_up, x) + x

        # Step 5
        flow_2_up = self.ResizeTransformer(2 * flow_2)
        m1 = self.SpatialTransformer(m1, flow_2_up)
        corr = correlation(m1, f1, 1)
        x = self.reghead_1(torch.cat([m1, corr, f1], dim=1))
        flow_1 = self.SpatialTransformer(flow_2_up, x) + x

        warped_mov = self.SpatialTransformer(mov, flow_1)
        return warped_mov, flow_1
