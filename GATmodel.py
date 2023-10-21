from torch import nn
import torch
import torch.nn.functional as F
from tcn import TemporalConvNet


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_c, out_c):
        super(GraphAttentionLayer, self).__init__()
        self.in_c = in_c
        self.out_c = out_c

        self.F = F.softmax
        self.W = nn.Linear(in_c, out_c, bias=False)
        self.b = nn.Parameter(torch.Tensor(out_c))

        nn.init.normal_(self.W.weight)
        nn.init.normal_(self.b)

    def forward(self, inputs, graph):
        # [B, T, N, C]*[B, T, C, N]*[N, N]
        h = self.W(inputs)
        outputs = torch.matmul(h, h.transpose(2, 3)) * graph.unsqueeze(0)
        outputs.data.masked_fill_(torch.eq(outputs, 0), -float(1e16))
        attention = self.F(outputs, dim=-1)
        return torch.matmul(attention, h) + self.b


class GATSubNet(nn.Module):
    def __init__(self, in_c, hid_c, out_c, n_heads):
        super(GATSubNet, self).__init__()
        self.attention_module = nn.ModuleList([GraphAttentionLayer(in_c, hid_c) for _ in range(n_heads)])
        self.out_att = GraphAttentionLayer(hid_c * n_heads, out_c)

        self.act = nn.LeakyReLU()

    def forward(self, inputs, graph):
        outputs = torch.cat([attn(inputs, graph) for attn in self.attention_module], dim=-1)  # [B, N, hid_c * h_head]
        outputs = self.act(outputs)

        outputs = self.out_att(outputs, graph)

        return self.act(outputs)


class GATNet(nn.Module):
    def __init__(self, in_c, hid_c, out_c, n_heads):
        super(GATNet, self).__init__()
        self.subnet = GATSubNet(in_c, hid_c, out_c, n_heads)

    def forward(self, data, graph):
        outputs = self.subnet(data, graph)
        return outputs




class GATTCNModel(nn.Module):
    def __init__(self, input_feature, embedding_size, num_channels, num_inputs, Lk, kernel_size,n_heads):
        super(GATTCNModel, self).__init__()
        self.Lk = Lk
        self.spatial = GATNet(in_c=input_feature, hid_c=embedding_size, out_c=1, n_heads=n_heads)
        self.temporal = TemporalConvNet(num_inputs=num_inputs, num_channels=num_channels, kernel_size=kernel_size)
        self.linear2 = nn.Linear(num_channels[-1], num_inputs)

    def forward(self, x):
        # 64, 12, 307, 1
        x = x.permute(0, 2, 3, 1)
        spatial_x = self.spatial(x, self.Lk)
        spatial_x = spatial_x + x # residual connect
        temporal_x = self.temporal(spatial_x.squeeze())
        temporal_x = self.linear2(temporal_x.permute(0, 2, 1))
        # print(temporal_x.shape)
        return temporal_x.permute(0, 2, 1)

