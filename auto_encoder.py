import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from tcn import TemporalConvNet

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

class GraphConvLayer(nn.Module):
    def __init__(self, i, c, Lk, ks=3):
        super(GraphConvLayer, self).__init__()
        self.Lk = Lk
        self.theta = nn.Parameter(torch.FloatTensor(i, c, ks))
        self.b = nn.Parameter(torch.FloatTensor(1, c, 1, 1))
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.theta, a=math.sqrt(5))
        fan_in, _ = init._calculate_fan_in_and_fan_out(self.theta)
        bound = 1 / math.sqrt(fan_in)
        init.uniform_(self.b, -bound, bound)

    def forward(self, x):
        x_c = torch.einsum("knm,bitm->bitkn", self.Lk, x)
        x_gc = torch.einsum("iok,bitkn->botn", self.theta, x_c) + self.b
        return torch.relu(x_gc+x)


class GraphConv(nn.Module):
    def __init__(self, input_feature, embedding_size, gconv_output,graph_lap):
        super(GraphConv, self).__init__()
        self.graph_conv_layer1 = GraphConvLayer(input_feature, embedding_size, graph_lap)
        self.graph_conv_layer2 = GraphConvLayer(embedding_size, embedding_size, graph_lap)
        self.fc = nn.Linear(embedding_size, gconv_output)

    def reset_parameters(self):
        init.kaiming_uniform_(self.theta, a=math.sqrt(5))
        fan_in, _ = init._calculate_fan_in_and_fan_out(self.theta)
        bound = 1 / math.sqrt(fan_in)
        init.uniform_(self.b, -bound, bound)

    def forward(self, x):
        x_1 = self.graph_conv_layer1(x)
        x_output = self.graph_conv_layer2(x_1)
        x_output = x_output.permute(0, 2, 3, 1)
        x_output = self.fc(x_output)
        return x_output


class AutoEncoderModel(nn.Module):
    def __init__(self, input_feature, embedding_size, num_channels, num_inputs, Lk, kernel_size):
        super(AutoEncoderModel, self).__init__()
        self.Lk = Lk
        self.spatial = GraphConv(input_feature=input_feature, embedding_size=embedding_size, graph_lap=Lk,
                                 gconv_output=1)
        self.temporal = TemporalConvNet(num_inputs=num_inputs, num_channels=num_channels, kernel_size=kernel_size)
        self.linear2 = nn.Linear(num_channels[-1], num_inputs)

    def forward(self, x):
        spatial_x = self.spatial(x)              # 64, 1, 12, 325、# 64, 12, 325
        temporal_x = self.temporal(spatial_x.squeeze())    # 64, 64, 325
        temporal_x = temporal_x.permute(0, 2, 1)
        dec_output = self.linear2(temporal_x)
        return dec_output.permute(0, 2, 1)


