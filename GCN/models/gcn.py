import argparse
import torch
import torch.nn as nn
from utils.graph_conv import calculate_laplacian_with_self_loop

class GCNLayer(nn.Module):
    def __init__(self,adj,input_dim:int,output_dim:int,**kwargs):
        super(GCNLayer,self).__init__()
        self.laplacian = nn.Parameter(calculate_laplacian_with_self_loop(torch.FloatTensor(adj)),requires_grad=False)
        self._input_dim = input_dim
        self._output_dim = output_dim
        self.w = nn.Parameter(torch.FloatTensor(input_dim,output_dim))
        nn.init.xavier_uniform_(self.w, gain=nn.init.calculate_gain("tanh"))

    def forward(self,inputs):
        # (batch_size, num_nodes, feature)
        # (batch_size, seq_len, num_nodes)
        self._num_nodes = inputs.shape[1]

        batch_size = inputs.shape[0]
        # (num_nodes, batch_size, seq_len)
        # inputs = inputs.transpose(0, 2).transpose(1, 2)
        inputs = inputs.transpose(0, 1)
        # (num_nodes, batch_size * seq_len)
        inputs = inputs.reshape((self._num_nodes, batch_size * self._input_dim))
        # AX (num_nodes, batch_size * seq_len)
        # print(self.laplacian.device,inputs.device)
        ax = self.laplacian @ inputs
        # (num_nodes, batch_size, seq_len)
        ax = ax.reshape((self._num_nodes, batch_size, self._input_dim))
        # (num_nodes * batch_size, seq_len)
        ax = ax.reshape((self._num_nodes * batch_size, self._input_dim))
        # act(AXW) (num_nodes * batch_size, output_dim)
        outputs = torch.tanh(ax @ self.w)
        # (num_nodes, batch_size, output_dim)
        outputs = outputs.reshape((self._num_nodes, batch_size, self._output_dim))
        # (batch_size, num_nodes, output_dim)
        outputs = outputs.transpose(0, 1)
        return outputs

    @property
    def hyperparameters(self):
        return {
            "num_nodes": self._num_nodes,
            "input_dim": self._input_dim,
            "output_dim": self._output_dim,
        }

class global_mean_pooling(nn.Module):
    def __init__(self,**kwargs):
        super(global_mean_pooling, self).__init__()

    def forward(self,input):
        #(batch_size,num_node,feature)

        # return(batch_size,feature)
        return input.sum(dim = 1) / input.shape[1]

class GCNModel(nn.Module):
    def __init__(self,num_layer,adj,input_dim,hidden_dim,output_dim,**kwargs):
        super(GCNModel, self).__init__()
        self.adj = adj
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.pooling = global_mean_pooling()
        self.num_layers = num_layer

        self.input_layer = GCNLayer(adj,input_dim,hidden_dim)
        self.act = nn.ELU()

        self.net = nn.ModuleList([self.input_layer,self.act])
        for i in range(num_layer - 1):
            self.net.append(GCNLayer(adj,hidden_dim,hidden_dim))
            self.net.append(self.act)
        self.net.append(self.pooling)
        self.net.append(nn.Linear(hidden_dim,output_dim))
        self.net.append(nn.Sigmoid())

    def forward(self,input):
        #(batch_size,num_node,feature)
        output = input
        for layer in self.net:
            output = layer(output)
        return output.squeeze()

    @staticmethod
    def add_model_specific_arguments(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--hidden_dim", type = int, default=64)
        parser.add_argument("--output_dim", type = int, default=1)
        parser.add_argument("--num_layers", type = int, default = 3)
        return parser

