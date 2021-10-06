import argparse
import torch
import torch.nn as nn
from utils.graph_conv import calculate_laplacian_with_self_loop
import torch.nn.functional as F

class global_mean_pooling(nn.Module):
    def __init__(self,**kwargs):
        super(global_mean_pooling, self).__init__()

    def forward(self,input):
        #(batch_size,num_node,feature)

        # return(batch_size,feature)
        return input.sum(dim = 1) / input.shape[1]

class GCNLayer(nn.Module):
    def __init__(self, adj,input_dim:int,output_dim:int,**kwargs):
        super(GCNLayer, self).__init__()
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

class TGCNCeil(nn.Module):      #计算中间的隐藏状态和输出
    def __init__(self,num_layer,adj,input_dim,output_dim,pooling_first = True,**kwargs):
        super(TGCNCeil, self).__init__()
        self.num_layer = num_layer
        self.adj = adj
        self.input_dim = input_dim
        # self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.pooling_first = pooling_first
        self.GCNNet = nn.ModuleList([GCNLayer(adj,input_dim,output_dim),nn.ELU()])

        for i in range(num_layer - 1):
            self.GCNNet.append(GCNLayer(adj,output_dim,output_dim))
            self.GCNNet.append(nn.ELU())
        self.Wu = nn.Linear(2 * output_dim,output_dim)
        self.Wr = nn.Linear(2 * output_dim,output_dim)
        self.Wc = nn.Linear(2 * output_dim,output_dim)


        self.pooling = global_mean_pooling()
        self.add_module('pooling',self.pooling)

    def forward(self,inputs,hidden_state):
        # inputs: (batch_size,num_nodes,features) | (batch_size, features)
        # hidden_state: (batch_size, num_node, features) | (batch_size, hidden_dim)
        output = inputs
        for layer in self.GCNNet:
            output = layer(output)
        if self.pooling_first == True:
            output = self.pooling(output)
        u_t = torch.cat([output,hidden_state],len(output.shape) - 1)
        r_t = torch.cat([output,hidden_state],len(output.shape) - 1)
        u_t = torch.tanh(self.Wu(u_t))
        r_t = torch.tanh(self.Wr(r_t))
        c_t = torch.cat([output,r_t * hidden_state],len(output.shape) - 1)
        c_t = torch.tanh(self.Wc(c_t))
        new_hidden_state = u_t * hidden_state + (1 - u_t) * c_t
        # print(new_hidden_state.shape)
        return new_hidden_state


class TGCNModel(nn.Module):
    def __init__(self,num_layer,adj,input_dim,hidden_dim,output_dim,pooling_first = True,**kwargs):
        super(TGCNModel, self).__init__()
        self.num_layer = num_layer
        self.adj = adj
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.pooling_first = pooling_first

        self.Ceil = TGCNCeil(num_layer,adj,1,hidden_dim,pooling_first)
        self.pooling = global_mean_pooling()

        self.act = nn.Sigmoid()
        self.W = nn.Linear(hidden_dim,output_dim)

        self.add_module('ceil', self.Ceil)
        self.add_module('pooling', self.pooling)

    def forward(self,inputs):
        # inputs: (batch_size, num_nodes, feature)
        batch_size, num_nodes, seq_len = inputs.shape

        hidden_state = torch.zeros(batch_size, num_nodes, self.hidden_dim).cuda()
        # print(hidden_state.device)
        for i in range(seq_len):
            hidden_state = self.Ceil(inputs[:,:,i].reshape(batch_size,num_nodes,1),hidden_state)

        if (self.pooling_first == False):
            hidden_state = self.pooling(hidden_state)
        # print(hidden_state.shape)
        output = self.act(self.W(hidden_state)).squeeze()
        # print(output.shape)
        return output




