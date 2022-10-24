import math

import torch
from torch import FloatTensor
from torch.nn import Conv2d, ModuleList, LayerNorm, Module, Parameter, Sequential
import torch.nn.functional as F

device = "cuda"
# class Attention(Module):
# 	def __init__(self, dk, requires_value=False):
# 		super(Attention, self).__init__()
# 		self.sqrt_dk = math.sqrt(dk)
# 		self.requires_value = requires_value
# 		self.W1 = Parameter(torch.zeros(dk, 10), requires_grad=True)
# 		self.W2 = Parameter(torch.zeros(10, dk), requires_grad=True)
#
# 	def forward(self, x: FloatTensor):
# 		x_out = x.reshape(*x.shape[:2], -1)
# 		# [B * A * Dk] @ [Dk * Dk] @ [B * Dk * A] => [B * A * A]
# 		att = x_out @ self.W1 @ self.W2 @ x_out.transpose(1, 2)
# 		att = torch.softmax(att / self.sqrt_dk, dim=-1)
# 		return (att @ x_out).reshape_as(x) if self.requires_value else att
# class Temporal_Attention_layer(Module):
#     def __init__(self, num_of_vertices, num_of_features, num_of_timesteps):
#         super(Temporal_Attention_layer, self).__init__()
#         global device
#         self.U_1 = torch.randn(num_of_vertices, requires_grad=True).to(device)
#         self.U_2 = torch.randn(num_of_features, num_of_vertices, requires_grad=True).to(device)
#         self.U_3 = torch.randn(num_of_features, requires_grad=True).to(device)
#         self.b_e = torch.randn(1, num_of_timesteps, num_of_timesteps, requires_grad=True).to(device)
#         self.V_e = torch.randn(num_of_timesteps, num_of_timesteps, requires_grad=True).to(device)
#
#     def forward(self, x):
#         lhs = torch.matmul(torch.matmul(x.permute(0, 3, 2, 1), self.U_1),self.U_2)
#         rhs = torch.matmul(x.permute((0, 1, 3, 2)), self.U_3)  # Is it ok to switch the position?
#
#         product = torch.matmul(lhs, rhs)  # wd: (batch_size, T, T)
#
#         # (batch_size, T, T)
#         E = torch.matmul(self.V_e, torch.sigmoid(product + self.b_e))
#
#         # normailzation
#         E = E - torch.max(E, 1, keepdim=True)[0]
#         exp = torch.exp(E)
#         E_normalized = exp / torch.sum(exp, 1, keepdim=True)
#         return E_normalized

class Temporal_Attention_layer(Module):
    def __init__(self, num_of_vertices, num_of_features, num_of_timesteps):
        super(Temporal_Attention_layer, self).__init__()

        global device
        self.U_1 = torch.randn(num_of_vertices, requires_grad=True).to(device)
        self.U_2 = torch.randn(num_of_features, num_of_vertices, requires_grad=True).to(device)
        self.U_3 = torch.randn(num_of_features, requires_grad=True).to(device)
        self.b_e = torch.randn(1, num_of_timesteps, num_of_timesteps, requires_grad=True).to(device)
        self.V_e = torch.randn(num_of_timesteps, num_of_timesteps, requires_grad=True).to(device)

    def forward(self, x):
        lhs = torch.matmul(torch.matmul(x.permute(0, 3, 2, 1), self.U_1), self.U_2)
        rhs = torch.matmul(x.permute((0, 1, 3, 2)), self.U_3)  # Is it ok to switch the position?
        product = torch.matmul(lhs, rhs)  # wd: (batch_size, T, T)

        # (batch_size, T, T)
        E = torch.matmul(self.V_e, torch.sigmoid(product + self.b_e))

        # normailzation
        E = E - torch.max(E, 1, keepdim=True)[0]
        exp = torch.exp(E)
        E_normalized = exp / torch.sum(exp, 1, keepdim=True)
        return E_normalized


class Spatial_Attention_layer(Module):
    def __init__(self, num_of_vertices, num_of_features, num_of_timesteps):
        super(Spatial_Attention_layer, self).__init__()

        global device
        self.W_1 = torch.randn(num_of_timesteps, requires_grad=True).to(device)
        self.W_2 = torch.randn(num_of_features, num_of_timesteps, requires_grad=True).to(device)
        self.W_3 = torch.randn(num_of_features, requires_grad=True).to(device)
        self.b_s = torch.randn(1, num_of_vertices, num_of_vertices, requires_grad=True).to(device)
        self.V_s = torch.randn(num_of_vertices, num_of_vertices, requires_grad=True).to(device)

    def forward(self, x):
        lhs = torch.matmul(torch.matmul(x, self.W_1), self.W_2)
        rhs = torch.matmul(x.permute((0, 3, 1, 2)), self.W_3)  # do we need to do transpose??
        product = torch.matmul(lhs, rhs)

        S = torch.matmul(self.V_s, torch.sigmoid(product + self.b_s))

        # normalization
        S = S - torch.max(S, 1, keepdim=True)[0]
        exp = torch.exp(S)
        S_normalized = exp / torch.sum(exp, 1, keepdim=True)
        return S_normalized
# class GCN(Module):
# 	def __init__(self, in_channels, in_timesteps, gcn_filters, A):
# 		super(GCN, self).__init__()
# 		self.A = A
# 		self.W = Parameter(torch.zeros(in_channels, gcn_filters), requires_grad=True)
# 		self.s_att = Spatial_Attention_layer(self.A.shape[0], in_channels, in_timesteps)
#
# 	def forward(self, x: FloatTensor):
# 		# In : B * V * C_i * T
# 		# Out: B * V * C_o * T
# 		att = self.s_att(x)  # => [B * V * V]
# 		x_out = (att * self.A) @ x.permute(3, 0, 1, 2) @ self.W
# 		return x_out.permute(1, 2, 3, 0)

class GCN(Module):
	def __init__(self, in_channels, gcn_filters, A):
		super(GCN, self).__init__()
		self.A = A
		self.W = Parameter(torch.zeros(in_channels, gcn_filters), requires_grad=True)

	def forward(self, x: FloatTensor, att):
		# In : B * V * C_i * T
		# Out: B * V * C_o * T
		x_out = (att * self.A) @ x.permute(3, 0, 1, 2) @ self.W
		x_out = F.relu(x_out)
		return x_out.permute(1, 2, 3, 0)


class ASTGCNBlock(Module):
	def __init__(self, in_channels, in_timesteps, out_channels, n_vertices, gcn_filters, tcn_strides, **kwargs):
		super(ASTGCNBlock, self).__init__()
		# self.t_att = Attention(n_vertices * gcn_filters, requires_value=True)
		# self.c_att = Attention(n_vertices * in_timesteps, requires_value=True)
		self.SAt = Spatial_Attention_layer(n_vertices, in_channels, in_timesteps)
		self.TAt = Temporal_Attention_layer(n_vertices, in_channels, in_timesteps)

		self.GCN = GCN(in_channels, gcn_filters, kwargs['A'])
		self.time_conv = Conv2d(gcn_filters, out_channels, [1, 3], stride=[1, tcn_strides], padding=[0, 1])
		self.residual_conv = Conv2d(in_channels, out_channels, [1, 1], stride=[1, tcn_strides])
		self.ln = LayerNorm(normalized_shape=out_channels)
	# def forward(self, x: FloatTensor):
	# 	# In : B * V * C_i * T_i
	# 	# Out: B * V * C_o * T_o
	# 	x_res = self.res(x.transpose(1, 2))
	# 	# [B * 1 * C * C] @ [B * V * C * T] => [B * V * C * T]
	# 	x = self.c_att(x.transpose(1, 2)).transpose(1, 2)
	# 	# [B * V * C_i * T] => [B * C_i * V * T] => [B * C' * V * T]
	# 	x = self.gcn(x)
	# 	# [B * 1 * T * T] @ [B * V * T * C] => [B * V * T * C]
	# 	x = self.t_att(x.transpose(1, 3)).transpose(1, 3)
	# 	# [B * C' * V * T] => [B * C_o * V * T]
	# 	x = self.tcn(x.transpose(1, 2)) + x_res
	# 	return self.ln(x.relu_().permute(0, 3, 2, 1)).permute(0, 2, 3, 1)

	def forward(self, x):
		(batch_size, num_of_vertices, num_of_features, num_of_timesteps) = x.shape
		temporal_At = self.TAt(x)  # (B, N, T)
		x_TAt = torch.matmul(x.reshape(batch_size, -1, num_of_timesteps), temporal_At).reshape(batch_size,num_of_vertices,num_of_features,num_of_timesteps)

		spatial_At = self.SAt(x_TAt)  # (B, N, N)
		spatial_gcn = self.GCN(x, spatial_At)  # (B, N, F, T)

		# convolution along time axis
		time_conv_output = (self.time_conv(spatial_gcn.permute(0, 2, 1, 3)).permute(0, 2, 1, 3))  # (B, N, F, T)

		# residual shortcut
		x_residual = (self.residual_conv(x.permute(0, 2, 1, 3)).permute(0, 2, 1, 3))  # (B, N, F, T)

		relued = F.relu(x_residual + time_conv_output)  # (B, N, F, T)
		return self.ln(relued.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)


class ASTGCNSubModule(Module):
	def __init__(self, blocks, **kwargs):
		super(ASTGCNSubModule, self).__init__()
		self.gcn_blocks = Sequential(*[ASTGCNBlock(**block, **kwargs) for block in blocks])
		self.final_conv = Conv2d(in_channels=blocks[-1]['in_timesteps'] // blocks[-1]['tcn_strides'],
								 out_channels=kwargs['n_predictions'],
								 kernel_size=[1, blocks[-1]['gcn_filters']])
		self.A = kwargs["A"]
	def forward(self, x: FloatTensor):
		# In : B * V * C_i * T_i
		# Out: B * V * C_o * T_o
		x = self.gcn_blocks(x)
		x = self.final_conv(x.permute(0, 3, 1, 2))
		# => (B * Tp * V) -> (B * V * Tp)
		return x[..., 0].transpose(1, 2)


class ASTGCN(Module):
	def __init__(self, submodules, mixin):
		super(ASTGCN, self).__init__()
		self.submodules = ModuleList([ASTGCNSubModule(**submodules[i], **mixin[i]) for i in range(len(submodules))])   # 这个邻接矩阵变一变
		self.W = Parameter(torch.zeros(len(submodules), mixin[0]['n_vertices'], mixin[0]['n_predictions']),
						   requires_grad=True)

	def forward(self, X):
		batch = X.size(0)  # batch: 32
		stock = X.size(1)  # stock: 758

		X = X.transpose(2, 3)

		out = sum(map(lambda fn, w: fn(X) * w, self.submodules, self.W))
		out = torch.reshape(out, (batch * stock, -1))

		return out
