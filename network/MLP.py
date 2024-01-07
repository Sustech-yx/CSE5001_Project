import torch
from torch import nn, optim, autograd
import numpy as npy


class MLP(nn.Module):
	def __init__(self, hidden_dim=784) -> None:
		super(MLP, self).__init__()
		lin1 = nn.Linear(10 * 28 * 28, hidden_dim)
		lin2 = nn.Linear(hidden_dim, hidden_dim)
		lin3 = nn.Linear(hidden_dim, hidden_dim)
		lin4 = nn.Linear(hidden_dim, 10)
		dropout = nn.Dropout(0.3)
		for lin in [lin1, lin2, lin3, lin4]:
			nn.init.xavier_uniform_(lin.weight)
			nn.init.zeros_(lin.bias)
		self._main = nn.Sequential(dropout, lin1, nn.ReLU(True), lin2, nn.ReLU(True), lin3, nn.ReLU(True), lin4)

	def forward(self, input):
		out = input.view(input.shape[0], 10*28*28)
		out = self._main(out)
		return out
