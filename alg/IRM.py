# https://github.com/facebookresearch/InvariantRiskMinimization/blob/main/code/colored_mnist/main.py
from alg.base import BaseAlg
import torch
from torch import nn, optim, autograd


def torch_bernoulli(p, size):
	return (torch.rand(size) < p).float()

def torch_xor(a, b):
	return (a-b).abs() # Assumes both inputs are either 0 or 1

def mean_nll(logits, y):
	return nn.functional.binary_cross_entropy_with_logits(logits, y)

def mean_accuracy(logits, y):
	preds = (logits > 0.).float()
	return ((preds - y).abs() < 1e-2).float().mean()

def penalty(logits, y):
	scale = torch.tensor(1.).cuda().requires_grad_()
	loss = mean_nll(logits * scale, y)
	grad = autograd.grad(loss, [scale], create_graph=True)[0]
	return torch.sum(grad**2)


class IRM(BaseAlg):
	def __init__(self, learning_rate=0.001):
		pass

	def train(self, train_loader):
		pass

	def predict(self, data_loader):
		pass

	def evaluate(self, test_loader):
		pass
	
	def save(self, path):
		pass
	
	def load(self, path):
		pass
	pass

