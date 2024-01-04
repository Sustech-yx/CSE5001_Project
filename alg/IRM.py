# https://github.com/facebookresearch/InvariantRiskMinimization/blob/main/code/colored_mnist/main.py
from alg.base import BaseAlg
import torch
from torch import nn, optim, autograd
from network.MLP import MLP
from tqdm import tqdm


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
	def __init__(self, penalty_weight=91257, steps=501, penalty_anneal_iters=190, learning_rate=0.001):
		self.lr = learning_rate
		self.model = MLP()
		self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
		self.penalty_weight = penalty_weight
		self.steps = steps
		self.diff = 5
		self.l2_regularizer_weight = 0.001
		self.penalty_anneal_iter = penalty_anneal_iters
		self.count = 0

	def mean_nll(logits, y):
		return nn.functional.binary_cross_entropy_with_logits(logits, y)

	def mean_accuracy(logits, y):
		preds = (logits > 0.).float()
		return ((preds - y).abs() < 1e-2).float().mean()

	def penalty(logits, y):
		scale = torch.tensor(1.).requires_grad_()
		loss = mean_nll(logits * scale, y)
		grad = autograd.grad(loss, [scale], create_graph=True)[0]
		return torch.sum(grad**2)

	def train(self, train_loader):
		self.model.train()
		self.count += 1
		for index, (data, labels) in tqdm(enumerate(train_loader)):
			if index % 2 == 0:
				outputs1 = self.model(data)
				labels1 = labels
				self.optimizer.step()
			elif index % 2 == 1:
				temp = torch.cat((torch.arange(data.size(1) - self.diff, data.size(1)), torch.arange(0, data.size(1) - self.diff)))
				data = data[:, temp, :, :]
				labels2 = labels
				outputs2 = self.model(data)
				train_nll = torch.cat((mean_nll(outputs1, labels1), mean_nll(outputs2, labels2))).mean()
				train_acc = torch.cat((mean_accuracy(outputs1, labels1), mean_accuracy(outputs2, labels2))).mean()
				train_penalty = torch.cat((penalty(outputs1, labels1), penalty(outputs2, labels2))).mean()
				weight_norm = torch.tensor(0.).cuda()
				for w in self.model.parameters():
					weight_norm += w.norm().pow(2)
				
				loss = train_nll.clone()
				loss += self.l2_regularizer_weight * weight_norm
				penalty_weight = (self.penalty_weight 
					if self.count >= self.penalty_anneal_iters else 1.0)
				loss += penalty_weight * train_penalty
				if penalty_weight > 1.0:
				# Rescale the entire loss to keep gradients in a reasonable range
					loss /= penalty_weight

				self.optimizer.zero_grad()
				loss.backward()
				self.optimizer.step()
		if self.count % 100 == 0:
			print("Train accuracy: ", train_acc)

	def predict(self, data_loader):
		pass

	def evaluate(self, test_loader):
		pass
	
	def save(self, path):
		pass
	
	def load(self, path):
		pass
	pass

