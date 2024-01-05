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
	# print(y.type(), logits.type())
	return nn.CrossEntropyLoss()(logits, y)(logits.float(), y.float())

def mean_accuracy(logits, y):
	preds = (logits > 0.).float()
	return ((preds - y).abs() < 1e-2).float().mean()

def penalty(logits, y):
	scale = torch.tensor(1.).requires_grad_()
	loss = mean_nll(logits * scale, y)
	grad = autograd.grad(loss, [scale], create_graph=True)[0]
	return torch.sum(grad**2)


class IRM_MLP(BaseAlg):
	def __init__(self, penalty_weight=10000, steps=501, penalty_anneal_iters=190, learning_rate=0.0002):
		self.lr = learning_rate
		self.model = MLP()
		self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
		self.penalty_weight = penalty_weight
		self.steps = steps
		self.diff = 5
		self.l2_regularizer_weight = 0.001
		self.penalty_anneal_iters = penalty_anneal_iters
		self.count = 0
		# torch.autograd.set_detect_anomaly(True)

	def train(self, train_loader):
		# self.model.train()
		# self.count += 1

		# temp = torch.cat((torch.arange(data.size(1) - self.diff, data.size(1)), torch.arange(0, data.size(1) - self.diff)))
		# data = data[:, temp, :, :]
		# labels2 = labels.clone().float()
		# outputs2 = self.model(data)
		# outputs2 = torch.squeeze(outputs2).float()
		# # print(outputs1.shape, labels1.shape)
		# # print(mean_nll(outputs1, labels1).shape, mean_nll(outputs2, labels2).shape)
		# train_nll = torch.stack((mean_nll(outputs1, labels1), mean_nll(outputs2, labels2))).mean()
		# train_acc = torch.stack((mean_accuracy(outputs1, labels1), mean_accuracy(outputs2, labels2))).mean()
		# train_penalty = torch.stack((penalty(outputs1, labels1), penalty(outputs2, labels2))).mean()
		# weight_norm = torch.tensor(0.)#.cuda()
		# for w in self.model.parameters():
		# 	weight_norm += w.norm().pow(2)
		
		# loss = train_nll.clone()
		# loss += self.l2_regularizer_weight * weight_norm
		# penalty_weight = (self.penalty_weight 
		# 	if self.count >= self.penalty_anneal_iters else 1.0)
		# loss += penalty_weight * train_penalty
		# if penalty_weight > 1.0:
		# # Rescale the entire loss to keep gradients in a reasonable range
		# 	loss /= penalty_weight

		# self.optimizer.zero_grad()
		# loss.backward()
		# self.optimizer.step()
		# if self.count % 100 == 0:
		# 	print("Train accuracy: ", train_acc)
		# # return outputs1
		pass

	def train_with_eval(self, train_loader1, train_loader2, val_loader):
		self.count += 1
		self.model.train()
		for (data1, label1), (data2, label2) in zip(train_loader1, train_loader2):
			# 准备两个环境，分别以0.1的概率、0.2的概率通过随机打乱移除标签和特征所在维度的关系
			# temp: 用于打乱第二个维度 [batch_size, 10, 28, 28] 的indices
			temp = torch.randperm(data1.size(1))
			# 只打乱前10%的数据
			num_samples = int(data1.size(0) * 0.1)
			data1[:num_samples] = data1[:num_samples, temp, :, :]
			# 为了保证是打乱的随机效果，再在第一个维度进行打乱，意图是让被打算的数据均匀分布在batch中
			random_indices = torch.randperm(data1.size(0))
			data1 = data1[random_indices]
			# 得到模型输出
			output1 = self.model(data1).squeeze()
			_, logits1 = torch.max(output1.data, 1)

			# 与env1相似，只是打散的比例为20%
			temp = torch.randperm(data2.size(1))
			num_samples = int(data2.size(0) * 0.2)
			data2[:num_samples] = data2[:num_samples, temp, :, :]
			random_indices = torch.randperm(data2.size(0))
			data2 = data2[random_indices]
			output2 = self.model(data2).squeeze()
			_, logits2 = torch.max(output2.data, 1)
			# 剩下这些就是照搬提供的代码了
			train_nll = torch.stack([mean_nll(logits1, label1), mean_nll(logits2, label2)]).mean()
			train_acc = torch.stack([mean_accuracy(logits1, label1), mean_accuracy(logits2, label2)]).mean()
			train_penalty = torch.stack([penalty(logits1, label1), penalty(logits2, label2)]).mean()

			weight_norm = torch.tensor(0.)#.cuda()
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
		# print(logits1, label1, logits2, label2)
		# print("Train accuracy: ", float(train_acc), ". Loss: ", float(loss))
		

	def predict(self, data_loader):
		self.model.eval()
		predictions = []
		with torch.no_grad():
			for data, _ in data_loader:
				predicted = self.model(data)
				_, predicted = torch.max(predicted.data, 1)
				predictions.extend(predicted.cpu().numpy())
		return predictions

	def evaluate(self, test_loader):
		self.model.eval()
		correct = 0
		total = 0
		wrong_pred = ([], [], [])
		correct_pred = ([], [], [])
		with torch.no_grad():
			for data, labels in test_loader:
				# temp = torch.randperm(data.size(1))
				# num_samples = int(data.size(0) * 0.9)
				# data[:num_samples] = data[:num_samples, temp, :, :]
				# random_indices = torch.randperm(data.size(0))
				# data = data[random_indices]
				predicted = self.model(data)#.squeeze()
				_, predicted = torch.max(predicted.data, 1)
				# print(predicted, labels)
				total += labels.size(0)
				correct += (predicted == labels).sum().item()
				wrong_pred[0].append(data[predicted != labels])
				wrong_pred[1].append(predicted[predicted != labels])
				wrong_pred[2].append(labels[predicted != labels])
				correct_pred[0].append(data[predicted==labels])
				correct_pred[1].append(predicted[predicted==labels])
				correct_pred[2].append(labels[predicted==labels])
		accuracy = correct / total
		# print(correct, total)
		print('Accuracy on the val set: %d %%' % (100 * correct / total))
		return wrong_pred, accuracy, correct_pred
	
	def save(self, path):
		torch.save(self.model.state_dict(), path)

	def load(self, path):
		self.model.load_state_dict(torch.load(path))

from network.CNN import CNN1

class IRM_CNN(BaseAlg):
	def __init__(self, penalty_weight=10000, steps=53, penalty_anneal_iters=19, learning_rate=0.0002):
		raise ValueError("This class is deprecated.")
		self.lr = learning_rate
		self.model = CNN1()
		self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
		self.penalty_weight = penalty_weight
		self.steps = steps
		self.diff = 5
		self.l2_regularizer_weight = 0.001
		self.penalty_anneal_iters = penalty_anneal_iters
		self.count = 0
		# torch.autograd.set_detect_anomaly(True)

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
		# self.model.train()
		# self.count += 1

		# temp = torch.cat((torch.arange(data.size(1) - self.diff, data.size(1)), torch.arange(0, data.size(1) - self.diff)))
		# data = data[:, temp, :, :]
		# labels2 = labels.clone().float()
		# outputs2 = self.model(data)
		# outputs2 = torch.squeeze(outputs2).float()
		# # print(outputs1.shape, labels1.shape)
		# # print(mean_nll(outputs1, labels1).shape, mean_nll(outputs2, labels2).shape)
		# train_nll = torch.stack((mean_nll(outputs1, labels1), mean_nll(outputs2, labels2))).mean()
		# train_acc = torch.stack((mean_accuracy(outputs1, labels1), mean_accuracy(outputs2, labels2))).mean()
		# train_penalty = torch.stack((penalty(outputs1, labels1), penalty(outputs2, labels2))).mean()
		# weight_norm = torch.tensor(0.)#.cuda()
		# for w in self.model.parameters():
		# 	weight_norm += w.norm().pow(2)
		
		# loss = train_nll.clone()
		# loss += self.l2_regularizer_weight * weight_norm
		# penalty_weight = (self.penalty_weight 
		# 	if self.count >= self.penalty_anneal_iters else 1.0)
		# loss += penalty_weight * train_penalty
		# if penalty_weight > 1.0:
		# # Rescale the entire loss to keep gradients in a reasonable range
		# 	loss /= penalty_weight

		# self.optimizer.zero_grad()
		# loss.backward()
		# self.optimizer.step()
		# if self.count % 100 == 0:
		# 	print("Train accuracy: ", train_acc)
		# # return outputs1
		pass

	def train_with_eval(self, train_loader1, train_loader2, val_loader):
		self.count += 1
		for (data1, label1), (data2, label2) in zip(train_loader1, train_loader2):
			output1 = self.model(data1).squeeze()
			_, logits1 = torch.max(output1.data, 1)
			# print(logits1, label1)
			temp = torch.cat((torch.arange(data2.size(1) - self.diff, data2.size(1)), torch.arange(0, data2.size(1) - self.diff)))
			data2 = data2[:, temp, :, :]
			output2 = self.model(data2).squeeze()
			_, logits2 = torch.max(output2.data, 1)
			train_nll = torch.stack([mean_nll(logits1, label1), mean_nll(logits2, label2)]).mean()
			train_acc = torch.stack([mean_accuracy(logits1, label1), mean_accuracy(logits2, label2)]).mean()
			train_penalty = torch.stack([penalty(logits1, label1), penalty(logits2, label2)]).mean()

			weight_norm = torch.tensor(0.)#.cuda()
			for w in self.model.parameters():
				weight_norm += w.norm().pow(2)

			loss = train_nll.clone()
			loss += self.l2_regularizer_weight * weight_norm
			penalty_weight = (self.penalty_weight 
				if self.count >= self.penalty_anneal_iters else 1.0)
			# print(loss, penalty_weight * train_penalty)
			loss += penalty_weight * train_penalty
			if penalty_weight > 1.0:
			# Rescale the entire loss to keep gradients in a reasonable range
				loss /= penalty_weight

			self.optimizer.zero_grad()
			loss.backward()
			self.optimizer.step()
		print("Train accuracy: ", float(train_acc), ". Loss: ", float(loss))

	def predict(self, data_loader):
		self.model.eval()
		predictions = []
		with torch.no_grad():
			for data, _ in data_loader:
				predicted = self.model(data)
				_, predicted = torch.max(predicted.data, 1)
				predictions.extend(predicted.cpu().numpy())
		return predictions

	def evaluate(self, test_loader):
		self.model.eval()
		correct = 0
		total = 0
		wrong_pred = ([], [], [])
		correct_pred = ([], [], [])
		with torch.no_grad():
			for data, labels in test_loader:
				predicted = self.model(data)
				_, predicted = torch.max(predicted.data, 1)
				# print(predicted, labels)
				total += labels.size(0)
				correct += (predicted == labels).sum().item()
				wrong_pred[0].append(data[predicted != labels])
				wrong_pred[1].append(predicted[predicted != labels])
				wrong_pred[2].append(labels[predicted != labels])
				correct_pred[0].append(data[predicted==labels])
				correct_pred[1].append(predicted[predicted==labels])
				correct_pred[2].append(labels[predicted==labels])
		accuracy = correct / total
		# print(correct, total)
		print('Accuracy on the val set: %d %%' % (100 * correct / total))
		return wrong_pred, accuracy,correct_pred
	
	def save(self, path):
		torch.save(self.model.state_dict(), path)

	def load(self, path):
		self.model.load_state_dict(torch.load(path))
