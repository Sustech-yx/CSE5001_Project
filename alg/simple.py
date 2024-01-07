import torch
from data.dataset import MNISTDataset
from network.CNN import CNN
from alg.base import BaseAlg
from tqdm import tqdm
from utils import reduce


class SimpleAlgorithm(BaseAlg):
	def __init__(self):
		self.model = CNN()
		self.criterion = torch.nn.CrossEntropyLoss()
		self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

	def train(self, train_loader):
		self.model.train()
		los = []
		for data, labels in tqdm(train_loader):
			data = reduce(data)
			# print(data.shape)
			self.optimizer.zero_grad()
			outputs = self.model(data)
			loss = self.criterion(outputs, labels)
			los.append(loss)
			loss.backward()
			self.optimizer.step()
		return los

	def predict(self, val_loader):
		self.model.eval()
		predictions = []
		with torch.no_grad():
			for data, file_path in val_loader:
				data = reduce(data)
				outputs = self.model(data)
				_, predicted = torch.max(outputs.data, 1)
				predictions.extend((file_path, predicted.cpu().numpy()))
		return predictions

	def evaluate(self, test_loader):
		self.model.eval()
		correct = 0
		total = 0
		wrong_pred = ([], [], [])
		correct_pred = ([], [], [])
		with torch.no_grad():
			for data, labels in test_loader:
				data = reduce(data)
				outputs = self.model(data)
				_, predicted = torch.max(outputs.data, 1)
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