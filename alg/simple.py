import torch
from data.dataset import MNISTDataset
from network.CNN import CNN
from alg.base import BaseAlg
from tqdm import tqdm


def reduce(batch_data):
	result_list = []
	for batch in batch_data:
		for index, channel in enumerate(batch):
			if torch.sum(channel) != 0:
				break
		result_list.append(batch[index])
	batch_data = torch.stack(result_list)
	return batch_data.unsqueeze(1)


class SimpleAlgorithm(BaseAlg):
	def __init__(self):
		self.model = CNN()
		self.criterion = torch.nn.CrossEntropyLoss()
		self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

	def train(self, train_loader):
		self.model.train()
		for data, labels in tqdm(train_loader):
			data = reduce(data)
			# print(data.shape)
			self.optimizer.zero_grad()
			outputs = self.model(data)
			loss = self.criterion(outputs, labels)
			loss.backward()
			self.optimizer.step()

	def predict(self, val_loader):
		self.model.eval()
		predictions = []
		with torch.no_grad():
			for data, _ in val_loader:
				data = reduce(data)
				outputs = self.model(data)
				_, predicted = torch.max(outputs.data, 1)
				predictions.extend(predicted.cpu().numpy())
		return predictions

	def evaluate(self, test_loader):
		self.model.eval()
		correct = 0
		total = 0
		with torch.no_grad():
			for data, labels in test_loader:
				data = reduce(data)
				outputs = self.model(data)
				_, predicted = torch.max(outputs.data, 1)
				total += labels.size(0)
				correct += (predicted == labels).sum().item()
		accuracy = correct / total
		print(correct, total)
		print('Accuracy on the val set: %d %%' % (100 * correct / total))
		return accuracy
	
	def save(self, path):
		torch.save(self.model, path)


	def load(self, path):
		self.model.load_state_dict(torch.load(path))