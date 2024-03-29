import torch
from data.dataset import MNISTDataset
from network.CNN import CNN1
from alg.base import BaseAlg
from tqdm import tqdm
from utils import reduce
import numpy as npy


class FuzzyAlgorithm(BaseAlg):
    def __init__(self, shuffle=True):
        self.model = CNN1()
        self.shuffle = shuffle
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

    def train(self, train_loader):
        los = []
        self.model.train()
        for data, labels in tqdm(train_loader):
            # data = reduce(data)
            # print(data.shape)
            # print(data.size(1))
            if self.shuffle:
                data=data[:, torch.randperm(data.size(1)), :, :]

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
                # file_path = data[0]
                # data = torch.Tensor(npy.load(file_path))
                outputs = self.model(data)
                _, predicted = torch.max(outputs.data, 1)
                predictions.extend((file_path, predicted.cpu().numpy()))
        return predictions

    def evaluate(self, test_loader):
        self.model.eval()
        correct = 0
        total = 0
        wrong_pred = ([], [], [])
        with torch.no_grad():
            for data, labels in test_loader:
                # data = reduce(data)
                outputs = self.model(data)
                _, predicted = torch.max(outputs.data, 1)
                # print(predicted, labels)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                wrong_pred[0].append(data[predicted != labels])
                wrong_pred[1].append(predicted[predicted != labels])
                wrong_pred[2].append(labels[predicted != labels])
        accuracy = correct / total
        # print(correct, total)
        print('Accuracy on the val set: %d %%' % (100 * correct / total))
        return wrong_pred, accuracy
    
    def save(self, path):
        torch.save(self.model, path)


    def load(self, path):
        self.model.load_state_dict(path)