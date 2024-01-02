from abc import ABC, abstractmethod


class BaseAlg(ABC):
	@abstractmethod
	def train(self, train_loader):
		"""
		训练算法
		:param train_loader: 训练数据加载器
		"""
		raise NotImplementedError

	@abstractmethod
	def predict(self, data_loader):
		"""
		使用训练好的算法进行预测
		:param test_loader: 测试数据加载器
		:return: 预测结果
		"""
		raise NotImplementedError

	@abstractmethod
	def evaluate(self, test_loader):
		"""
		评估算法性能
		:param test_loader: 测试数据加载器
		:return: 算法性能指标
		"""
		raise NotImplementedError
	
	@abstractmethod
	def save(self, path):
		"""
		保存模型文件
		:param path: 保存路径
		"""
		raise NotImplementedError
	
	@abstractmethod
	def load(self, path):
		"""
		将模型文件导入
		:param path: 保存路径
		"""
		raise NotImplementedError