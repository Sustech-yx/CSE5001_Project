# CSE5001_Project
A Robust Handwritten Digit Recognition Network.

## Start Up

First create a conda virtual environment with the following script.

```shell
CONDA_RESTORE_FREE_CHANNEL=1 conda env create -f environment.yml
```

or

```shell
conda env create -f environment4mac.yml
```

## Develop

### Step 1

Create or reuse a net work in `./network`.

### Step 2

Implement your algorithm in `./alg`, with the base class `class BaseAlg(ABC):`.

``` python
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
```

### Step 3

Evaluate your algorithm in `main.ipynb`.

## TODO

- [-] Implement visualized components.

- [ ] Try different algorithms.
