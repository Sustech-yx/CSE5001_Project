from alg.simple import SimpleAlgorithm
from alg.fuzzy import FuzzyAlgorithm
from alg.IRM import IRM_CNN, IRM_MLP
from data.dataset import MNISTDataset
import torch
from torch.utils.data import DataLoader
import numpy as npy
from tqdm import tqdm

import torch
import numpy as np
import random
import os
 
def setup_seed(seed=3407):
    random.seed(seed)  # Python的随机性
    os.environ['PYTHONHASHSEED'] = str(seed)  # 设置Python哈希种子，为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)  # numpy的随机性
    torch.manual_seed(seed)  # torch的CPU随机性，为CPU设置随机种子
    torch.cuda.manual_seed(seed)  # torch的GPU随机性，为当前GPU设置随机种子
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.   torch的GPU随机性，为所有GPU设置随机种子
    torch.backends.cudnn.deterministic = True # 选择确定性算法
    torch.backends.cudnn.benchmark = False # if benchmark=True, deterministic will be False
    torch.backends.cudnn.enabled = False
setup_seed(3407)

train_dataset = MNISTDataset("./processed_data", "train")
val_dataset = MNISTDataset("./processed_data", "val")

train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True, num_workers=4)
val_loader = DataLoader(dataset=val_dataset, batch_size=64, shuffle=True, num_workers=4)

train_dataset4IRM1 = train_dataset[::2]
train_dataset4IRM2 = train_dataset[1::2]
val_dataset4IRM = val_dataset

train_loader4IRM1 = DataLoader(dataset=train_dataset4IRM1, batch_size=64, shuffle=True)
train_loader4IRM2 = DataLoader(dataset=train_dataset4IRM2, batch_size=64, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=64, shuffle=True)
alg_irm = IRM_MLP(penalty_weight=10000)
for epoches in tqdm(range(alg_irm.steps)):
    alg_irm.train_with_eval(train_loader4IRM1, train_loader4IRM2, val_loader)
    if epoches % 100 == 0:
        alg_irm.evaluate(val_loader)

alg_irm.save("./model/irm.pkl")
