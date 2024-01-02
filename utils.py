import torch
import matplotlib.pyplot as plt

def reduce(batch_data):
    result_list = []
    for batch in batch_data:
        for index, channel in enumerate(batch):
            if torch.sum(channel) != 0:
                break
        result_list.append(batch[index])
    batch_data = torch.stack(result_list)
    return batch_data.unsqueeze(1)


def visial(data, label):        
    for index, channel in enumerate(data):
        if torch.sum(channel) != 0:
            break
        
    data = data[index]
    plt.imshow(data.reshape(28,28), cmap='gray')
    plt.axis('off')
    plt.show()