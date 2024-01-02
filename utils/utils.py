import matplotlib.pyplot as plt

def visualize(input_data,num_samples): #input_data: a list of loaded numpy data, num_samples: the length of the list
    row = 0
    for sample in input_data:
        for i in range(sample.shape[0]):
            plt.subplot(num_samples,sample.shape[0], row*sample.shape[0]+i+1)
            plt.imshow(sample[i].reshape(28,28), cmap='gray')
            plt.axis('off')
        row += 1
    plt.show()