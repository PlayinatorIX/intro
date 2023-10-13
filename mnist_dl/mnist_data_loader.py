#%%
import gzip
import numpy as np
import matplotlib.pyplot as plt

#%%
class DataLoader:
    def __init__(self, *args):
        self.images = []
        self.labels = []

        for i in range(0, len(args), 2):
            images_path = args[i]
            labels_path = args[i + 1]

            if images_path and labels_path:
                images, labels = self.load_images_and_labels(images_path, labels_path)
                self.images.append(images)
                self.labels.append(labels)

        self.images = np.concatenate(self.images, axis=0)
        self.labels = np.concatenate(self.labels, axis=0)

    def load_images_and_labels(self, images_path, labels_path):
        # Check if the images file ends with .gz and unzip if necessary
        if images_path.endswith('.gz'):
            with gzip.open(images_path, 'rb') as f:
                images = np.frombuffer(f.read(), dtype=np.uint8, offset=16).reshape(-1, 28, 28)

        # Check if the labels file ends with .gz and unzip if necessary
        if labels_path.endswith('.gz'):
            with gzip.open(labels_path, 'rb') as f:
                labels = np.frombuffer(f.read(), dtype=np.uint8, offset=8)

        return images, labels
    
    def size(self):
            return len(self.images)

    def mat_sample(self, num_samples=5):
        indices = np.random.choice(len(self.images), num_samples, replace=False)
        sampled_images = self.images[indices]
        sampled_labels = self.labels[indices]
        return sampled_images, sampled_labels
    
    def plot_sample(self, num_samples=5):
        indices = np.random.choice(len(self.images), num_samples, replace=False)
        sampled_images = self.images[indices]
        sampled_labels = self.labels[indices]

        for i in range(num_samples):
            plt.subplot(1, num_samples, i + 1)
            plt.imshow(sampled_images[i], cmap='gray')
            plt.title(str(sampled_labels[i]))
            plt.axis('off')
        plt.show()
    
#%%
# Shortened file paths
train_images_path = 'mnist/train-images-idx3-ubyte.gz'
train_labels_path = 'mnist/train-labels-idx1-ubyte.gz'
test_images_path = 'mnist/t10k-images-idx3-ubyte.gz'
test_labels_path = 'mnist/t10k-labels-idx1-ubyte.gz'

# Example usage with the shortened paths
mnist_data_loader = DataLoader(train_images_path, train_labels_path, test_images_path, test_labels_path)

# %%
# Print the size of the DataLoader
print("DataLoader size:", mnist_data_loader.size())

# %%
# Sample from the DataLoader
sampled_images, sampled_labels = mnist_data_loader.mat_sample(num_samples=5)
# Print the sampled data
for i in range(len(sampled_images)):
    print(f"Sampled Image {i}:\n{sampled_images[i]}\nSampled Label {i}:\n{sampled_labels[i]}\n")
# %%
mnist_data_loader.plot_sample(num_samples=5)
# %%
train_images_path = 'fashion_mnist/train-images-idx3-ubyte.gz'
train_labels_path = 'fashion_mnist/train-labels-idx1-ubyte.gz'
test_images_path = 'fashion_mnist/t10k-images-idx3-ubyte.gz'
test_labels_path = 'fashion_mnist/t10k-labels-idx1-ubyte.gz'
# %%
fashion_data_loader = DataLoader(train_images_path, train_labels_path, test_images_path, test_labels_path)
# %%
# Print the size of the DataLoader
print("DataLoader size:", fashion_data_loader.size())
# %%
# Sample from the DataLoader
sampled_images, sampled_labels = fashion_data_loader.mat_sample(num_samples=5)
# Print the sampled data
for i in range(len(sampled_images)):
    print(f"Sampled Image {i}:\n{sampled_images[i]}\nSampled Label {i}:\n{sampled_labels[i]}\n")
# %%
fashion_data_loader.plot_sample(num_samples=5)
# %%
