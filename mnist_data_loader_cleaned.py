import gzip
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class DataLoader:
    def __init__(self, *args):
        self.images = []
        self.labels = []
        self.load_data_from_args(*args)
        self.images = np.concatenate(self.images, axis=0)
        self.labels = np.concatenate(self.labels, axis=0)


    def load_data_from_args(self, *args):
        #Load data from file paths provided in pairs (images, labels).
        for i in range(0, len(args), 2):
            images_path = args[i]
            labels_path = args[i + 1]
            if images_path and labels_path:
                images, labels = self.load_images_and_labels(images_path, labels_path)
                self.images.append(images)
                self.labels.append(labels)


    def load_images_and_labels(self, images_path, labels_path):
        #Load image and label data from provided paths.
        if images_path.endswith('.gz'):
            with gzip.open(images_path, 'rb') as f:
                images = np.frombuffer(f.read(), dtype=np.uint8, offset=16).reshape(-1, 28, 28)
        if labels_path.endswith('.gz'):
            with gzip.open(labels_path, 'rb') as f:
                labels = np.frombuffer(f.read(), dtype=np.uint8, offset=8)
        return images, labels


    def size(self):
        #Get the total number of images in the dataset.
        return len(self.images)


    def sample(self, num_samples=5):
        #Randomly sample a specified number of images and labels.
        indices = np.random.choice(len(self.images), num_samples, replace=False)
        sampled_images = self.images[indices]
        sampled_labels = self.labels[indices]
        return sampled_images, sampled_labels


    def plot_sample(self, num_samples=5):
        #Plot a random sample of images and their corresponding labels.
        indices = np.random.choice(len(self.images), num_samples, replace=False)
        sampled_images = self.images[indices]
        sampled_labels = self.labels[indices]

        for i in range(num_samples):
            plt.subplot(1, num_samples, i + 1)
            plt.imshow(sampled_images[i], cmap='gray')
            plt.title(str(sampled_labels[i]))
            plt.axis('off')
        plt.show()


    def to_df(self):        
        #Convert image and label data to a Pandas DataFrame.
        return self.create_dataframe(self.preprocess_data(self.clean_data()))


    def clean_data(self):
        #Normalize pixel values to the range [0, 1].
        cleaned_images = self.images / 255.0
        return cleaned_images


    def preprocess_data(self, cleaned_images):        
        #Reshape images and perform one-hot encoding for labels.
        flattened_images = cleaned_images.reshape(cleaned_images.shape[0], -1)
        one_hot_labels = np.eye(len(np.unique(self.labels)))[self.labels]
        return flattened_images, one_hot_labels


    def create_dataframe(self, flattened_images, one_hot_labels):        
        #Create a Pandas DataFrame from flattened images and one-hot encoded labels.
        data = {
            "flattened_images": [img.flatten() for img in flattened_images],
            **{f"label_{i}": one_hot_labels[:, i] for i in range(one_hot_labels.shape[1])}
        }
        df = pd.DataFrame(data)
        return df
