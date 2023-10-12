# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# %%
# URL to the Parquet file
url = "https://github.com/PlayinatorIX/intro/raw/main/fashion_mnist.parquet"

# Download the Parquet file and create a DataFrame
df = pd.read_parquet(url)

# Work with the DataFrame (e.g., display the first few rows)
print(df.head())

# %%
class dataloader:
    def __init__(self, df):
        self.labels = None
        self.images = None
        self.df = df
        self.load_images_and_labels()

    def load_images_and_labels(self):
        self.labels = self.df.iloc[:, 1:].values
        self.images = self.df['flattened_images'].values

    def splot(self, num_samples=5):
        if self.images is not None:
            indices = np.random.choice(len(self.images), num_samples, replace=False)
            sample_images = [self.images[i] for i in indices]
            sample_labels = [self.labels[i] for i in indices]

            fig, axes = plt.subplots(1, num_samples, figsize=(15, 3))
            for i in range(num_samples):
                axes[i].imshow(sample_images[i].reshape(28, 28), cmap='gray')
                axes[i].set_title(f"Label: {np.argmax(sample_labels[i])}")

            plt.show()
        else:
            print("Images and labels not loaded. Call load_images_and_labels first.")

# %%
df_to_dl = dataloader(df)
# %%
print(df_to_dl)
# %%
df_to_dl.splot()
# %%
