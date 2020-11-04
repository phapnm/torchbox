import os
import torch
import random
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from sklearn.model_selection import train_test_split
		
# define a data class
class ClassificationDataset(Dataset):
	def __init__(self, data, data_path, transform, training = True):
		"""Define the dataset for classification problems
		Args:
			data ([dataframe]): [a dataframe that contain 2 columns: image name and label]
			data_path ([str]): [path/to/folder that contains image file]
			transform : [augmentation methods and transformation of images]
			training (bool, optional): []. Defaults to True.
		"""

		self.data = data
		self.imgs = data["file_name"].unique().tolist()
		self.data_path = data_path
		self.training = training
		self.transform = transform

	def __getitem__(self, idx):
		img = Image.open(os.path.join(self.data_path, self.data.iloc[idx, 0])).convert('RGB')
		label = self.data.iloc[idx, 1]
		# label = torch.tensor(label, dtype=torch.int8)

		if self.transform is not None:
			img = self.transform(img)
		
		return img, label

	def __len__(self):
		return len(self.imgs)
