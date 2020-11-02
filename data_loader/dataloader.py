import os
import torch
import numpy as np
import pandas as pd
from data_loader import transform
import utils
from sklearn.model_selection import train_test_split

def data_split(data, test_size):
	x_train, x_test, y_train, y_test = train_test_split(data, data["label"], test_size=test_size)
	return x_train, x_test, y_train, y_test

def get_data_loader(cfg):

	data = cfg["data"]["data_csv_name"]
	valid_data = cfg["data"]["validation_csv_name"]
	test_data = cfg["data"]["test_csv_name"]
	data_path = cfg["data"]["data_path"]
	
	train_set = pd.read_csv(data)
	test_set = pd.read_csv(test_data)

	if (valid_data == ""):
		print("No validation set available, auto split the training into validation")
		print("Splitting dataset into train and valid....")
		split_ratio = float(cfg["data"]["validation_ratio"])
		train_set, valid_set, _ , _ = data_split(train_set, split_ratio)
		print("Done Splitting !!!")
	else:
		print("Creating validation set from file")
		print("Reading validation data from file: ", valid_data)
		valid_set = pd.read_csv(valid_data)
	
	# Get Custom Dataset inherit from torch.utils.data.Dataset
	dataset, mod, dataset_name = utils.general.get_attr_by_name(cfg["data"]["data.class"])
	# Create Dataset
	batch_size = int(cfg["data"]["batch_size"])
	train_set = dataset(train_set, data_path, transform.train_transform)
	valid_set = dataset(valid_set, data_path, transform.val_transform)
	test_set = dataset(test_set, data_path, transform.val_transform)
	# DataLoader
	train_loader = torch.utils.data.DataLoader(
		train_set, batch_size=batch_size, shuffle=True
    )
	valid_loader = torch.utils.data.DataLoader(
        valid_set, batch_size=batch_size, shuffle=False
    )
	test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=32, shuffle=False
    )
	return train_loader, valid_loader, test_loader