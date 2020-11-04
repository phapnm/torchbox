import os
import torch
import importlib
import pandas as pd
from utils import custom_loss
import data_loader
from data_loader import transform
from data_loader import dataloader

    
def model_loader(config):
    model_dict = config.get('model')
    mod_name, func_name = model_dict['model.class'].rsplit('.', 1)
    mod = importlib.import_module(mod_name)
    func = getattr(mod, func_name)
    # removed_value = model_dict.pop('model.class', 'No Key found')
    return func(**model_dict)

def get_attr_by_name(func_str):
    """
    Load function by full name
    :param func_str:
    :return: fn, mod
    """
    mod_name, func_name = func_str.rsplit('.', 1)
    mod = importlib.import_module(mod_name)
    func = getattr(mod, func_name)
    return func, mod, func_name

def get_optimizer(config):
    optimizer_name = config["optimizer"]["name"]
    optimizer = getattr(torch.optim, optimizer_name, "The optimizer {} is not available".format(optimizer_name))
    return optimizer

def get_loss_fn(config):
    loss_function = config["train"]["loss"]
    try:
        # if the loss function comes from nn package
        criterion = getattr(torch.nn, loss_function, "The loss {} is not available".format(loss_function))
    except:
        # use custom loss
        criterion = getattr(custom_loss, loss_function, "The loss {} is not available".format(loss_function))
    return criterion

def make_dir_epoch_time(base_path, session_name, time_str):
    """
    make a new dir on base_path with epoch_time
    :param base_path:
    :return:
    """
    new_path = os.path.join(base_path, session_name + "_" + time_str)
    os.makedirs(new_path, exist_ok=True)
    return new_path
