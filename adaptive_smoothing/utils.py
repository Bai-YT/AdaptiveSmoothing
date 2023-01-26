import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from collections import OrderedDict
import numpy as np
import os, sys
par_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(par_path)
root_path = os.path.dirname(os.getcwd())

from models import resnet, bit_rn, dm_rn
from models.comp_model import CompositeModel


def get_cmodel_and_loaders(forward_settings, std_load_path, adv_load_path, dataset_path, 
                           dataset_name="CIFAR-10", batch_size_train=500, batch_size_test=500, 
                           use_data_aug=True, train_with_test=False, train_shuffle=True):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    std_model_type = forward_settings["std_model_type"]
    adv_model_type = forward_settings["adv_model_type"]

    # Data transforms
    transform_test = transforms.Compose([transforms.ToTensor()])
    if use_data_aug:
        transform_train = transforms.Compose([transforms.ToTensor(),
                                            transforms.RandomHorizontalFlip(p=.5),
                                            transforms.RandomCrop(32, padding=4)])
    else:
        transform_train = transforms.Compose([transforms.ToTensor()])

    # Datasets
    if dataset_name == "CIFAR-10":
        train_set = datasets.CIFAR10(root=dataset_path, train=(not train_with_test),
                                    download=True, transform=transform_train)
        test_set = datasets.CIFAR10(root=dataset_path, train=False, download=True, transform=transform_test)
    elif dataset_name == "CIFAR-100":
        train_set = datasets.CIFAR100(root=dataset_path, train=(not train_with_test),
                                     download=True, transform=transform_train)
        test_set = datasets.CIFAR100(root=dataset_path, train=False, download=True, transform=transform_test)
    else:
        raise ValueError('Unknown dataset name.')

    # Data loaders
    trainloader = DataLoader(train_set, batch_size=batch_size_train, pin_memory=True, 
                             shuffle=train_shuffle, num_workers=4, drop_last=True)
    trainloader_fast = DataLoader(train_set, batch_size=batch_size_train * 9, pin_memory=True, 
                                  shuffle=train_shuffle, num_workers=4, drop_last=True)
    testloader = DataLoader(test_set, batch_size=batch_size_test, pin_memory=True, 
                            shuffle=False, num_workers=4, drop_last=False)
    
    # Standard base model
    assert std_model_type in ["rn18", 'rn50', 'rn152']
    if std_model_type in ['rn50', 'rn152']:
        model_name = "BiT-M-R50x1" if std_model_type == 'rn50' else "BiT-M-R152x2"
        std_model = bit_rn.KNOWN_MODELS[model_name](head_size=len(test_set.classes), zero_head=False)
        try:
            std_model.load_from(np.load(std_load_path))
        except:
            std_model = nn.DataParallel(std_model)
            std_model.load_state_dict(torch.load(std_load_path, map_location=device)["model"])
            std_model = std_model.module
    else:
        std_model = resnet.ResNet18()
        std_model.load_state_dict(torch.load(std_load_path))

    if forward_settings['parallel'] == 1:
        std_model = nn.DataParallel(std_model)
    std_model.eval()
    for param in std_model.parameters():
        param.requires_grad = False

    # Robust base model
    assert adv_model_type in ["rn18", "wrn", 'wrn7016']
    if adv_model_type == 'wrn7016':
        adv_model = dm_rn.WideResNet(num_classes=len(test_set.classes), activation_fn=dm_rn.Swish, 
                                     depth=70, width=16, mean=dm_rn.CIFAR100_MEAN, std=dm_rn.CIFAR100_STD)
    else:
        adv_model = resnet.ResNet18() if adv_model_type == "rn18" else resnet.WideResNet()

    adv_model.load_state_dict(torch.load(adv_load_path))
    if forward_settings['parallel'] == 1:
        adv_model = nn.DataParallel(adv_model)
    adv_model.eval()
    for param in adv_model.parameters():
        param.requires_grad = False

    # Comp model
    comp_model = CompositeModel([std_model, adv_model], forward_settings=forward_settings).to(device)
    return comp_model, trainloader, testloader, trainloader_fast, transform_train, transform_test


def load_ckpt(comp_model, optimizer, scheduler, warmup_scheduler, lr, load_path, parallel,
              enable_BN=False, reset_scheduler=False, device='cpu'):
    print(f"Loading checkpoint {load_path}.")
    state_dict = torch.load(load_path, map_location=device)

    if "bn.running_mean" in state_dict["model"].keys():
        state_dict["bn"] = OrderedDict()
        for key in ["running_mean", "running_var", "num_batches_tracked"]:
            state_dict["bn"][key] = state_dict["model"][f"bn.{key}"]
            del state_dict["model"][f"bn.{key}"]

    lsd_model = comp_model.policy_net.module if parallel == 1 else(
        comp_model.module.policy_net if parallel == 2 else comp_model.policy_net)
    lsd_model.load_state_dict(state_dict["model"])

    try:
        cur_bn = comp_model.module.bn if parallel == 2 else comp_model.bn
        cur_bn.load_state_dict(state_dict["bn"])
    except:
        comp_model.bn = state_dict["bn"]

    try:
        optimizer.load_state_dict(state_dict["optimizer"])
        print("Successfully loaded the optimizer state_dict.")
    except:
        print("Something went wrong or loading the optimizer!!!")
        print("The optimizer state_dict is not loaded!!!!!\n")

    if "scheduler" in state_dict.keys() and not reset_scheduler:
        scheduler.load_state_dict(state_dict["scheduler"])
        print("Successfully loaded the scheduler state_dict.")
        if "warmup_scheduler" in state_dict.keys():
            warmup_scheduler.load_state_dict(state_dict["warm_scheduler"])
            print("Successfully loaded the warmup scheduler state_dict.")
    else:
        for g in optimizer.param_groups:
            g['lr'] = lr
        print("The scheduler has been reset.")
    ep_start = 1 if reset_scheduler else state_dict["ep"] + 1
    
    # Enable BN affine
    if enable_BN:
        comp_model.policy_net.module.bn.affine = True
        comp_model.policy_net.module.bn.weight = nn.parameter.Parameter(torch.empty((1,), device=device))
        comp_model.policy_net.module.bn.bias = nn.parameter.Parameter(torch.empty((1,), device=device))
        comp_model.policy_net.module.bn.reset_parameters()
        optimizer.add_param_group({'params': comp_model.policy_net.module.bn.parameters()})

    return ep_start, state_dict["ba"], state_dict["img_inds"]
