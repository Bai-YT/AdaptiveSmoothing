import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from collections import OrderedDict
import yaml
import random
import numpy as np
import os

from models.base_models import small_rn, trades_wrn, bit_rn, dm_rn
from models.comp_model import CompositeModel


def get_comp_model(
    forward_settings, std_load_path, rob_load_path, num_classes=10
):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    std_model_type = forward_settings["std_model_type"]
    rob_model_type = forward_settings["rob_model_type"]

    # Standard base model
    assert std_model_type in ["rn18", 'rn50', 'rn152']
    print(f"Loading standard model: {std_model_type} from {std_load_path}...")

    if std_model_type in ['rn50', 'rn152']:
        model_name = "BiT-M-R50x1" if std_model_type == 'rn50' else "BiT-M-R152x2"
        std_model = bit_rn.KNOWN_MODELS[model_name](
            head_size=num_classes, zero_head=False
        )
        # Load model from disk
        try:
            std_model.load_from(np.load(std_load_path))
        except:
            std_model = nn.DataParallel(std_model)
            std_model.load_state_dict(
                torch.load(std_load_path, map_location='cpu')["model"]
            )
            std_model = std_model.module
    else:
        std_model = small_rn.ResNet18()
        std_model.load_state_dict(torch.load(std_load_path))

    std_model.eval()
    std_model.requires_grad_(False)

    # Robust base model
    assert rob_model_type in ["rn18", "wrn", 'wrn7016', 'wrn7016_silu']
    print(f"Loading robust model: {rob_model_type} from {rob_load_path}...")

    mean = dm_rn.CIFAR100_MEAN if num_classes == 100 else dm_rn.CIFAR10_MEAN
    std = dm_rn.CIFAR100_STD if num_classes == 100 else dm_rn.CIFAR10_STD
    if rob_model_type == 'wrn7016':
        # Use DeepMind's Swish
        rob_model = dm_rn.WideResNet(
            num_classes=num_classes, activation_fn=dm_rn.Swish,
            depth=70, width=16, mean=mean, std=std
        )
    elif rob_model_type == 'wrn7016_silu':
        # Use PyTorch's SiLU
        rob_model = dm_rn.WideResNet(
            num_classes=num_classes, activation_fn=nn.SiLU, 
            depth=70, width=16, mean=mean, std=std
        )
    elif rob_model_type == "wrn":
        rob_model = trades_wrn.WideResNet()
    else:
        rob_model = small_rn.ResNet18()  # Normalization handled in class init

    state_dict = torch.load(rob_load_path)
    if rob_model_type == 'wrn7016_silu':
        model_state_dict = {}
        for key, val in state_dict["model_state_dict"].items():
            if key.startswith("module.0."):
                model_state_dict[key[9:]] = val
        rob_model.load_state_dict(model_state_dict)
    else:
        rob_model.load_state_dict(state_dict)

    rob_model.eval()
    rob_model.requires_grad_(False)

    # Comp model
    return CompositeModel(
        [std_model, rob_model], forward_settings=forward_settings
    ).to(device)


def get_cmodel_and_loaders(
    forward_settings, std_load_path, rob_load_path, dataset_path,
    dataset_name="CIFAR-10", batch_size_train=500, batch_size_test=500,
    train_with_test=False, train_shuffle=True
):
    # Data transforms
    transform_test = transforms.Compose([transforms.ToTensor()])
    transform_train = transforms.Compose([
        transforms.ToTensor(), transforms.RandomHorizontalFlip(p=.5),
        transforms.RandomCrop(32, padding=4)
    ])

    # Datasets
    if dataset_name == "CIFAR-10":
        train_set = datasets.CIFAR10(
            root=dataset_path, train=(not train_with_test),
            download=True, transform=transform_train
        )
        test_set = datasets.CIFAR10(
            root=dataset_path, train=False, download=True, transform=transform_test
        )
    elif dataset_name == "CIFAR-100":
        train_set = datasets.CIFAR100(
            root=dataset_path, train=(not train_with_test),
            download=True, transform=transform_train
        )
        test_set = datasets.CIFAR100(
            root=dataset_path, train=False, download=True, transform=transform_test
        )
    else:
        raise ValueError('Unknown dataset name.')

    # Data loaders
    trainloader = DataLoader(
        train_set, batch_size=batch_size_train, pin_memory=True, 
        shuffle=train_shuffle, num_workers=4, drop_last=True
    )
    trainloader_fast = DataLoader(
        train_set, batch_size=batch_size_train * 9, pin_memory=True, 
        shuffle=train_shuffle, num_workers=4, drop_last=True
    )
    testloader = DataLoader(
        test_set, batch_size=batch_size_test, pin_memory=True, 
        shuffle=False, num_workers=4, drop_last=False
    )

    # Composite model
    comp_model = get_comp_model(
        forward_settings, std_load_path, rob_load_path,
        num_classes=len(test_set.classes)
    )

    return comp_model, trainloader, testloader, \
        trainloader_fast, transform_train, transform_test


def load_ckpt(
    comp_model, optimizer, scheduler, warmup_scheduler, grad_scaler, lr,
    load_path, batch_per_ep, enable_BN=False, reset_scheduler=False, device='cpu'
):
    print(f"Loading checkpoint {load_path}.")
    state_dict = torch.load(load_path, map_location=device)

    try:  # Optimizer
        optimizer.load_state_dict(state_dict["optimizer"])
        print("Successfully loaded the optimizer state_dict.")
    except Exception as err_msg:
        print("Something went wrong when loading the optimizer!!!")
        print(f"Error message: {err_msg}")
        print(optimizer.state_dict()['param_groups'])
        print(state_dict["optimizer"]['param_groups'])
        print("The optimizer state_dict is not loaded!!!!!\n")

    # Scheduler
    if "scheduler" in state_dict.keys() and not reset_scheduler:
        scheduler.load_state_dict(state_dict["scheduler"])
        print("Successfully loaded the scheduler state_dict.")
    else:
        for g in optimizer.param_groups:
            g['lr'] = lr
        print("The scheduler has been reset.")

    # Warmup scheduler
    for ws in ["warmup_scheduler", "warm_scheduler"]:
        if ws in state_dict.keys():
            warmup_scheduler.load_state_dict(state_dict[ws])
            print("Successfully loaded the warmup scheduler state_dict.")
            break
        else:
            print("The warmup scheduler has been reset.")

    # Grad scaler
    if "grad_scaler" in state_dict.keys():
        grad_scaler.load_state_dict(state_dict["grad_scaler"])
        print("Successfully loaded the grad scaler state_dict.")

    # Epoch and batch
    if state_dict['ba'] is not None:
        state_dict['ba'] += 1
        if state_dict['ba'] % batch_per_ep != 0:
            print("Warning: Attempting to load from middle of an epoch.")
            print("Continuing training from a checkpoint only works "
                  "for loading from end of epoch.")
            print(f"batch count is {state_dict['ba']} “
                  f"but batch_per_ep is {batch_per_ep}")
    else:
        print("Please note that continuing training from a checkpoint only "
              "works for loading from end of epoch.")
    ep_start = 1 if reset_scheduler else (state_dict['ep'] + 1 if
        state_dict['ba'] is None or state_dict['ba'] % batch_per_ep == 0
        else state_dict['ep'])

    # Batch normalization
    # Process state dict (remove "model" prefix)
    if "bn.running_mean" in state_dict["model"].keys():
        state_dict['bn'] = OrderedDict()
        for key in ["running_mean", "running_var", "num_batches_tracked"]:
            state_dict['bn'][key] = state_dict["model"][f"bn.{key}"]
            del state_dict["model"][f"bn.{key}"]
    # Load processed state dict
    comp_model._comp_model.bn.load_state_dict(state_dict['bn'])

    # Gamma scale and bias (after BN)
    if "gamma_scale" in state_dict.keys():
        comp_model._comp_model.set_gamma_scale_bias(
            state_dict["gamma_scale"], state_dict["gamma_bias"]
        )
    print(f"Gamma scale: {comp_model._comp_model.gamma_scale.item()}, "
          f"Gamma bias: {comp_model._comp_model.gamma_bias.item()}.\n")

    # Policy network
    comp_model._comp_model.policy_net.load_state_dict(state_dict["model"])

    # Enable BN affine (default is False)
    if enable_BN:
        print("Enabled BN affine layer training.")
        comp_model.policy_net.module.bn.affine = True
        comp_model.policy_net.module.bn.weight = \
            nn.parameter.Parameter(torch.empty((1,), device=device))
        comp_model.policy_net.module.bn.bias = \
            nn.parameter.Parameter(torch.empty((1,), device=device))
        comp_model.policy_net.module.bn.reset_parameters()
        optimizer.add_param_group(
            {'params': comp_model.policy_net.module.bn.parameters()}
        )

    # Pre-attacked image indices
    img_inds = state_dict["img_inds"]
    if img_inds is not None:  # If there are pre-attacked images, send to CPU.
        img_inds = img_inds.cpu()

    return ep_start, state_dict["ba"], img_inds


def read_yaml(filename):
    with open(filename, 'r') as file:
        try:
            return yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)


def seed_all(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def eval_configs(config):
    for key, val in config.items():
        if isinstance(val, str):
            if ("(" in val or "[" in val or val == 'None' or
                ("e-" in val and len(val) < 8)):
                try:
                    config[key] = eval(val)
                except:
                    print(f"Could not convert {val}.")
        elif isinstance(val, dict):
            eval_configs(val)
