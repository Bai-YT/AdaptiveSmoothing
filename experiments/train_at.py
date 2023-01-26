import torch

# Add parent folder to directory
import os, sys
import numpy as np
par_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(par_path)
root_path = os.path.dirname(os.getcwd())
print("Root directory:", root_path)

from models.resnet import *
from adaptive_smoothing.trainer import *
from adaptive_smoothing.utils import *

# Reproducibility
import random
random.seed(20220304)
torch.manual_seed(20220304)
np.random.seed(20220304)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device:", device)


# Load paths
model_parent = os.path.join(root_path, "log")
std_model_load_path = os.path.join(model_parent, "Base/CIFAR-10_basic_training_.pt")
adv_model_load_path = os.path.join(model_parent, "Base/CIFAR-10_pgd_adversarial_training_.pt")

# Experiment settings
# # For evaluating the AT model only (without a policy)
# forward_settings = {"accelerate": True,
#                     "parallel": False,
#                     "nonlinearity": "relu",
#                     "normalization_type": "batch",
#                     "use_softmax": False,
#                     "defense_type": 'l_inf',
#                     "in_planes": (64, 64),
#                     "base_graph": True,
#                     "NN_alphas": False,
#                     "alpha_graph": False,
#                     "alpha": np.inf}
# For training / evaluating the policy
forward_settings = {"accelerate": True,
                    "parallel": True,
                    "nonlinearity": "relu",
                    "normalization_type": "batch",
                    "use_softmax": False,
                    "defense_type": 'l_inf',
                    "in_planes": (64, 64),
                    "base_graph": True,
                    "NN_alphas": True,
                    "alpha_graph": True,
                    "alpha": 3.5}

dataset_settings = {"name": "CIFAR-10",
                    "train_with_test": False,  # If True, then train on the test set of CIFAR-10
                    "path": os.path.join(root_path, "Datasets"),
                    "pa_path": None,  # os.path.join(model_parent, "CustomDatasets/Linf_CIFAR-10/CC_AT/dataset_argmax.pickle"),
                    "alpha_0": 7.1,
                    "blocks": 8}

comp_loss_params = {"consts": (0.5, 1, 0.1),
                    "scale": {"clean": 2., "ada": 1., "pa": 1.}}

training_settings = {"epochs": 5,
                     "batch_size_train": 800,
                     "batch_size_test": 800 * torch.cuda.device_count(),
                     "lr": 1e-4,
                     "weight_decay": 1e-5,
                     "comp_loss_params": comp_loss_params,
                     "schedule": "none",
                     "accum_grad": True,
                     "consts": {"train": (2, 0.4), "eval": (3, 20)}}  # (scale, bias=20)

pgd_settings = {"type": 'l_inf',
                "eps": 8. / 255.,
                "iters_test": 20,
                "alpha_test": 0.0027,
                "use_aa": True,
                "n_target_classes": 9,
                "aa_batch_size": training_settings["batch_size_test"] * 5 // dataset_settings["blocks"],
                "pgd_eval_loss": 'ce'}
pgd_settings["attacks_to_run"] = ['apgd-ce', 'apgd-t']  # For training val
# pgd_settings["attacks_to_run"] = ['apgd-ce', 'apgd-t', 'fab-t', 'square']  # For testing

randomized_attack_settings = {"randomize": True,
                              "mode": "sample",
                              "apgd": True,
                              "random_start": True,
                              "eps_factor_loc": 0.4,
                              "eps_factor_scale": 0.7,
                              "mom_decay_loc": 0.,
                              "mom_decay_scale": 0.9,
                              "iters_df": 80,
                              "iters_const": 0,
                              "alpha_factor": 5,
                              "comp_loss_wmax": 0.4,
                              "curriculum": False}

# Experiment driver code
std_model, adv_model = ResNet18(), ResNet18()

comp_model = get_comp_model(std_model=std_model, std_load_path=std_model_load_path,
                            adv_model=adv_model, adv_load_path=adv_model_load_path,
                            forward_settings=forward_settings)

comp_trainer = CompModelTrainer(comp_model=comp_model, 
                                dataset_settings=dataset_settings, training_settings=training_settings, 
                                pgd_settings=pgd_settings, randomized_attack_settings=randomized_attack_settings)

# Evaluate (please also update AutoAttack settings)
comp_trainer.comp_model.enable_autocast = False
load_path = "/home/ubuntu/project/Adaptive-Smoothing/log/CompModel/CIFAR-10_AT_Linf/V3_APGD_CC_BN_5/epoch_5.pt"
save_path = "/home/ubuntu/project/Adaptive-Smoothing/log/CompModel/CIFAR-10_AT_Linf/V3_APGD_CC_BN_5"
_ = comp_trainer.load_checkpoint(load_path)
comp_trainer.evaluate(load_path=load_path, save_path=save_path, auto_attack=True, full=True)
comp_trainer.evaluate_adap(clean_only=True, verbose=False, full=True)

# Training (please also update AutoAttack settings)
# comp_trainer.comp_model.enable_autocast = True
# load_path = None  # "/home/ubuntu/project/Adaptive-Smoothing/log/CompModel/CIFAR-10_AT_Linf/V3_APGD_CC_BN_4/epoch_3.pt"
# save_path = os.path.join(model_parent, "CompModel/CIFAR-10_AT_Linf/V3_APGD_CC_BN_6")
# eval_freq = 2 if dataset_settings["train_with_test"] else 1
# comp_trainer.train(save_path=save_path, load_path=load_path, eval_freq=eval_freq, debug=False)
