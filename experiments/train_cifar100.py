import torch

# Add parent folder to directory
import os, sys
import numpy as np
par_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(par_path)
root_path = os.path.dirname(os.getcwd())
print("Root directory:", root_path)

from adaptive_smoothing.trainer import *
from adaptive_smoothing.utils import *

# Reproducibility
import random
random.seed(20220304)
torch.manual_seed(20220304)
np.random.seed(20220304)


# Load paths
model_parent = os.path.join(root_path, "log")
std_model_load_path = os.path.join(model_parent, "Base/bit-rn152-cifar100.pth.tar")
adv_model_load_path = os.path.join(model_parent, "Base/cifar100_linf_wrn70-16_with.pt")

# Experiment settings
# For evaluating the model without a policy (fixed alpha)
# forward_settings = {"std_model_type": 'rn152',
#                     "adv_model_type": 'wrn7016',
#                     "pn_version": 4,  # 1 or 3 or 4
#                     "parallel": 2,
#                     "nonlinearity": "relu",
#                     "normalization_type": "batch",
#                     "use_softmax": True,
#                     "defense_type": 'l_inf',
#                     "in_planes": (512, 256),
#                     "base_graph": True,
#                     "NN_alphas": False,
#                     "alpha_graph": False,
#                     "out_scale": 1,
#                     "alpha_scale": 1,  # This applies to constant alpha as well
#                     "alpha": 2}
# For training / evaluating the policy
forward_settings = {"std_model_type": 'rn152',
                    "adv_model_type": 'wrn7016',
                    "pn_version": 4,  # 1 or 3 or 4
                    # 0 is no parallel, 1 is component-wise parallel, 2 is overall parallel
                    "parallel": 2 if torch.cuda.device_count() > 1 else 0,
                    "nonlinearity": "relu",
                    "normalization_type": "batch",
                    "use_softmax": True,
                    "defense_type": 'l_inf',
                    "in_planes": (512, 256),
                    "base_graph": True,
                    "NN_alphas": True,
                    "alpha_graph": True,
                    "out_scale": 1,
                    "alpha_scale": 1,  # This applies to constant alpha as well
                    "alpha": 2}

dataset_settings = {"name": "CIFAR-100",
                    "train_with_test": False,  # If True, then train on the test set of CIFAR-10
                    "train_shuffle": True,
                    "use_data_aug": True,
                    "path": "/home/ubuntu/project/data",
                    "pa_path": None,  # os.path.join(model_parent, "CustomDatasets/Linf_CIFAR-10/CC_TRADE/dataset_argmax.pickle"),
                    "alpha_0": None,
                    "blocks": 5}

comp_loss_params = {"consts": (0.5, 1, 0.1),
                    "scale": {"clean": 2., "ada": 1., "pa": 1.}}

training_settings = {"epochs": 3,
                     "batch_size_train": 200,  # 160 (160) for V100, 200 (200) for A10G for AA (PGD)
                     # 25 for V100, 40 for A10G without policy; 12 for V100, 30 for A10G with policy.
                     # 50 for V100, 60 for A10G if PGD.
                     "batch_size_test": 40 * torch.cuda.device_count(),
                     "save_eval_imgs": True,
                     "lr": 1e-4,  # 1e-5 for APGD, 1e-4 for PGD
                     "weight_decay": 1e-5,
                     "comp_loss_params": comp_loss_params,
                     "schedule": "none",
                     # Number of mini-batches before optimizer step. -1 for batch optimization.
                     "accum_grad": -1,  # 4 for APGD, -1 for PGD
                     "use_fast_loader": False,  # True for APGD, False for PGD
                     "consts": {"train": (2, 0.4), "eval": (.5, 2)},  # (scale, bias) (200, 40),
                     "use_no_clamp": False}  # If use_no_clamp, then disable clamping during attacks

pgd_settings = {"type": 'l_inf',
                "eps": 8. / 255.,
                "iters_test": 20,
                "alpha_test": 0.0027,
                "use_aa": True,  # Whether use AutoAttack during evaluation within training
                "n_target_classes": 9,
                "aa_batch_size": training_settings["batch_size_test"] * 5 // dataset_settings["blocks"],
                "pgd_eval_loss": 'ce'}
#                 "pgd_eval_loss": 'comp_simple'}  # 'ce'
# pgd_settings["attacks_to_run"] = ['apgd-ce', 'apgd-t']  # For training val
# pgd_settings["attacks_to_run"] = ['apgd-ce', 'apgd-t', 'fab-t', 'square']  # For testing
pgd_settings["attacks_to_run"] = ['apgd-ce-rand', 'apgd-t-rand', 'fab-t', 'square']  # For testing

randomized_attack_settings = {"randomize": True,
                              "mode": "sample",
                              "apgd": pgd_settings["use_aa"],
                              "random_start": True,
                              "eps_factor_loc": 0.4,
                              "eps_factor_scale": 0.7,
                              "mom_decay_loc": 0.,
                              "mom_decay_scale": 0.9,
                              "iters_df": 20,  # 75
                              "iters_const": 0,
                              "alpha_factor": 5,  # The factor related to the step size alpha (not the smoothing alpha)
                              "comp_loss_wmax": 0.4,
                              "curriculum": False}  # Curriculum is not needed

# Experiment driver code
comp_trainer = CompModelTrainer(std_load_path=std_model_load_path, adv_load_path=adv_model_load_path,
                                forward_settings=forward_settings, dataset_settings=dataset_settings, 
                                training_settings=training_settings, pgd_settings=pgd_settings, 
                                randomized_attack_settings=randomized_attack_settings)

# Evaluate (please also update AutoAttack settings)
comp_trainer._comp_model.enable_autocast = True
load_path = "/home/ubuntu/project/Adaptive-Smoothing/log/CompModel/CIFAR-100_Linf/V3_APGD_CC_BN_1/epoch_2_ba_500.pt"
save_path = os.path.join(model_parent, "CompModel/CIFAR-100_Linf/V3_APGD_CC_BN_1_eval/")
# load_path = "/home/ubuntu/project/Adaptive-Smoothing/log/CompModel/CIFAR-100_Linf/V3_PGD20_CC_BN_SettingA/epoch_3_ba_750.pt"
# save_path = os.path.join(model_parent, "CompModel/CIFAR-100_Linf/V3_PGD20_CC_BN_SettingA")
comp_trainer.evaluate(load_path=load_path, save_path=save_path, save_adv_imgs=True, auto_attack=True, full=True)
# _ = comp_trainer.load_checkpoint(load_path)
# comp_trainer.evaluate_adap(clean_only=True, full=True)
# comp_trainer.evaluate_data()

# Training (please also update AutoAttack settings)
# comp_trainer._comp_model.enable_autocast = True
# load_path = None
# save_path = os.path.join(model_parent, "CompModel/CIFAR-100_Linf/V3_PGD20_CC_BN_SettingB")  # V3_APGD_CC_BN_1
# eval_freq = {"epoch": training_settings["epochs"] + 1, "iter": 75}  # 25 for AA, 75 for PGD
# comp_trainer.train(save_path=save_path, load_path=load_path, eval_freq=eval_freq, debug=False)
