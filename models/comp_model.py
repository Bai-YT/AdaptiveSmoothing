import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List
import numpy as np

from .policy_net import PolicyNetV1, PolicyNetV4, PolicyNetV3


# @title Define Composite Model
class CompositeModel(nn.Module):
    enable_autocast = False
    sigmoid = nn.Sigmoid()
    softmax = nn.Softmax(dim=1)
    alpha = -np.inf

    def __init__(self, models: List[nn.Module], forward_settings):
        super().__init__()
        self.set_alpha(forward_settings["alpha"], forward_settings["NN_alphas"])
        self.out_scale, self.alpha_scale = forward_settings["out_scale"], forward_settings["alpha_scale"]
        self.no_clamp = False  # Clamp the output

        self.defense_type = forward_settings["defense_type"]
        if self.defense_type == 'l_inf':
            self.defense_p = 1  # l_inf defense
        elif self.defense_type == 'l_2':
            self.defense_p = 2  # l_2 defense
        else:
            raise ValueError("Unknown defense type.")

        self.parallel = forward_settings["parallel"] == 1
        to_cuda = forward_settings["parallel"] == 0
        self.base_graph = forward_settings["base_graph"]
        self.alpha_graph = forward_settings["alpha_graph"]
        if self.alpha_graph and not self.base_graph:
            raise ValueError("alpha_graph cannot be created without base_graph.")

        self.pn_version = forward_settings["pn_version"]
        if self.pn_version == 1:
            self.policy_net = PolicyNetV1(forward_settings)
        elif self.pn_version == 3:
            self.policy_net = PolicyNetV3(forward_settings)
        elif self.pn_version == 4:
            self.policy_net = PolicyNetV4(forward_settings)
        else:
            raise "Unsupported Policy Network version."
        if self.parallel:
            self.policy_net = nn.DataParallel(self.policy_net)

        self.models = nn.ModuleList(models)
        for model in self.models:
            for param in model.parameters():
                assert param.requires_grad == False

        self.policy_net = self.policy_net.cuda() if to_cuda else self.policy_net
        self.bn = nn.BatchNorm1d(num_features=1, affine=False, momentum=0.01)
        self.use_softmax = forward_settings["use_softmax"]
        self.softmax = nn.Softmax(dim=1)
        self.const = (1., 0.)
        self.resize = forward_settings["std_model_type"] in ["rn50", "rn152"]

        for model, typ in zip(self.models, ["STD", "ADV"]):
            print(f"The {typ} model has {sum(p.numel() for p in model.parameters())} parameters. "
                  f"{sum(p.numel() for p in model.parameters() if p.requires_grad)} parameters are trainable.")
        print(f"The policy network has {sum(p.numel() for p in self.policy_net.parameters())} parameters. "
              f"{sum(p.numel() for p in self.policy_net.parameters() if p.requires_grad)} parameters are trainable.\n")

    def set_alpha(self, alpha, NN_alphas):
        self.NN_alphas = NN_alphas
        if self.NN_alphas:
            if self.alpha != alpha:
                self.alpha = alpha
                print(f"Alpha has been set to {self.alpha}, "
                      "but the policy network is active so the change is not effective.")
            else:
                print(f"Policy network activated. Alpha is {self.alpha}.")
        else:
            self.alpha = alpha
            print(f"Using fixed alpha={self.alpha}. No policy.")
            if self.alpha == -np.inf:
                print("Using the STD network only.")
            elif self.alpha == np.inf:
                print("Using the ADV network only.")

    def forward(self, images):
        if self.alpha_graph and (not self.base_graph or not self.NN_alphas):
            raise ValueError('alpha_graph cannot be created without base_graph or without the policy.')
        for model in self.models:
            model.eval()
        if (not self.parallel) and hasattr(self.models[0], "root") and (
            self.models[0].root.conv.weight.device != images.device):
            print(self.models[0].root.conv.weight.device, 
                  self.models[1].logits.weight.device, 
                  self.policy_net.linear.weight.device, 
                  self.bn.running_mean.device, images.device)
            raise ValueError("Device mismatch!")

        # Create image variable to allow gradient flow
        images_var = images if (images.requires_grad or not self.base_graph
                                ) else images.clone().detach().requires_grad_(True)
        images_var_resized = F.interpolate(images_var, size=(128, 128), mode='bilinear') \
                                if (self.resize and self.alpha != np.inf) else images_var

        # Forward passes
        with torch.cuda.amp.autocast(enabled=self.enable_autocast):
            # Single model special cases
            if self.alpha == -np.inf and not self.NN_alphas:  # STD Model only
                out, _ = self.models[0](images_var_resized)
                # print(out.max())
                return out, images_var, torch.tensor(-np.inf) * torch.ones((out.shape[0],)).to(out.device)
            elif self.alpha == np.inf and not self.NN_alphas:  # ADV Model only
                out, _ = self.models[1](images_var)
                return out, images_var, torch.tensor(np.inf) * torch.ones((out.shape[0],)).to(out.device)

            # General case -- use both models
            out_data_std, interm_std = self.models[0](images_var_resized)
            out_data_adv, interm_adv = self.models[1](images_var)

            if self.NN_alphas:  # Get alphas
                if self.alpha_graph:
                    alphas = self.policy_net([interm_std[0], interm_adv[0]], [interm_std[1], interm_adv[1]])
                else:
                    alphas = self.policy_net([interm_std[0].detach().clone(), interm_adv[0].detach().clone()],
                                             [interm_std[1].detach().clone(), interm_adv[1].detach().clone()])
                # print(alphas.mean().item(), alphas.std().item())

                # Clamp alphas so the BN works the best
                if self.training:
                    amean, astd = alphas.mean().item(), alphas.std().item()
                    alphas = torch.clamp(alphas, min=(-.6) * astd + amean, max=.6 * astd + amean)
                alphas = self.bn(alphas)
                if not self.training and not self.no_clamp:
                    alphas = torch.clamp(alphas, min=-3, max=6)
                alphas = alphas * self.const[0] + self.const[1]

        # print(alphas.mean().item(), alphas.std().item())
        if not self.NN_alphas:
            alphas = torch.ones((out_data_std.shape[0], 1)).to(out_data_std.device) * self.alpha
        trade_off = self.sigmoid(alphas)

        if self.use_softmax:
            out_data = torch.log((1 - trade_off) * self.softmax(out_data_std) + trade_off * self.softmax(out_data_adv))
        else:
            out_data = (1 - trade_off) * out_data_std + trade_off * out_data_adv

        if not self.base_graph:
            images_var = images_var.detach()
        return out_data * self.out_scale, images_var, alphas.reshape(-1) * self.alpha_scale
