import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from torch.utils.data import Subset, ConcatDataset

import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from os.path import join

import sys
module_path = "/content/drive/MyDrive/Adaptive Smoothing"
if module_path not in sys.path:
    sys.path.append(module_path)
from adaptive_smoothing.attacks import single_pgd_attack, comp_pgd_attack


class ConvNextTforCIFAR(nn.Module):
    def __init__(self, weights=ConvNeXt_Tiny_Weights.DEFAULT):
        super().__init__()
        self.convnext = convnext_tiny(weights=weights)
        self.convnext.classifier[-1] = nn.Linear(
            in_features=768, out_features=10, bias=True)
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).cuda()
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).cuda()

    def forward(self, image):
        image = F.interpolate(image, size=224, mode='bilinear')
        image = (image - self.mean) / (self.std + 1e-8)
        return self.convnext(image), None


class SimpleCompositeModel(nn.Module):
    def __init__(self, std_model, adv_model, comp_model_setting, allow_grad=True, alpha=None):
        super().__init__()
        self.std_model, self.adv_model = std_model, adv_model
        self.use_gradnorm_std = comp_model_setting["use_gradnorm_std"]
        self.use_gradnorm_adv = comp_model_setting["use_gradnorm_adv"]
        self.accelerate = comp_model_setting["accelerate"]
        self.allow_grad = allow_grad

        self.alpha = alpha
        self.sm, self.use_sm = nn.Softmax(dim=1), comp_model_setting["use_sm"]
        self.defense_p = 1 if comp_model_setting["attack_type"] == 'l_inf' else 2

    def get_grad_norm(self, out_data, images_var):
        # Compute gradient norm
        # To verify if gradient can flow through gradnorm, try
        # x_grad = autograd.grad(gradnorm, images_var,
        #     grad_outputs=torch.ones((n,1), device=device))
        n, c = out_data.shape
        if self.accelerate:
            images_grad = autograd.grad(
                out_data.max(dim=1).values, images_var,
                create_graph=self.allow_grad, retain_graph=True,
                grad_outputs=torch.ones((n,), device='cuda'))[0]
            gradnorm = images_grad.reshape(n, -1).norm(
                dim=1, p=self.defense_p).unsqueeze(1)
        else:
            gradnorm = torch.zeros((n, c), device='cuda')
            for i in range(c):
                images_grad = autograd.grad(
                    out_data[:, i], images_var,
                    create_graph=self.allow_grad, retain_graph=True,
                    grad_outputs=torch.ones((n,), device='cuda'))[0]
                gradnorm[:, i] = images_grad.reshape(n, -1).norm(
                    dim=1, p=self.defense_p)
        return gradnorm

    def forward(self, images):
        assert self.alpha is not None
        if isinstance(self.alpha, list):
            assert all([a >= 0 for a in self.alpha])
        else:
            assert self.alpha >= 0  # Here, we are using the [0, inf) scale

        images_var = images if images.requires_grad else (
            images.clone().detach().requires_grad_(True))
        outputs_adv = self.adv_model(images_var)[0]
        outputs_std = self.std_model(images_var)[0]
        if self.use_sm:
            outputs_std, outputs_adv = self.sm(outputs_std), self.sm(outputs_adv)

        trade_off = torch.tensor(1.)
        if self.use_gradnorm_std:
            gradnorm_std = self.get_grad_norm(outputs_std, images_var)
            trade_off = trade_off * gradnorm_std  # Don't use in-place operations
        if self.use_gradnorm_adv:
            gradnorm_adv = self.get_grad_norm(outputs_adv, images_var)
            trade_off = trade_off / gradnorm_adv  # Don't use in-place operations

        if isinstance(self.alpha, list):
            outputs = [None for _ in self.alpha]
            for alpha_ind, cur_alpha in enumerate(self.alpha):
                cur_R = trade_off * cur_alpha
                outputs[alpha_ind] = torch.log(outputs_std + cur_R * outputs_adv) - torch.log(
                    1 + cur_R) if self.use_sm else (outputs_std + cur_R * outputs_adv) / (1 + cur_R)
        else:
            cur_R = trade_off * self.alpha
            outputs = torch.log(outputs_std + cur_R * outputs_adv) - torch.log(
                1 + cur_R) if self.use_sm else (outputs_std + cur_R * outputs_adv) / (1 + cur_R)

        return outputs, images_var, self.alpha


def load_model(model, path):
    model = model.cuda()
    model.load_state_dict(torch.load(path))
    model = torch.nn.DataParallel(model)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model


def get_dataloaders(batch_size, img_per_class=200):
    transform = transforms.Compose([transforms.ToTensor()])

    testset = datasets.CIFAR10(root='/content/drive/MyDrive/Datasets',
                               train=False, download=True, transform=transform)

    testset = ConcatDataset([
        Subset(testset, list(np.arange(i*1000, i*1000+img_per_class))) for i in range(10)])
    print(f"Number of test images: {len(testset)}.")
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size["Test"], shuffle=False)

    trainset = datasets.CIFAR10(root='/content/drive/MyDrive/Datasets',
                                train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size["Train"], shuffle=False)
    return trainloader, testloader


def eval_base(std_model, adv_model, testloader, pgd_setting):
    pgd_loss = nn.CrossEntropyLoss()
    base_acc = {"STD": {}, "ADV": {}}

    for model_name, model in zip(["STD", "ADV"], [std_model, adv_model]):
        print(f"Evaluating the {model_name} model.")
        correct, total = {"Clean": 0, "Attacked": 0}, 0

        tepoch = tqdm(testloader)
        for images, labels in tepoch:
            images, labels = images.cuda(), labels.cuda()
            attacked_images = single_pgd_attack(
                model, images, labels, pgd_type=pgd_setting["type"], pgd_eps=pgd_setting["eps"],
                pgd_alpha=pgd_setting["alpha"], pgd_iters=pgd_setting["iters"], pgd_loss=pgd_loss)

            with torch.no_grad():
                total += labels.size(0)
                _, predicted = model(images)[0].detach().max(dim=1)
                _, att_predicted = model(attacked_images)[0].detach().max(dim=1)
                correct["Clean"] += (predicted == labels).sum().item()
                correct["Attacked"] += (att_predicted == labels).sum().item()
                tepoch.set_postfix({"CleanAcc": 100 * correct["Clean"] / total,
                                    "AttackedAcc": 100 * correct["Attacked"] / total})

        for typ in ["Clean", "Attacked"]:
            base_acc[model_name][typ] = 100 * correct[typ] / total
            print(f"Accuracy of the {model_name} model on 10000 "
                  f"{typ} test images: {base_acc[model_name][typ]:.3f} %.")
    return base_acc


def eval_comp(comp_model, testloader, alphas, pgd_setting):
    types = ["Clean", "STD", "ADV", "Comp"]
    correct, total, comp_acc = {}, {}, {}
    for typ in types:
        correct[typ], total[typ] = [0 for _ in alphas[typ]], 0
    total["Comp"] = [0 for _ in alphas["Comp"]]
    pgd_loss = nn.CrossEntropyLoss()
    print("Evaluating the adaptively smoothed model.")

    num_batches = np.inf
    # Clean data and attacks targeting the base models
    print("Attack targeting base models.")
    comp_model.allow_grad = False
    for data_ind, (images, labels) in enumerate(tqdm(testloader)):
        if data_ind >= num_batches:
            break
        images, labels = images.cuda(), labels.cuda()
        torch.cuda.empty_cache()

        for typ, model in zip(types[:-1], [None, comp_model.std_model, comp_model.adv_model]):
            attacked_images = single_pgd_attack(
                model, images, labels, pgd_type=pgd_setting["type"], pgd_eps=pgd_setting["eps"],
                pgd_alpha=pgd_setting["alpha"], pgd_iters=pgd_setting["iters"], pgd_loss=pgd_loss
            ) if typ != "Clean" else images

            comp_model.alpha = alphas[typ]
            outputs = comp_model(attacked_images)[0]
            total[typ] += labels.size(0)
            for alpha_ind, output in enumerate(outputs):
                _, predicted = output.max(dim=1)
                correct[typ][alpha_ind] += (predicted == labels).sum().item()

    for typ in types[:-1]:
        comp_acc[typ] = list(np.array(correct[typ]) / np.array(total[typ]) * 100)
        print(f"Accuracy of the smoothed model with {typ} data: \n{comp_acc[typ]}.")

    # Attacks targeting the composite model
    print("Attack targeting the smoothed model.")
    comp_model.allow_grad = True
    for data_ind, (images, labels) in tqdm(enumerate(testloader)):
        if data_ind >= num_batches:
            break
        for alpha_ind, alpha in enumerate(alphas["Comp"]):
            comp_model.alpha = alpha
            images, labels = images.cuda(), labels.cuda()
            torch.cuda.empty_cache()

            attacked_images = comp_pgd_attack(
                comp_model, images, labels, pgd_type=pgd_setting["type"], pgd_eps=pgd_setting["eps"],
                pgd_alpha=pgd_setting["alpha"], pgd_iters=pgd_setting["iters"], pgd_loss=pgd_loss)

            _, predicted = comp_model(attacked_images)[0].max(dim=1)
            total["Comp"][alpha_ind] += labels.size(0)
            correct["Comp"][alpha_ind] += (predicted == labels).sum().item()

    comp_acc["Comp"] = list(np.array(correct["Comp"]) / np.array(total["Comp"]) * 100)
    print(f"Accuracy of the smoothed model with Comp data: \n{comp_acc['Comp']}.")
    return comp_acc


def plot_results(alphas, base_acc, comp_acc, save_dir, pgd_type='l_inf'):
    alpha_max, alpha_min = max(alphas["Clean"]), min(alphas["Clean"])

    _ = plt.figure(figsize=(6.2, 3.8))
    plt.plot(
        [0] + alphas["Clean"] + [alpha_max * 10],
        [base_acc["STD"]["Clean"]] + comp_acc["Clean"] + [base_acc["ADV"]["Clean"]],
        '-', label=r"$g_{CNN}^\theta (\cdot)$ only, Clean")
    plt.plot([0, alpha_max * 10], [base_acc["STD"]["Clean"]] * 2, '--',
             label=r"$g (\cdot)$ only, Clean")
    plt.plot([0, alpha_max * 10], [base_acc["ADV"]["Clean"]] * 2, '--',
             label=r"$h (\cdot)$ only, Clean")

    plt.title(r"Clean accuracy vs $\alpha$")
    plt.xlabel(r"$\alpha$")
    plt.ylabel("Clean accuracy (%)")
    plt.xscale('log')
    plt.xlim(alpha_min / 3, alpha_max * 3)
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig(join(save_dir, "clean_nograd.pdf"))

    _ = plt.figure(figsize=(6.2, 4.4))
    for typ in ["Clean", "STD", "ADV", "Comp"]:
        plt.plot(alphas[typ], comp_acc[typ], '-',
                 label=r"$g_{CNN}^\theta (\cdot)$" + f", {typ}")

    plt.plot([0, alpha_max * 10], [base_acc["STD"]["Clean"]] * 2, '--',
             label=r"$g (\cdot)$ only, Clean")
    plt.plot([0, alpha_max * 10], [base_acc["ADV"]["Clean"]] * 2, '--',
             label=r"$h (\cdot)$ only, Clean")
    plt.plot([0, alpha_max * 10], [base_acc["STD"]["Attacked"]] * 2, '--',
             label=r"$g (\cdot)$ only, Attacked")
    plt.plot([0, alpha_max * 10], [base_acc["ADV"]["Attacked"]] * 2, '--',
             label=r"$h (\cdot)$ only, Attacked")

    if pgd_type == 'l_inf':
        plt.title(r"PGD$_{20}$ accuracy vs $\alpha$ "
                  r"($\ell_\infty$ radius: $\frac{8}{255}$)")
    else:
        plt.title(r"PGD$_{20}$ accuracy vs $\alpha$ ($\ell_2$ radius: $0.5$)")
    plt.xlabel(r"$\alpha$")
    plt.ylabel(r"Clean and PGD$_{10}$ accuracy (%)")
    plt.xscale('log')
    plt.xlim(alpha_min, alpha_max)
    plt.ylim(0, 100)
    plt.yticks(np.arange(0, 101, 10))
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig(join(save_dir, "all.pdf"))
