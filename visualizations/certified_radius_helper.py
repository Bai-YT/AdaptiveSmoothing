import json
from tqdm import tqdm
import os
from os.path import join
from datetime import datetime, timezone

import numpy as np
from scipy.stats import norm
import torch
import matplotlib.pyplot as plt
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

from visualizations.compare_grad_helper import get_dataloaders, SimpleCompositeModel


def eval_base(std_model, adv_model, testloader):
    base_acc = {"STD": None, "ADV": None}
    all_res = {"STD": [], "ADV": []}

    for model_name, model in zip(["ADV", "STD"], [adv_model, std_model]):
        print(f"Evaluating the {model_name} model.")
        correct, total = 0, 0

        tepoch = tqdm(testloader)
        for images, labels in tepoch:
            images = images.cuda()
            
            with torch.no_grad():
                res = model(images)[0].detach()  # Results per class
                if len(res.shape) == 1:  # After argmax
                    predicted = res
                else:
                    _, predicted = res.max(dim=1)
                
                correct += (predicted.cpu() == labels).sum().item()
                total += labels.size(0)
                tepoch.set_postfix({"CleanAcc": 100 * correct / total})
                all_res[model_name] += [res]

        base_acc[model_name] = 100 * correct / total
        print(f"Accuracy of the {model_name} model on {total} "
              f"test images: {base_acc[model_name]:.3f} %.")
        all_res[model_name] = torch.cat(all_res[model_name], dim=0).cpu().numpy().tolist()
    return base_acc, all_res


def eval_comp(comp_model, testloader, alphas):
    correct, comp_acc, all_res, all_labels, total = {}, {}, {}, [], 0
    for alpha in alphas:
        all_res[f"{alpha:.2e}"] = []
        correct[f"{alpha:.2e}"] = 0
    print("Evaluating the adaptively smoothed model.")

    comp_model.allow_grad = False
    comp_model.alpha = alphas
    with torch.no_grad():
        for data_ind, (images, labels) in enumerate(tqdm(testloader)):
            images, labels = images.cuda(), labels.cuda()

            outputs = comp_model(images)[0]
            total += labels.size(0)
            for alpha_ind, alpha in enumerate(alphas):
                _, predicted = outputs[alpha_ind].max(dim=1)
                correct[f"{alpha:.2e}"] += (predicted == labels).sum().item()
                all_res[f"{alpha:.2e}"] += [outputs[alpha_ind].detach().cpu()]
            all_labels += [labels.detach().cpu()]
            
    for alpha in alphas:
        all_res[f"{alpha:.2e}"] = torch.cat(all_res[f"{alpha:.2e}"], dim=0).numpy().tolist()
        comp_acc[f"{alpha:.2e}"] = correct[f"{alpha:.2e}"] / total
    return comp_acc, all_res, torch.cat(all_labels, dim=0).numpy().tolist()


def run_evals(std_model, adv_model, comp_model_setting, alpha_setting, 
              batch_size, img_per_class, num_skip, save_dir):
    # For logging
    timestr = str(datetime.now(timezone.utc).isoformat()).split(".")[0]
    timestr = timestr.replace("T", "").replace("-", "").replace(":", "")
    save_dir = save_dir + "_" + timestr
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Dataloader
    trainloader, testloader = get_dataloaders(batch_size, img_per_class, num_skip)

    # # Base classifier accuracy
    # base_acc, all_res = eval_base(std_model, adv_model, testloader)
    # with open(join(save_dir, 'base_res.json'), 'w') as p:
    #     json.dump(all_res, f)
    # return base_acc

    # Composite classifier accuracy
    # alphas = [0] + list(np.logspace(
    #     alpha_setting["min"], alpha_setting["max"], alpha_setting["num"])) + [np.inf]
    real_alphas = np.array([0, 0.5, 0.67, 0.76, 1])
    alphas = list(real_alphas / (1 - real_alphas))

    comp_model = SimpleCompositeModel(std_model, adv_model, comp_model_setting)
    comp_acc, all_res, all_labels = eval_comp(comp_model, testloader, alphas)

    save_dic = {"save_dir": save_dir,
                "alpha_setting": alpha_setting,
                "alphas": alphas,
                "comp_model_setting": comp_model_setting,
                "comp_acc": comp_acc,
                "all_res": all_res,
                "all_labels": all_labels}
    
    with open(join(save_dir, "results.json"), 'w') as f:
        json.dump(save_dic, f)
    return save_dic


def get_lipschitz_radii(max_diff: np.ndarray, real_alpha: float, Lip_h: float, 
                        correct_mask_h: np.ndarray):
    """
    :param  max_diff:       Shape (n,). The difference between the predict class 
                            and the most confident confounding class.
    :param  real_alpha:     The alpha value after reparameterization.
    :param  Lip_h:          The Lipschitz constant of h.
    :param  correct_mask_h: Shape (n,). The predicted classes returned by h.
    :return lip_radii:      Shape (n,). 
                            The Lipschitz-based certified radii of mixed classifier.
    """
    if real_alpha == 0 or np.isnan(real_alpha):
        return np.zeros_like(correct_mask_h)
    numer = np.maximum(0, real_alpha * max_diff + real_alpha - 1)
    denom = real_alpha * 2 * Lip_h
    lip_radii_all = (numer / denom).numpy()
    return np.where(correct_mask_h, lip_radii_all, np.zeros_like(lip_radii_all))


def get_rs_radii(top_2: np.ndarray, real_alpha: float, sigma: float,
                 correct_mask_h: np.ndarray):
    """
    :param  top_2:          Shape (n, 2). The probabilities of the top two most probable classes.
    :param  real_alpha:     The alpha value after reparameterization.
    :param  sigma:          The standard deviation of the noise used in RS.
    :param  correct_mask_h: Shape (n,). The predicted classes returned by h.
    :return rs_radii:       Shape (n,). The RS-based certified radii of mixed classifier.
    """
    if real_alpha == 0 or np.isnan(real_alpha):
        return np.zeros((top_2.shape[0]))
    mu = (1 - real_alpha) / real_alpha
    first_term = norm.ppf(top_2[:, 0].numpy() / (1 + mu))
    second_term = norm.ppf((top_2[:, 1].numpy() + mu) / (1 + mu))
    rs_radii_all = sigma / 2 * np.maximum(0, first_term - second_term)
    return np.where(correct_mask_h, rs_radii_all, np.zeros_like(rs_radii_all))


def process_results(json_path, max_x=1):
    with open(json_path, 'r') as f:
        return_dic = json.load(f)  # Load file

    # Calculate Lipschitz constant
    sigma = float(json_path.split("_")[0][-4:])
    Lip_h = np.sqrt(2 / np.pi) / sigma
    print(f"Lipschitz constant for h_i: {Lip_h}")

    # Load data
    alphas = np.array(return_dic["alphas"])
    comp_acc = return_dic["comp_acc"]
    all_res_h = np.asarray(return_dic["all_res"][f"{alphas[-1]:.2e}"])
    all_labels = np.asarray(return_dic["all_labels"])

    # Reparameterize alpha
    assert alphas[-1] == np.inf  # Otherwise this code will not work properly
    real_alphas = alphas / (1 + alphas)
    real_alphas[-1] = 1.

    # Initialize plots
    _, axs = plt.subplots(2, len(alphas), figsize=(20, 6))
    thresh = np.linspace(0, max_x, 1000)
    bins = np.linspace(0, max_x, 20)

    # Process the predictions of h
    correct_mask_h = all_res_h.argmax(axis=1) == all_labels
    top_2 = torch.tensor(all_res_h).topk(2, dim=1).values
    max_diff = top_2[:, 0] - top_2[:, 1]  # This is needed for the Lipschitz-based radii

    for alpha_ind, cur_real_alpha in enumerate(real_alphas):
        # Calculate the radius via the Lipschitz constant
        lip_radii = get_lipschitz_radii(max_diff, cur_real_alpha, Lip_h, correct_mask_h)
        rs_radii = get_rs_radii(top_2, cur_real_alpha, sigma, correct_mask_h)
        robust_radii = {"Lipschitz": lip_radii, "RS": rs_radii}

        for typ in ["Lipschitz", "RS"]:
            # Plot the histogram of robust radii
            weights = np.ones_like(robust_radii[typ]) / float(len(robust_radii[typ])) * 100
            axs[0][alpha_ind].hist(robust_radii[typ], bins=bins, weights=weights, alpha=.8,
                                   label=f"{typ}-based\ncertified radius")

            # Plot the certified accuracy versus r
            rob_tab = robust_radii[typ].reshape(-1, 1) > thresh.reshape(1, 1000)
            rob_perc = rob_tab.mean(axis=0) * 100
            axs[1][alpha_ind].plot(thresh, rob_perc, label=f"{typ}-based\ncertification")

        # Formatting for the histogram
        axs[0][alpha_ind].set_ylim([0., 100.])
        axs[0][alpha_ind].grid()
        axs[0][alpha_ind].set_title(r"$\alpha=$"
                                    f"{cur_real_alpha:.3f}")
        
        # Formatting for the curves
        axs[1][alpha_ind].plot(0, list(comp_acc.values())[alpha_ind] * 100, 'o', label="Clean")
        axs[1][alpha_ind].set_ylim([-5., 100.])
        axs[1][alpha_ind].set_xlabel(r"$\ell_2$ radius")
        axs[1][alpha_ind].grid()
    
    for id, name in zip([0, 1], ["Histogram (%)", "Accuracy (%)"]):
        axs[id][-1].legend()
        axs[id][0].set_ylabel(name)
        
    plt.tight_layout()
    plt.savefig(join(return_dic["save_dir"], "rad_plots.pdf"))

    # fig = plt.figure(figsize=(4.5, 3))
    # plt.plot(real_alphas, list(comp_acc.values()), "-x")
    # plt.tight_layout()
