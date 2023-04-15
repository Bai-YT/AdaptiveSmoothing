import torch
import torch.nn as nn

import numpy as np
from scipy.stats import chi2

from adaptive_smoothing.losses import CompLoss, DLRLoss, SimpleCompLoss
from comp_autoattack.autopgd_base import APGDAttack


# Helper / worker functions
def pgd_update(images, ori_images, pgd_type, pgd_alpha, pgd_eps, n, momentum=0):
    grad = images.grad + momentum

    if pgd_type == 'l_inf':
        adv_images = images.detach() + pgd_alpha * grad.sign()
        eta = torch.clamp(adv_images - ori_images, min=-pgd_eps, max=pgd_eps)  # Project
    elif pgd_type == 'l_2':
        gradnorms = grad.reshape(n, -1).norm(dim=1, p=2).view(n, 1, 1, 1)
        adv_images = images.detach() + pgd_alpha * grad / gradnorms
        eta = adv_images - ori_images
        etanorms = eta.reshape(n, -1).norm(dim=1, p=2).view(n, 1, 1, 1)
        eta = eta * torch.minimum(torch.tensor(1.), pgd_eps / etanorms)  # Project

    images = torch.clamp(ori_images + eta, min=0, max=1).detach()
    return images, grad


def single_pgd_attack(model, images, labels, pgd_type, pgd_loss, pgd_eps, 
                      pgd_alpha, pgd_iters, mom_decay=0, random_start=False):
    assert pgd_type in ['l_inf', 'l_2']
    ori_images = images.clone().detach()
    n, c, h, w = tuple(images.shape)
    if random_start:
        images = images + torch.empty_like(images).uniform_(-pgd_eps, pgd_eps)
        images = torch.clamp(images, min=0, max=1).detach()

    momentum = 0  # Initial momentum
    for _ in range(pgd_iters):
        images = images.clone().detach().requires_grad_(True)
        outputs, _ = model(images)

        model.zero_grad()
        cost = pgd_loss(outputs, labels).cuda()
        cost.backward()  # This is incompatable with accum_grad
        images, momentum = pgd_update(
            images, ori_images, pgd_type, pgd_alpha, pgd_eps, n, momentum * mom_decay)
    return images


def comp_pgd_attack(comp_model, images, labels, pgd_type, pgd_loss,
                    pgd_eps, pgd_alpha, pgd_iters, mom_decay=0, random_start=False):
    assert pgd_type in ['l_inf', 'l_2']
    comp_model.module.eval()  # Call the modified eval()

    ori_images = images.clone().detach()
    n, c, h, w = tuple(images.shape)
    if random_start:
        images = images + torch.empty_like(images).uniform_(-1, 1) * pgd_eps.view(n, 1, 1, 1)
        images = torch.clamp(images, min=0, max=1).detach()

    momentum = 0  # Initial momentum
    for _ in range(pgd_iters):
        comp_model.zero_grad()
        images = images.clone().detach().requires_grad_(True)
        outputs, alphas = comp_model(images)

        if hasattr(pgd_loss, "bce_loss"):  # CompLoss and SimpleCompLoss have "bce_loss"
            cost = pgd_loss(outputs, labels, alphas, sup_labels=torch.ones_like(alphas)).cuda()
        else:  # CE or DLR
            cost = pgd_loss(outputs, labels).cuda()
        cost.backward()  # This is incompatable with accum_grad
        
        images, momentum = pgd_update(
            images, ori_images, pgd_type, pgd_alpha, pgd_eps, n, momentum * mom_decay)
    return images


class PGDAdversary:
    def __init__(self, comp_model, randomized_attack_settings, pgd_settings):
        self.comp_model = comp_model

        self.randomized_attack_settings = randomized_attack_settings
        self.pgd_type = pgd_settings["type"]  # L2 or Linf
        self.pgd_eps = pgd_settings["eps"]
        self.pgd_iters_test = pgd_settings["iters_test"]
        self.pgd_alpha_test = pgd_settings["alpha_test"]

        self.n_target_classes = pgd_settings["n_target_classes"]
        # Loss functions for PGD
        self.dlr_loss_targeted = SimpleCompLoss(base_loss=DLRLoss(reduction='none'), w=0)
        self.ce_loss = SimpleCompLoss(base_loss=nn.CrossEntropyLoss(reduction='none'), w=0)
        # self.comp_loss = CompLoss()
        self.comp_loss_simple = SimpleCompLoss(base_loss=nn.CrossEntropyLoss(reduction='none'))

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.type_convert = {"l_inf": "Linf", "l_2": "L2", "l_1": "L1"}
        self.apgd_adversary = APGDAttack(
            self.comp_model, n_iter=self.pgd_iters_test, n_restarts=1,
            norm=self.type_convert[self.pgd_type], eps=self.pgd_eps, seed=0, loss='ce',
            eot_iter=1, rho=.75, comp=True, check_zero_grad=False, device=self.device)

    def pgd_attack_eval(self, images, labels, pgd_type, pgd_eps, pgd_alpha, pgd_iters,
                        loss_type='ce', random_start=False):

        # Attack loss function
        if loss_type not in ['ce', 'dlr', 'comp_simple']:
            raise ValueError("Attack loss type must be one of ['ce', 'dlr', 'comp_simple']")
        attack_loss = {
            'ce': self.ce_loss, 'dlr': self.dlr_loss_targeted,
            'comp_simple': self.comp_loss_simple}[loss_type]  # 'comp': self.comp_loss

        # If using targeted DLR loss, we need to iterate through the target classes.
        if loss_type == 'dlr':
            dlr_adv_imgs = [None for _ in range(self.n_target_classes)]
            output, gammas = self.comp_model(images)
            sorted_output = output.sort(dim=1)[1]
            for tc in range(2, self.n_target_classes + 2):
                self.dlr_loss_targeted.y_target = sorted_output[:, -tc]
                dlr_adv_imgs[tc - 2] = comp_pgd_attack(
                    comp_model=self.comp_model, images=images, labels=labels,
                    pgd_type=pgd_type, pgd_loss=self.dlr_loss_targeted,
                    pgd_eps=pgd_eps, pgd_alpha=pgd_alpha, pgd_iters=pgd_iters,
                    mom_decay=0, random_start=random_start)
            return dlr_adv_imgs

        return comp_pgd_attack(
            comp_model=self.comp_model, images=images, labels=labels, pgd_type=pgd_type,
            pgd_loss=attack_loss, pgd_eps=pgd_eps, pgd_alpha=pgd_alpha, pgd_iters=pgd_iters,
            mom_decay=0, random_start=random_start)

    def get_pgd_params(self, n):
        if self.randomized_attack_settings["randomize"]:
            eps_factor = (torch.rand(size=(n,), device=self.device) *
                            self.randomized_attack_settings["eps_factor_scale"] +
                            self.randomized_attack_settings["eps_factor_loc"])  # Unif(.4, 1.1)
            cur_eps = self.pgd_eps * eps_factor
            # The number of iterations drawn from a Chi^2 distribution
            iters_rv = (chi2.rvs(self.randomized_attack_settings["iters_df"]) +
                        self.randomized_attack_settings["iters_const"])
        else:
            cur_eps = self.pgd_eps
            iters_rv = self.randomized_attack_settings["iters_const"]

        # Make sure that at least one PGD iteration is scheduled
        cur_iters = np.maximum(1, int(np.rint(iters_rv)))
        # alpha_factor is 1.8 for L_inf, 5 for L_2
        cur_alpha = cur_eps / iters_rv * self.randomized_attack_settings["alpha_factor"]
        # Momentum decay factor drawn from a uniform distribution
        cur_mom_decay = (torch.rand(size=(n,), device=self.device) *
                         self.randomized_attack_settings["mom_decay_scale"] +
                         self.randomized_attack_settings["mom_decay_loc"])
        
        return (cur_eps.view(-1, 1, 1, 1), cur_iters,
                cur_alpha.view(-1, 1, 1, 1), cur_mom_decay.view(-1, 1, 1, 1))

    def randomized_pgd_attack(self, images, labels):
        # Get PGD parameters
        cur_eps, cur_iters, cur_alpha, cur_mom_decay = self.get_pgd_params(images.shape[0])
        old_w_ce = self.ce_loss.w
        self.ce_loss.w = (torch.rand_like(labels.float()) * 
                          self.randomized_attack_settings["comp_loss_wmax"])

        # Cross-entropy attack
        ce_adv_imgs = comp_pgd_attack(
            comp_model=self.comp_model, images=images, labels=labels,
            pgd_type=self.pgd_type, pgd_loss=self.ce_loss, pgd_eps=cur_eps,
            pgd_alpha=cur_alpha, pgd_iters=cur_iters, mom_decay=cur_mom_decay,
            random_start=self.randomized_attack_settings["random_start"]).detach().cpu()
        self.ce_loss.w = old_w_ce

        # Targeted DLR attack
        old_w_dlr = self.dlr_loss_targeted.w
        dlr_adv_imgs = [None for _ in range(self.n_target_classes)]
        if self.n_target_classes != 0:
            with torch.no_grad():
                output, gammas = self.comp_model(images)
                sorted_output = output.detach().sort(dim=1)[1]

        for tc in range(2, self.n_target_classes + 2):
            self.dlr_loss_targeted.base_loss.y_target = sorted_output[:, -tc]
            self.dlr_loss_targeted.w = (
                torch.rand_like(labels.float()) *  # Random comp loss weight
                self.randomized_attack_settings["comp_loss_wmax"])

            # Get PGD parameters and run attack
            cur_eps, cur_iters, cur_alpha, cur_mom_decay = self.get_pgd_params(images.shape[0])
            dlr_adv_imgs[tc - 2] = comp_pgd_attack(
                comp_model=self.comp_model, images=images, labels=labels,
                pgd_type=self.pgd_type, pgd_loss=self.dlr_loss_targeted,
                pgd_eps=cur_eps, pgd_alpha=cur_alpha, pgd_iters=cur_iters,
                random_start=self.randomized_attack_settings["random_start"],
                mom_decay=cur_mom_decay).detach().cpu()

        self.dlr_loss_targeted.w = old_w_dlr
        return [ce_adv_imgs] + dlr_adv_imgs

    def randomized_apgd_attack(self, images, labels):
        self.apgd_adversary = APGDAttack(
            self.comp_model, n_iter=self.pgd_iters_test, n_restarts=1,
            norm=self.type_convert[self.pgd_type], eps=self.pgd_eps, seed=0, 
            loss='ce', eot_iter=1, rho=.75, comp=True, device=self.device)

        # Cross-entropy attack
        self.apgd_adversary.y_target = None
        self.apgd_adversary.w = (torch.rand_like(labels.float()) * 
                                 self.randomized_attack_settings["comp_loss_wmax"])
        self.apgd_adversary.loss = 'ce-randomized'

        pgd_params = self.get_pgd_params(images.shape[0])
        self.apgd_adversary.eps, self.apgd_adversary.n_iter, _, _ = pgd_params
        self.apgd_adversary.init_hyperparam(x=images)
        ce_adv_imgs = self.apgd_adversary.attack_single_run(
            x=images, y=labels)[-1].detach().cpu()
        # print("Done with CE.")

        # Targeted DLR attack
        dlr_adv_imgs = [None for _ in range(self.n_target_classes)]
        if self.n_target_classes != 0:
            with torch.no_grad():
                output, gammas = self.comp_model(images)
                sorted_output = output.detach().sort(dim=1)[1]

        for tc in range(2, self.n_target_classes + 2):
            self.apgd_adversary = APGDAttack(
                self.comp_model, n_iter=self.pgd_iters_test, n_restarts=1, rho=.75,
                norm=self.type_convert[self.pgd_type], eps=self.pgd_eps, seed=0, 
                loss='dlr-randomized', eot_iter=1, comp=True, device=self.device)

            self.apgd_adversary.y_target = sorted_output[:, -tc]
            self.apgd_adversary.w = (
                torch.rand_like(labels.float()) *  # Random comp loss weight
                self.randomized_attack_settings["comp_loss_wmax"])

            pgd_params = self.get_pgd_params(images.shape[0])
            self.apgd_adversary.eps, self.apgd_adversary.n_iter, _, _ = pgd_params
            assert self.apgd_adversary.y_target.shape[0] == self.apgd_adversary.eps.shape[0]
            assert self.apgd_adversary.y_target.shape[0] == self.apgd_adversary.w.shape[0]
            assert self.apgd_adversary.y_target.shape[0] == images.shape[0]

            self.apgd_adversary.init_hyperparam(x=images)
            dlr_adv_imgs[tc - 2] = self.apgd_adversary.attack_single_run(
                x=images, y=labels)[-1].detach().cpu()
            # print(f"Done with DLR {tc}.")

        return [ce_adv_imgs] + dlr_adv_imgs
