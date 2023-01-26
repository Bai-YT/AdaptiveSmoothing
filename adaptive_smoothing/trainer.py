import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
import pytorch_warmup as warmup

import pickle
import numpy as np
from tqdm import tqdm
import json

# Add the parent folder to directory
import os, sys
par_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(par_path)

from adaptive_smoothing import utils, attacks
from autoattack import AutoAttack
from adaptive_smoothing.losses import CompLoss

SCHEDULER_FACTOR = 1  # Number of optimizer updates between scheduler updates


# @title Define Model Trainer
class CompModelTrainer:
    train_data, train_labels = None, None
    test_data, test_labels = None, None
    adv_alphas, sup_labels = None, None
    n, n_test, transform = None, None, None
    no_batches, ba, img_inds, glob_batch_num = None, None, None, None
    writer, save_path = None, None  # Tensorboard writer
    print(f"Number of GPUs: {torch.cuda.device_count()}.")
    print(f"Number of CPUs: {os.cpu_count()}.")

    def __init__(self, forward_settings, std_load_path, adv_load_path, dataset_settings, 
                 training_settings, pgd_settings, randomized_attack_settings):
        """
            TODO: Implement CIFAR-10.1 training
        """
        # Comp model settings, for debugging purposes
        self.setting_dics = {"forward_settings": forward_settings, "dataset_settings": dataset_settings, 
                             "training_settings": training_settings, "pgd_settings": pgd_settings, 
                             "randomized_attack_settings:": randomized_attack_settings}
        
        # Loss functions
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')
        self.comp_loss_params = training_settings["comp_loss_params"]
        self.comp_loss = CompLoss(consts=self.comp_loss_params["consts"])

        # Dataset settings
        self.batch_size_train = int(training_settings["batch_size_train"])
        self.batch_size_test = int(training_settings["batch_size_test"])
        self.num_blocks = dataset_settings["blocks"]
        self.train_with_test = dataset_settings["train_with_test"]
        self.use_fast_loader = training_settings["use_fast_loader"]
        print(f"Train on the {'test' if self.train_with_test else 'training'} set.")
        if dataset_settings["pa_path"] is not None:
            self.set_dataset(dataset_settings["pa_path"], dataset_settings["alpha_0"])

        # Get loaders
        print("Dataset:", dataset_settings["name"])
        cmdl_and_loaders = utils.get_cmodel_and_loaders(forward_settings=forward_settings,
                                                        std_load_path=std_load_path, adv_load_path=adv_load_path,
                                                        dataset_name=dataset_settings["name"],
                                                        dataset_path=dataset_settings["path"],
                                                        batch_size_train=self.batch_size_train * 5 // self.num_blocks,
                                                        batch_size_test=self.batch_size_test * 5 // self.num_blocks,
                                                        train_shuffle=dataset_settings["train_shuffle"], 
                                                        use_data_aug=dataset_settings["use_data_aug"],
                                                        train_with_test=self.train_with_test)
                                                       
        (self.comp_model, self.trainloader, self.testloader, self.trainloader_fast, 
            self.transform_train, self.transform_test) = cmdl_and_loaders  # Fasterloader has a hardcoded factor of 9 now
        if forward_settings["parallel"] == 2:
            print("Parallelizing the entire composite model.")
            self.comp_model = nn.DataParallel(self.comp_model)
            self._comp_model = self.comp_model.module  # This is the unparallelized model
        else:
            self._comp_model = self.comp_model

        # Trainer settings
        self.training_settings = training_settings
        self.accum_grad = self.training_settings["accum_grad"]
        self.epochs, self.cur_ep = self.training_settings["epochs"], 0
        self.optimizer = torch.optim.AdamW(self.comp_model.parameters(), lr=self.training_settings["lr"],
                                           weight_decay=self.training_settings["weight_decay"])

        # self.n_mini_batches = self.pgd_adversary.n_target_classes + (2 if self.n is None else 3)
        self.n_mini_batches = (pgd_settings["n_target_classes"] + 1) * 2
        if self.accum_grad == self.n_mini_batches:
            self.accum_grad = -1
        if self.accum_grad != -1:
            assert self.n_mini_batches % self.accum_grad == 0
        base_len = (1 if self.accum_grad == -1 else self.n_mini_batches // self.accum_grad)
        num_opt_steps = base_len * len(self.trainloader) * self.epochs
        print(f"Number of minibatches per batch: {self.n_mini_batches}. Accummulate {self.accum_grad} grad evaluations.")
        print(f"Total number of optimization steps: {num_opt_steps}. Warmup period: {num_opt_steps // 25}.")

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, num_opt_steps)
        self.warmup_scheduler = warmup.ExponentialWarmup(self.optimizer, warmup_period=num_opt_steps // 25)
        self.scheduler_cntr = 0
        self.bn_consts = self.training_settings["consts"]
        self.save_eval_imgs = self.training_settings["save_eval_imgs"]

        # PGD settings
        self.pgd_eval_loss = pgd_settings["pgd_eval_loss"]
        self.use_apgd_training = randomized_attack_settings["apgd"]
        self.use_aa = pgd_settings["use_aa"]
        self.aa_batch_size = pgd_settings["aa_batch_size"]

        self.pgd_adversary = attacks.PGDAdversary(self.comp_model, randomized_attack_settings, pgd_settings)
        assert self.pgd_adversary.pgd_type == self._comp_model.defense_type
        if pgd_settings["type"] == 'l_inf':
            self.aa_adversary = AutoAttack(self.comp_model, norm='Linf',
                                           eps=self.pgd_adversary.pgd_eps, version='standard')
        elif pgd_settings["type"] == 'l_2':
            self.aa_adversary = AutoAttack(self.comp_model, norm='L2',
                                           eps=self.pgd_adversary.pgd_eps, version='standard')
        else:
            raise ValueError("Unknown attack norm type.")
        self.aa_adversary.attacks_to_run = pgd_settings["attacks_to_run"]

        # Device settings
        # self.device = torch.device(f"cuda:{torch.cuda.device_count() - 1}" if (torch.cuda.device_count() > 1 and 
        #     forward_settings["parallel"]) else ("cuda:0" if torch.cuda.is_available() else "cpu"))
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print(f"Trainer device: {self.device}.")

    def set_dataset(self, dataset_path, alpha_0):
        print(f"Loading pre-attacked data from {dataset_path}")
        with open(dataset_path, 'rb') as readfile:
            dataset = pickle.load(readfile)
            
        # Training data
        if not self.train_with_test:
            self.train_data = dataset["train_data"]
            self.train_labels = dataset["train_labels"]
        else:
            self.train_data = dataset["test_data"]
            self.train_labels = dataset["test_labels"]
        # Test data
        self.test_data = dataset["test_data"]
        self.test_labels = dataset["test_labels"]
        
        # Alpha values used for generating attacks
        self.adv_alphas = dataset["test_adv_alphas" if self.train_with_test else "train_adv_alphas"]
        alphas = [self.adv_alphas[int(i * (10000 if self.train_with_test else 50000))].item()
                  for i in range(2, self.num_blocks)]
        print("alphas:", alphas)
        # Labels for the policy network
        # If adv_alphas < alpha_0, then the ADV network should be used, and sup_label should be 1
        # If adv_alphas >= alpha_0, then the STD network should be used, and sup_label should be 0
        self.sup_labels = (self.adv_alphas < alpha_0).half()

        if self.train_data is not None:
            print("There is no training data.")
            self.n = self.train_data.shape[0]
        self.n_test = self.test_data.shape[0]
        self.transform = transforms.Compose([transforms.RandomHorizontalFlip(p=.5),
                                             transforms.RandomCrop(32, padding=4)])
        if self.train_data is not None:
            assert self.num_blocks == self.n // (10000 if self.train_with_test else 50000)
            assert self.n % (10000 if self.train_with_test else 50000) == 0
            assert self.n == self.n_test * (1 if self.train_with_test else 5)
            assert self.n == self.train_labels.shape[0]
            assert self.n == self.sup_labels.shape[0]
            assert self.n_test == self.test_labels.shape[0]
            self.no_batches = int(np.ceil(self.n / self.batch_size_train))
            self.ba = 0  # Batch count
            self.img_inds = torch.randperm(self.n)  # For randomly using the pre-attacked data

        print(f"Training instances count: {self.n}, test instances count: {self.n_test}.")
        print(f"Number of blocks: {self.num_blocks}.")

    def get_loss_acc(self, images, labels):
        with torch.no_grad():
            preds, _, alphas = self.comp_model(images)
            loss = self.ce_loss(preds, labels).mean().item()
            correct = (preds.argmax(dim=1) == labels).sum().item()

        # import matplotlib.pyplot as plt
        # plt, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(9, 3))
        # ax1.hist(alphas.detach().cpu().numpy(), bins=40, label='all')
        # ax1.hist(alphas[preds.argmax(dim=1) == labels].detach().cpu().numpy(), bins=40, label='correct')
        # ax1.hist(alphas[preds.argmax(dim=1) != labels].detach().cpu().numpy(), bins=40, alpha=0.6, label='incorrect')
        # plt.legend()
        # ax2.plot(alphas.detach().cpu().numpy(), self.ce_loss(preds, labels).detach().cpu().numpy(), 'x')
        # ax2.set_xlabel("alpha")
        # ax2.set_ylabel("loss")
        # ax3.plot(alphas.detach().cpu().numpy(), (preds.argmax(dim=1) == labels).detach().cpu().numpy(), 'x')
        # ax3.set_xlabel("alpha")
        # ax3.set_ylabel("correctness")
        # plt.tight_layout()
        # plt.show()
        # import pdb; pdb.set_trace()

        return loss, correct, alphas.mean().item(), (alphas>self.bn_consts['eval'][1]).sum().item()

    def evaluate_adap(self, pgd_eps=8./255., pgd_alpha=0.0027, pgd_iters=20, start_ind=0, save_adv_imgs=False,
                      clean_only=False, auto_attack=True, verbose=False, full=True, debug=False):
        self.comp_model.eval()
        if self._comp_model.NN_alphas:
            self._comp_model.const = self.bn_consts["eval"]
            print(f"The output mean of the policy is set to {self._comp_model.const[1]:.1f} "
                f"and the standard deviation is set to {self._comp_model.const[0]:.1f}.")
        else:
            print(f"Using a constant alpha of {self._comp_model.alpha}.")
            
        total_loss, correct = {"clean": [], "adv": []}, {"clean": 0, "adv": 0}
        alpha_mean, alpha_large = {"clean": [], "adv": []}, {"clean": 0, "adv": 0}
        total = 0

        all_adv_images = []
        titer = tqdm(self.testloader, unit="batch", disable=verbose)  # If verbose, disable TQDM
        for ind, (images, labels) in enumerate(titer):
            # if ind < 0:  # Here, put the number of the desired first batch
            #     continue
            print(f"batch {ind + 1}")
            titer.set_description(f"Epoch {self.cur_ep}, adaptive evaluation")
            if ind >= start_ind:
                images, labels = images.to(self.device, non_blocking=True), labels.to(self.device, non_blocking=True)
                if clean_only:
                    adv_images = None
                else:
                    self._comp_model.no_clamp = self.training_settings["use_no_clamp"]
                    if auto_attack:
                        adv_images = self.aa_adversary.run_standard_evaluation(images, labels, bs=self.aa_batch_size,
                                                                               perform_checks=False)
                    else:
                        adv_images = self.pgd_adversary.pgd_attack_eval(images, labels, pgd_type=self.pgd_adversary.pgd_type,
                                                                        pgd_eps=pgd_eps, pgd_alpha=pgd_alpha, pgd_iters=pgd_iters,
                                                                        loss_type=self.pgd_eval_loss, random_start=False)
                    self._comp_model.no_clamp = False

                # import pickle
                # with open("/home/ubuntu/project/Adaptive-Smoothing/experiments/test_img.pt", 'wb') as save_file:
                #     pickle.dump([images, adv_images, labels], save_file)
                if save_adv_imgs and not clean_only:
                    try:
                        all_adv_images += [adv_images.detach().cpu().numpy()]
                    except:
                        print("Something went wrong with saving all adv images.")

                for key, imgs in zip(["clean", "adv"], [images, adv_images]):
                    cur_loss, cur_correct, cur_alpha_mean, cur_alpha_large = self.get_loss_acc(imgs, labels)
                    total_loss[key] += [cur_loss]
                    correct[key] += cur_correct
                    alpha_mean[key] += [cur_alpha_mean]
                    alpha_large[key] += cur_alpha_large
                    torch.cuda.empty_cache()

                    if verbose: 
                        print(f"Batch {ind + 1} of {len(self.testloader)}, "
                              f"{key} correct: {cur_correct} out of {labels.size(0)},")
                        print(f"mean alpha: {np.mean(alpha_mean[key]):.3f}, "
                              f"large alpha: {cur_alpha_large} out of {labels.size(0)}.")
                    if clean_only:
                        break
                total += labels.size(0)
                if clean_only:
                    titer.set_postfix(clean_acc=correct["clean"] / total * 100., clean_loss=np.mean(total_loss["clean"]), 
                                      alpha_acc=(1 - alpha_large["clean"] / total) * 100.)
                else:
                    titer.set_postfix(clean_acc=correct["clean"] / total * 100., adv_acc=correct["adv"] / total * 100.,
                                      adv_alpha=np.mean(alpha_mean["adv"]))

            # If debug, then run only one batch. Else, use 1000 images (10% of the test set).
            if (debug and ind >= 0) or (not full and (ind * self.batch_size_test * 5 // self.num_blocks) >= 1000): 
                break

        # Log validation stats with clean and adaptive attack
        clean_loss, clean_acc = np.mean(total_loss["clean"]), correct["clean"] / total
        clean_alpha_mean, clean_alpha_acc = np.mean(alpha_mean["clean"]), 1 - (alpha_large["clean"] / total)
        print(f"Average clean test accuracy: {(clean_acc * 100.):.2f} %, mean alpha: {clean_alpha_mean:.3f}, "
              f"alpha accuracy: {(clean_alpha_acc * 100.):.2f} %.")
        if not clean_only:
            adv_loss, adv_acc = np.mean(total_loss["adv"]), correct["adv"] / total
            adv_alpha_mean, adv_alpha_acc = np.mean(alpha_mean["adv"]), alpha_large["adv"] / total
            print(f"Average adv test accuracy: {(adv_acc * 100.):.2f} %, mean alpha: {adv_alpha_mean:.3f}, "
                  f"alpha accuracy: {(adv_alpha_acc * 100.):.2f} %.\n")

        if self.writer is not None:
            self.writer.add_scalar("Clean val loss", clean_loss, self.cur_ep)
            self.writer.add_scalar("Clean val acc", clean_acc * 100., self.cur_ep)
            self.writer.add_scalar("Clean mean alpha", clean_alpha_mean, self.cur_ep)
            self.writer.add_scalar("Clean alpha acc", clean_alpha_acc * 100., self.cur_ep)
            if not clean_only:
                self.writer.add_scalar("Adaptive adv val loss", adv_loss, self.cur_ep)
                self.writer.add_scalar("Adaptive adv val acc", adv_acc * 100., self.cur_ep)
                self.writer.add_scalar("Adaptive adv mean alpha", adv_alpha_mean, self.cur_ep)
                self.writer.add_scalar("Adaptive adv alpha acc", adv_alpha_acc * 100., self.cur_ep)

        if self.save_path is None:
            print("No save path provided. Validation results not saved.")
        else:
            os.makedirs(self.save_path, exist_ok=True)
            with open(os.path.join(self.save_path, "log_eval_ada.csv"), 'a') as wfile:
                wfile.write(f"Epoch {self.cur_ep}, clean_only: {clean_only}, auto_attack: {auto_attack}\n")
                wfile.write("Clean data\n")
                wfile.write(f"Average clean test loss: {clean_loss:.4f}.\n")
                wfile.write(f"Average clean test accuracy: {(clean_acc * 100.):.2f} %.\n")
                wfile.write(f"Average clean mean alpha: {clean_alpha_mean:.3f}.\n")
                wfile.write(f"Average clean alpha acc: {(clean_alpha_acc * 100.):.2f} %.\n")
                if not clean_only:
                    wfile.write("Adaptive adversary\n")
                    wfile.write(f"Average adv test loss: {adv_loss:.4f}.\n")
                    wfile.write(f"Average adv test accuracy: {(adv_acc * 100.):.2f} %.\n")
                    wfile.write(f"Average adv mean alpha: {adv_alpha_mean:.3f}.\n")
                    wfile.write(f"Average adv alpha acc: {(adv_alpha_acc * 100.):.2f} %.\n\n")
            
            if save_adv_imgs:
                fname = (f"adv_imgs_ep_{self.cur_ep}_ba_{self.glob_batch_num}{'_full' if full else ''}"
                         f"{'_aa' if auto_attack else '_pgd'}.pkl")
                pth = os.path.join(self.save_path, fname)
                with open(pth, 'wb') as wfile:
                    pickle.dump(all_adv_images, wfile)

        return clean_loss, clean_acc, None if clean_only else adv_loss, None if clean_only else adv_acc

    def evaluate_data(self, debug=False):
        if self.test_data is None:
            print("No pre-attacked data specified. Test with existing data not performed.")
            return

        self.comp_model.eval()
        if self._comp_model.NN_alphas:
            self._comp_model.const = self.bn_consts["eval"]
            print(f"The output mean of the policy is set to {self._comp_model.const[1]:.1f} "
                f"and the standard deviation is set to {self._comp_model.const[0]:.1f}.")
        else:
            print(f"Using a constant alpha of {self._comp_model.alpha}.")
        
        no_test_batches = int(np.ceil(self.n_test / (self.batch_size_test * 5 // self.num_blocks)))
        block_loss = [None for _ in range(self.num_blocks)]
        block_acc = [None for _ in range(self.num_blocks)]
        real_bs = self.batch_size_test * 5 // self.num_blocks

        titer = tqdm(range(self.num_blocks), disable=self.save_path is None)
        for ind in titer:
            titer.set_description(f"Epoch {self.cur_ep}, pre-attacked evaluation")
            total_loss, correct = [], 0

            for ba in range(ind * no_test_batches // self.num_blocks,
                            (ind + 1) * no_test_batches // self.num_blocks):
                images = self.test_data[ba * real_bs: (ba + 1) * real_bs, 
                                        :, :, :].detach().to(self.device, non_blocking=True)
                labels = self.test_labels[ba * real_bs: (ba + 1) * real_bs].to(self.device, non_blocking=True)
                with torch.no_grad():
                    preds, _, _ = self.comp_model(images)
                    loss = self.ce_loss(preds, labels).mean()
                total_loss += [loss.item()]
                correct += (preds.argmax(dim=1) == labels).sum().item()
                if debug and ba % 2 == 1: 
                    break

            # Log validation stats with pre-attacked data
            block_loss[ind] = np.mean(total_loss)
            block_acc[ind] = correct / self.n_test * self.num_blocks
            if self.writer is not None:
                self.writer.add_scalar(f"Block {ind + 1} val loss", block_loss[ind], self.cur_ep)
                self.writer.add_scalar(f"Block {ind + 1} val acc", block_acc[ind] * 100, self.cur_ep)
            if self.save_path is not None:
                os.makedirs(self.save_path, exist_ok=True)
                with open(os.path.join(self.save_path, "log_eval_pre.csv"), 'a') as wfile:
                    wfile.write(f"Epoch {self.cur_ep} Block {ind + 1}:\n")
                    wfile.write(f"Average test loss: {block_loss[ind]:.4f}.\n")
                    wfile.write(f"Average test accuracy: {(block_acc[ind] * 100):.2f} %.\n")
            else:
                print(f"Epoch {self.cur_ep} Block {ind + 1}:")
                print(f"Average test loss: {block_loss[ind]:.4f}.")
                print(f"Average test accuracy: {(block_acc[ind] * 100):.2f} %.")

        return block_loss, block_acc

    def evaluate(self, load_path=None, save_path=None, pgd_eps=None, pgd_alpha=None, 
                 pgd_iters=None, auto_attack=True, save_adv_imgs=False, full=True, debug=False):
        if save_path is not None:
            self.save_path = save_path
        if load_path is not None:
            print("Loading model. Current Comp Model overwritten.")
            _ = self.load_checkpoint(load_path)
        print(f"Autocast enabled: {self._comp_model.enable_autocast}.")

        if pgd_eps is None: pgd_eps = self.pgd_adversary.pgd_eps
        if pgd_alpha is None: pgd_alpha = self.pgd_adversary.pgd_alpha_test
        if pgd_iters is None: pgd_iters = self.pgd_adversary.pgd_iters_test
        if not auto_attack:
            print(f"PGD settings: loss: {self.pgd_eval_loss}, epsilon={pgd_eps}, "
                  f"alpha={pgd_alpha}, iterations={pgd_iters}.")
        self.evaluate_adap(pgd_eps=pgd_eps, pgd_alpha=pgd_alpha, pgd_iters=pgd_iters, save_adv_imgs=save_adv_imgs,
                           auto_attack=auto_attack, full=full, verbose=auto_attack, debug=debug)
        self.evaluate_data(debug=debug)
        
    def step_rand_curriculum(self):
        if self.cur_ep > 5 and self.cur_ep <= 20:
            self.pgd_adversary.randomized_attack_settings["iters_df"] = 40  # (self.cur_ep - 5) * 2 + 10
            self.pgd_adversary.randomized_attack_settings["iters_const"] = 30  # (self.cur_ep - 5) * 2
            self.pgd_adversary.randomized_attack_settings["alpha_factor"] = 5  # (self.cur_ep - 5) * .1 + 2.5
            print(f"Randomized PGD DF: {self.pgd_adversary.randomized_attack_settings['iters_df']}, "
                  f"const: {self.pgd_adversary.randomized_attack_settings['iters_const']}, "
                  f"alpha factor: {self.pgd_adversary.randomized_attack_settings['alpha_factor']}.")

    def get_data_buffer(self, images, labels):
        if self.use_fast_loader:
            clean_images_2, clean_labels_2 = next(iter(self.trainloader_fast))
            # clean_images_2 = clean_images_2.to(self.device, non_blocking=True)
            clean_labels_2 = clean_labels_2.to(self.device, non_blocking=True)

        clean_images = images.to(self.device, non_blocking=True)
        clean_labels = labels.to(self.device, non_blocking=True)
        zeros = torch.zeros_like(clean_labels, device=self.device)
        ones = torch.ones_like(clean_labels, device=self.device)

        # Attack before zero_grad so that we don't remove the clean gradients
        self.comp_model.eval()
        self._comp_model.no_clamp = self.training_settings["use_no_clamp"]
        if self.use_apgd_training:
            adv_images_list = self.pgd_adversary.randomized_apgd_attack(clean_images, clean_labels)
        else:
            adv_images_list = self.pgd_adversary.randomized_pgd_attack(clean_images, clean_labels)
        self._comp_model.no_clamp = False
        self.comp_model.train()
        # print("Done with attacking.")

        if self.train_data is None:  # No pre-attacked data
            # Assemble buffer
            all_images = torch.cat([clean_images.cpu()] + ([clean_images_2] if self.use_fast_loader else []) + 
                                    adv_images_list, dim=0)
            all_labels = torch.cat([clean_labels] + ([clean_labels_2] if self.use_fast_loader else []) + 
                                   [clean_labels] * (self.pgd_adversary.n_target_classes + 1), dim=0)
            all_sup_labels = torch.cat([zeros] * (10 if self.use_fast_loader else 1) + 
                                       [ones] * (self.pgd_adversary.n_target_classes + 1), dim=0)
            all_scales = torch.cat([ones * self.comp_loss_params["scale"]["clean"]] * (10 if self.use_fast_loader else 1) + 
                                   [ones * self.comp_loss_params["scale"]["ada"]] * 
                                   (self.pgd_adversary.n_target_classes + 1), dim=0) / (
                            self.n_mini_batches if self.accum_grad == -1 else self.accum_grad)

        else:  # There are pre-attacked data
            assert not self.use_fast_loader  # Incompatible
            cur_inds = self.img_inds[self.ba * self.batch_size_train: (self.ba + 1) * self.batch_size_train]
            pa_images = self.transform(self.train_data[cur_inds, :, :, :].detach())  # .to(self.device, non_blocking=True))
            pa_labels = self.train_labels[cur_inds].to(self.device, non_blocking=True)
            pa_sup_labels = self.sup_labels[cur_inds].to(self.device, non_blocking=True)

            # Assemble buffer
            all_images = torch.cat([clean_images.cpu()] + adv_images_list + [pa_images], dim=0)
            all_labels = torch.cat([clean_labels] * (self.pgd_adversary.n_target_classes + 2) + [pa_labels], dim=0)
            all_sup_labels = torch.cat(
                [zeros] + [ones] * (self.pgd_adversary.n_target_classes + 1) + [pa_sup_labels], dim=0)
            all_scales = torch.cat(
                [ones * self.comp_loss_params["scale"]["clean"]] + 
                [ones * self.comp_loss_params["scale"]["ada"]] * (self.pgd_adversary.n_target_classes + 1) + 
                [torch.ones_like(pa_sup_labels, device=self.device) * self.comp_loss_params["scale"]["pa"]], 
                dim=0)
        return (all_images, all_labels, all_sup_labels.detach().half(), all_scales.detach().half())

    def get_gradient_batch(self, images, labels, sup_labels, scale):
        # with open("/home/ubuntu/project/Adaptive-Smoothing/experiments/img.pt", 'wb') as save_file:
        #     pickle.dump((images.detach().cpu().numpy(), 
        #                  labels.detach().cpu().numpy(), 
        #                  sup_labels.detach().cpu().numpy(),
        #                  scale.detach().cpu().numpy()), save_file)
        preds, img_var, alphas = self.comp_model(images)

        # import matplotlib.pyplot as plt
        # plt.hist(alphas.detach().cpu().numpy(), bins=100)
        # # plt.hist(alphas[sup_labels==0].detach().cpu().numpy(), bins=50)
        # plt.show()
        # figname = "/log/CompModel/CIFAR-10_AT_Linf/V3_APGD_CC_BN_2/hist_alpha.pdf"
        # plt.savefig("/home/ubuntu/project/Adaptive-Smoothing" + figname)
        # import pdb; pdb.set_trace()
        
        loss = self.comp_loss(preds, labels, alphas, sup_labels, scale)
        assert not torch.isnan(loss)
        loss.backward()
        mean_beta_pos = alphas[sup_labels == 1].mean().item()
        mean_beta_neg = alphas[sup_labels == 0].mean().item()
        return preds, loss, (mean_beta_pos, mean_beta_neg)

    def step_opt_sch(self):
        self.optimizer.step()
        self.scheduler_cntr = (self.scheduler_cntr + 1) % SCHEDULER_FACTOR
        if self.scheduler_cntr == 0:
            with self.warmup_scheduler.dampening():
                self.scheduler.step()

    def train_epoch(self, eval_freq_iter=200, debug=False):
        if self.pgd_adversary.randomized_attack_settings["curriculum"]:
            self.step_rand_curriculum()

        total_loss, correct, total = [], 0, 0  # Epoch-wise information
        torch.manual_seed(20221105 + self.cur_ep)  # Set seed here so that we can control the suffling
        tepoch = tqdm(self.trainloader, unit="batch")
        if self.train_data is None:
            print("No pre-attacked data specified. Using adaptive attack only.")
        
        for batch_num, (images, labels) in enumerate(tepoch):
            tepoch.set_description(f"Epoch {self.cur_ep}")
            
            # Data buffer (to ensure correct distribution in BN)
            images, labels, sup_labels, scales = self.get_data_buffer(images, labels)
            rand_inds = torch.randperm(images.shape[0])
            mini_batch_size = int(np.ceil(images.shape[0] / self.n_mini_batches))

            # Perform training updates
            torch.cuda.empty_cache()
            mean_beta = np.zeros((2, self.n_mini_batches))
            if self.accum_grad == -1:  # Gradient update after all mini-batches
                self.optimizer.zero_grad()

            for ind in range(self.n_mini_batches):
                cur_inds = rand_inds[ind * mini_batch_size: (ind+1) * mini_batch_size]
                images_batch = images[cur_inds, :, :, :].to(self.device, non_blocking=True)
                labels_batch = labels[cur_inds]
                sup_labels_batch = sup_labels[cur_inds]
                scales_batch = scales[cur_inds]

                if self.accum_grad != -1 and (ind % self.accum_grad) == 0:
                    self.optimizer.zero_grad()

                # The *argument* "scale" is for compensating for the last batch's potentially smaller size
                preds, loss, (mean_beta[0, ind], mean_beta[1, ind]) = \
                    self.get_gradient_batch(images_batch, labels_batch, sup_labels_batch,
                                            scale=(images_batch.shape[0] / mini_batch_size) * scales_batch)

                if self.accum_grad != -1 and ((ind + 1) % self.accum_grad) == 0:
                    self.step_opt_sch()  # Optimizer and scheduler updates

                # Print training status
                total_loss += [loss.item()]
                total += (labels_batch.size(0))
                correct += ((preds.argmax(dim=1) == labels_batch).sum().item())

            if self.accum_grad == -1:  # Gradient update after all mini-batches
                self.step_opt_sch()  # Optimizer and scheduler updates

            # Log training stats
            EaP, EaN = mean_beta[0, :].mean(), mean_beta[1, :].mean()  # E[log(alpha)]
            SaP, SaN = mean_beta[0, :].std(), mean_beta[1, :].std()
            Tl = np.mean(total_loss[-self.n_mini_batches:])  # Training loss
            # TQDM postfix
            tepoch.set_postfix(loss=Tl.round(decimals=4))
            self.glob_batch_num = (self.cur_ep - 1) * len(tepoch) + batch_num + 1  # Global batch count
            # Tensorboard
            self.writer.add_scalar('training loss', Tl, self.glob_batch_num)
            self.writer.add_scalar('Minibatch size', mini_batch_size, self.glob_batch_num)
            self.writer.add_scalar('E[log(alpha)] for sup_labels=1', EaP, self.glob_batch_num)
            self.writer.add_scalar('E[log(alpha)] for sup_labels=0', EaN, self.glob_batch_num)
            self.writer.add_scalar('Std[log(alpha)] for sup_labels=1', SaP, self.glob_batch_num)
            self.writer.add_scalar('Std[log(alpha)] for sup_labels=0', SaN, self.glob_batch_num)
            self.writer.add_scalar('LR', self.optimizer.param_groups[0]['lr'], self.glob_batch_num)
            # Write to log file
            with open(os.path.join(self.save_path, "log_train.csv"), 'a') as wfile:
                wfile.write(f"Epoch {self.cur_ep}, "
                            f"Batch {1 + (batch_num if self.train_data is None else self.ba)} "
                            f"of {len(tepoch) if self.train_data is None else self.no_batches}:\n")
                wfile.write(f"Training loss: {Tl:.4f}.\n")
                wfile.write(f"E[log(alpha)] = {EaP:.3f} for sup_labels=1.\n")
                wfile.write(f"E[log(alpha)] = {EaN:.3f} for sup_labels=0.\n")

            if self.glob_batch_num % eval_freq_iter == 0 and self.glob_batch_num != 0:
                self.save_and_eval(debug=debug)

            # Re-shuffle pretrained attacks after going through all of them
            if self.train_data is not None:
                self.ba += 1  # This is the batch counter for the big (5-epoch) loop
                if self.ba >= self.no_batches:
                    self.ba -= self.no_batches
                    self.img_inds = torch.randperm(self.n)  # Reshuffle pre-attacked data

            if debug and self.ba % 2 == 0:  # Run two iterations 
                break
        print(f"Average training loss: {np.mean(total_loss):.4f}.")
        print(f"Average training accuracy: {(correct / total * 100):.2f} %.")

    def train(self, save_path, load_path=None, eval_freq={"epoch": 1, "iter": 200}, debug=False):
        print(f"Autocast enabled: {self._comp_model.enable_autocast}.")
        self.save_path = save_path  # Save folder path
        self.writer = SummaryWriter(self.save_path)  # TensorBoard writer
        print("Using TensorBoard writer.")
        self.comp_model.train()

        # Dump experiment settings
        with open(os.path.join(self.save_path, "settings.json"), 'a') as wfile:
            json.dump(self.setting_dics, wfile)

        # Shuffle pre-attacked data
        if self.train_data is not None:
            self.no_batches = int(np.ceil(self.n / self.batch_size_train))
            self.ba, self.img_inds = 0, torch.randperm(self.n)
        
        # Continue training from a checkpoint
        if load_path is not None:
            ep_start = self.load_checkpoint(load_path)
        else:
            ep_start = 1
        print(f"Start epoch is {ep_start}.")
        
        for cur_ep in range(ep_start, self.epochs + 1):
            self.cur_ep = cur_ep
            # If Step schedule, adjust the LR for each epoch manually
            if self.training_settings["schedule"] == "step":
                print("Adjusting the learning rate.")
                for p in self.optimizer.param_groups:
                    p['lr'] = self.training_settings["lr"] / (2 ** ((self.cur_ep - 1) // 2 - 1))
            print(f"Epoch {self.cur_ep}, LR is {self.optimizer.param_groups[0]['lr']}.")
            
            # Actual training for an epoch
            self._comp_model.const = self.bn_consts["train"]
            print(f"The output mean of the policy is set to {self._comp_model.const[1]:.1f} "
                  f"and the standard deviation is set to {self._comp_model.const[0]:.1f}.")
            self.train_epoch(eval_freq_iter=eval_freq["iter"], debug=debug)
            
            # Save current epoch model and evaluate
            if self.cur_ep % eval_freq["epoch"] == 0 or self.cur_ep == self.epochs:
                self.save_and_eval(debug=debug)

    def save_and_eval(self, debug=False):
        torch.cuda.empty_cache()
        os.makedirs(self.save_path, exist_ok=True)
        save_pth = os.path.join(self.save_path, f"epoch_{self.cur_ep}_ba_{self.glob_batch_num}.pt")
        print(f"The path of the saved file is: {save_pth}.")

        parallel = self.setting_dics["forward_settings"]["parallel"]
        sd_model = self.comp_model.policy_net.module if parallel == 1 else (
            self.comp_model.module.policy_net if parallel == 2 else self.comp_model.policy_net)
        torch.save({"model": sd_model.state_dict(), "bn": self._comp_model.bn.state_dict(), 
                    "ep": self.cur_ep, "img_inds": self.img_inds, "ba": self.ba,
                    "optimizer": self.optimizer.state_dict(), "scheduler": self.scheduler.state_dict(),
                    "warm_scheduler": self.warmup_scheduler.state_dict()}, save_pth)

        self.comp_model.eval()
        self._comp_model.const = self.bn_consts["eval"]
        self.evaluate(pgd_eps=self.pgd_adversary.pgd_eps, pgd_alpha=self.pgd_adversary.pgd_alpha_test,
                      pgd_iters=self.pgd_adversary.pgd_iters_test, save_adv_imgs=self.save_eval_imgs, 
                      auto_attack=self.use_aa, full=False, debug=debug) 
        self.comp_model.train()
        self._comp_model.const = self.bn_consts["train"]
        torch.cuda.empty_cache()

    def load_checkpoint(self, load_path, enable_BN=False, reset_scheduler=False):
        ep_start, ba, img_inds = utils.load_ckpt(self.comp_model, self.optimizer, self.scheduler, self.warmup_scheduler, 
                                                 lr=self.training_settings['lr'], load_path=load_path, enable_BN=enable_BN,
                                                 parallel=self.setting_dics["forward_settings"]["parallel"], 
                                                 reset_scheduler=reset_scheduler, device=self.device)

        if ep_start % 5 == 1 and self.n is not None:
            self.ba, self.img_inds = 0, torch.randperm(self.n)
        else:
            self.ba, self.img_inds = ba, img_inds
            if self.n is not None and (self.img_inds is None or self.img_inds.max().item() >= self.n):
                self.ba, self.img_inds = ep_start * len(self.testloader), torch.randperm(self.n)
        if self.n is not None:
            assert (len(self.img_inds) == self.n) and (self.n == self.img_inds.max() + 1)

        return ep_start
