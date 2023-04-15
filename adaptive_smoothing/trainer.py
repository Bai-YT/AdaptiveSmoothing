import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
import pytorch_warmup as warmup

import pickle
import numpy as np
from tqdm import tqdm
import json
import os

from comp_autoattack import AutoAttack
from adaptive_smoothing import utils, attacks, losses
from models.comp_model import CompositeModelWrapper


class CompModelTrainer:
    train_data, train_labels = None, None
    test_data, test_labels = None, None
    adv_gammas, sup_labels = None, None
    n, n_test, transform = None, None, None
    no_batches, ba, img_inds, glob_batch_num = None, None, None, None
    writer, save_path = None, None  # Tensorboard writer
    print(f"Number of GPUs: {torch.cuda.device_count()}.")
    print(f"Number of CPUs: {os.cpu_count()}.")

    def __init__(self, forward_settings, std_load_path, rob_load_path, dataset_settings, 
                 training_settings, pgd_settings, randomized_attack_settings):
        # Comp model settings, for debugging purposes
        self.setting_dics = {
            "forward_settings": forward_settings, "dataset_settings": dataset_settings, 
            "training_settings": training_settings, "pgd_settings": pgd_settings, 
            "randomized_attack_settings:": randomized_attack_settings}

        # Loss functions
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')
        self.comp_loss_params = training_settings["comp_loss_params"]
        self.comp_loss = losses.CompLoss(consts=self.comp_loss_params["consts"])

        # Dataset settings
        self.batch_size_train = int(training_settings["batch_size_train"])
        self.batch_size_test = int(training_settings["batch_size_test"])
        self.num_blocks = dataset_settings["blocks"]
        self.train_with_test = dataset_settings["train_with_test"]
        self.use_fast_loader = training_settings["use_fast_loader"]
        print(f"Train on the {'test' if self.train_with_test else 'training'} set.")
        if dataset_settings["pa_path"] is not None:
            self.set_pa_dataset(dataset_settings["pa_path"], dataset_settings["gamma_0"])

        # Get loaders
        # Fasterloader has a hardcoded factor of 9 for now
        print("Dataset:", "CIFAR-100")
        cmdl_and_loaders = utils.get_cmodel_and_loaders(
            forward_settings=forward_settings,
            std_load_path=std_load_path, rob_load_path=rob_load_path,
            dataset_name=dataset_settings["name"],
            dataset_path=dataset_settings["path"],
            batch_size_train=self.batch_size_train * 5 // self.num_blocks,
            batch_size_test=self.batch_size_test * 5 // self.num_blocks,
            train_shuffle=dataset_settings["train_shuffle"],
            train_with_test=self.train_with_test)
        (comp_model, self.trainloader, self.testloader, self.trainloader_fast, 
         self.transform_train, self.transform_test) = cmdl_and_loaders

        # self.comp_model has one output for compatibility with RobustBench
        self.comp_model = CompositeModelWrapper(comp_model, parallel=forward_settings["parallel"])
        self._comp_model = self.comp_model._comp_model  # This is the unparallelized model

        # Trainer settings
        self.training_settings = training_settings
        self.accum_grad = self.training_settings["accum_grad"]
        self.epochs, self.cur_ep = self.training_settings["epochs"], 0
        self.gamma_consts = self.training_settings["gamma_consts"]
        self.save_eval_imgs = self.training_settings["save_eval_imgs"]

        self.n_mini_batches = (pgd_settings["n_target_classes"] + 1) * 2
        if self.accum_grad == self.n_mini_batches:
            self.accum_grad = -1
        if self.accum_grad != -1:
            assert self.n_mini_batches % self.accum_grad == 0

        # Batch count
        # if there is pre-attacked data, then this is for the big 5-epoch loop.
        # Otherwise, this is for the total batch count.
        self.ba = 0

        # Calculate the number of optimization steps
        base_len = (1 if self.accum_grad == -1 
                    else self.n_mini_batches // self.accum_grad)
        num_opt_steps = base_len * len(self.trainloader) * self.epochs
        print(f"Number of minibatches per batch: {self.n_mini_batches}. "
              f"Accummulate {self.accum_grad} grad evaluations.")
        print(f"Total number of optimization steps: {num_opt_steps}. "
              f"Warmup period: {num_opt_steps // 25}.")

        # Optimizer, scheduler, and grad scaler
        self.optimizer = torch.optim.AdamW(
            self.comp_model.parameters(), lr=self.training_settings["lr"],
            weight_decay=self.training_settings["weight_decay"])
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, num_opt_steps)
        self.warmup_scheduler = warmup.ExponentialWarmup(
            self.optimizer, warmup_period=num_opt_steps // 25)
        self.grad_scaler = torch.cuda.amp.GradScaler()

        # PGD settings
        self.pgd_eval_loss = pgd_settings["pgd_eval_loss"]
        self.use_apgd_training = randomized_attack_settings["apgd"]
        self.use_aa = pgd_settings["use_aa"]
        self.aa_batch_size = pgd_settings["aa_batch_size"]

        self.pgd_adversary = attacks.PGDAdversary(
            self.comp_model.comp_model, randomized_attack_settings, pgd_settings)
        if pgd_settings["type"] == 'l_inf':
            self.aa_adversary = AutoAttack(
                self.comp_model.comp_model, norm='Linf',
                eps=self.pgd_adversary.pgd_eps, version='standard')
        elif pgd_settings["type"] == 'l_2':
            self.aa_adversary = AutoAttack(
                self.comp_model.comp_model, norm='L2',
                eps=self.pgd_adversary.pgd_eps, version='standard')
        else:
            raise ValueError("Unknown attack norm type.")
        self.aa_adversary.attacks_to_run = pgd_settings["attacks_to_run"]

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print(f"Trainer device: {self.device}.")

    def set_pa_dataset(self, dataset_path, gamma_0):
        print(f"Loading pre-attacked data from {dataset_path}")
        with open(dataset_path, 'rb') as readfile:
            dataset = pickle.load(readfile)

        # Pre-attacked training data
        if not self.train_with_test:
            self.pa_train_data = dataset["train_data"]
            self.pa_train_labels = dataset["train_labels"]
        else:
            self.pa_train_data = dataset["test_data"]
            self.pa_train_labels = dataset["test_labels"]
        # Pre-attacked test data
        self.pa_test_data = dataset["test_data"]
        self.pa_test_labels = dataset["test_labels"]

        # Gamma values used for generating pre-attacked data
        self.adv_gammas = dataset[
            "test_adv_gammas" if self.train_with_test else "train_adv_gammas"]
        gammas = [
            self.adv_gammas[int(i * (10000 if self.train_with_test else 50000))].item()
            for i in range(2, self.num_blocks)]
        print(f"Pre-attacked gammas: {gammas}.")
        # Labels for the policy network
        # If adv_gammas < gamma_0,
        #     then the ROB network should be used, and sup_label should be 1.
        # If adv_gammas >= gamma_0,
        #     then the STD network should be used, and sup_label should be 0.
        self.pa_sup_labels = (self.adv_gammas < gamma_0).half()

        if self.pa_train_data is not None:
            self.n = self.pa_train_data.shape[0]
        else:
            print("There is no pre-attacked training data.")
        self.n_test = self.pa_test_data.shape[0]
        self.transform = transforms.Compose([transforms.RandomHorizontalFlip(p=.5),
                                             transforms.RandomCrop(32, padding=4)])
        if self.pa_train_data is not None:
            assert self.num_blocks == self.n // (10000 if self.train_with_test else 50000)
            assert self.n % (10000 if self.train_with_test else 50000) == 0
            assert self.n == self.n_test * (1 if self.train_with_test else 5)
            assert self.n == self.pa_train_labels.shape[0]
            assert self.n == self.pa_sup_labels.shape[0]
            assert self.n_test == self.pa_test_labels.shape[0]
            self.img_inds = torch.randperm(self.n)  # For randomly using the pre-attacked data
            self.no_batches = int(np.ceil(self.n / self.batch_size_train))

        print(f"Training instances count: {self.n}, test instances count: {self.n_test}.")
        print(f"Number of pre-attacked blocks: {self.num_blocks}.\n")

    def log_eval_stats(self, total_loss, correct, gamma_mean, gamma_large, total, 
                       clean_only=False, auto_attack=True):

        # Log validation stats with clean and adaptive attack
        clean_loss, clean_acc = np.mean(total_loss["clean"]), correct["clean"] / total
        clean_gamma_mean = np.mean(gamma_mean["clean"]) 
        clean_gamma_acc = 1 - (gamma_large["clean"] / total)
        print(f"Average clean test accuracy: {(clean_acc * 100.):.2f} %, "
              f"mean gamma: {clean_gamma_mean:.3f}, "
              f"gamma accuracy: {(clean_gamma_acc * 100.):.2f} %.")

        if not clean_only:
            adv_loss, adv_acc = np.mean(total_loss["adv"]), correct["adv"] / total
            adv_gamma_mean = np.mean(gamma_mean["adv"])
            adv_gamma_acc = gamma_large["adv"] / total
            print(f"Average adv test accuracy: {(adv_acc * 100.):.2f} %, "
                  f"mean gamma: {adv_gamma_mean:.3f}, "
                  f"gamma accuracy: {(adv_gamma_acc * 100.):.2f} %.\n")

        if self.writer is not None:
            log_ba = (self.cur_ep - 1) * len(self.trainloader) + \
                (self.glob_batch_num - 1) % len(self.trainloader) + 1
            self.writer.add_scalar("Clean val loss", clean_loss, log_ba)
            self.writer.add_scalar("Clean val acc", clean_acc * 100., log_ba)
            self.writer.add_scalar("Clean mean gamma", clean_gamma_mean, log_ba)
            self.writer.add_scalar("Clean gamma acc", clean_gamma_acc * 100., log_ba)

            if not clean_only:
                self.writer.add_scalar("Adaptive adv val loss", adv_loss, log_ba)
                self.writer.add_scalar("Adaptive adv val acc", adv_acc * 100., log_ba)
                self.writer.add_scalar("Adaptive adv mean gamma", adv_gamma_mean, log_ba)
                self.writer.add_scalar(
                    "Eval gamma mean difference", adv_gamma_mean - clean_gamma_mean, log_ba)
                self.writer.add_scalar("Adaptive adv gamma acc", adv_gamma_acc * 100., log_ba)

        if self.save_path is None:
            print("No save path provided. Validation results not saved.")

        else:
            os.makedirs(self.save_path, exist_ok=True)
            with open(os.path.join(self.save_path, "log_eval_ada.csv"), 'a') as wfile:
                wfile.write(
                    f"Epoch {self.cur_ep}, clean_only: {clean_only}, auto_attack: {auto_attack}\n")
                wfile.write("Clean data\n")
                wfile.write(f"Average clean test loss: {clean_loss:.4f}.\n")
                wfile.write(f"Average clean test accuracy: {(clean_acc * 100.):.2f} %.\n")
                wfile.write(f"Average clean mean gamma: {clean_gamma_mean:.3f}.\n")
                wfile.write(f"Average clean gamma acc: {(clean_gamma_acc * 100.):.2f} %.\n")

                if not clean_only:
                    wfile.write("Adaptive adversary\n")
                    wfile.write(f"Average attacked test loss: {adv_loss:.4f}.\n")
                    wfile.write(f"Average attacked test accuracy: {(adv_acc * 100.):.2f} %.\n")
                    wfile.write(f"Average attacked mean gamma: {adv_gamma_mean:.3f}.\n")
                    wfile.write(f"Average attacked gamma acc: {(adv_gamma_acc * 100.):.2f} %.\n\n")

        if clean_only:
            return clean_loss, clean_acc, None, None
        return clean_loss, clean_acc, adv_loss, adv_acc

    def get_loss_acc(self, images, labels):
        with torch.no_grad():
            preds, gammas = self.comp_model.comp_model(images)
            loss = self.ce_loss(preds, labels).mean().item()
            correct = (preds.argmax(dim=1) == labels).sum().item()
        return loss, correct, gammas.mean().item(), (gammas>self.gamma_consts['eval'][1]).sum().item()

    def evaluate_adap(self, pgd_eps=8./255., pgd_alpha=0.0027, pgd_iters=20,
                      start_ind=0, save_adv_imgs=False, clean_only=False,
                      auto_attack=True, verbose=False, full=True, debug=False):
        self.comp_model.eval()
        if self._comp_model.use_policy:
            self._comp_model.set_gamma_scale_bias(*self.gamma_consts["eval"])
            print("Using the policy network.")
            print(f"Gamma scale: {self._comp_model.gamma_scale.item():.3f}, "
                  f"Gamma bias: {self._comp_model.gamma_bias.item():.3f}.")
            print(f"Alpha range: ({self._comp_model.alpha_bias.item():.3f}, "
                  f"{(self._comp_model.alpha_scale + self._comp_model.alpha_bias).item():.3f}).")
        else:
            print(f"Using a constant gamma of {self._comp_model.gamma}.")

        total_loss, correct = {"clean": [], "adv": []}, {"clean": 0, "adv": 0}
        gamma_mean, gamma_large = {"clean": [], "adv": []}, {"clean": 0, "adv": 0}
        all_adv_images, total = [], 0

        titer = tqdm(self.testloader, unit="batch", disable=verbose)  # If verbose, disable TQDM
        for ind, (images, labels) in enumerate(titer):
            titer.set_description(f"Epoch {self.cur_ep}, adaptive evaluation.")

            if ind >= start_ind:
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                if clean_only:
                    adv_images = None
                else:
                    if auto_attack:
                        adv_images = self.aa_adversary.run_standard_evaluation(
                            images, labels, bs=self.aa_batch_size, perform_checks=False)
                    else:
                        adv_images = self.pgd_adversary.pgd_attack_eval(
                            images, labels, pgd_type=self.pgd_adversary.pgd_type, pgd_eps=pgd_eps, 
                            pgd_alpha=pgd_alpha, pgd_iters=pgd_iters, loss_type=self.pgd_eval_loss, 
                            random_start=False)

                if save_adv_imgs and not clean_only:
                    try:
                        all_adv_images += [adv_images.detach().cpu().numpy()]
                    except:
                        print("Something went wrong with saving all adv images.")

                for key, imgs in zip(["clean", "adv"], [images, adv_images]):
                    res = self.get_loss_acc(imgs, labels)
                    cur_loss, cur_correct, cur_gamma_mean, cur_gamma_large = res
                    total_loss[key] += [cur_loss]
                    correct[key] += cur_correct
                    gamma_mean[key] += [cur_gamma_mean]
                    gamma_large[key] += cur_gamma_large
                    torch.cuda.empty_cache()

                    if verbose: 
                        print(f"Batch {ind + 1} of {len(self.testloader)}, "
                              f"{key} correct: {cur_correct} out of {labels.size(0)}, ")
                        print(f"mean gamma: {np.mean(gamma_mean[key]):.3f}, "
                              f"large gamma: {cur_gamma_large} out of {labels.size(0)}.")
                    if clean_only:
                        break
                total += labels.size(0)

                if clean_only:
                    titer.set_postfix(
                        clean_acc=correct["clean"] / total * 100., 
                        clean_loss=np.mean(total_loss["clean"]), 
                        gamma_acc=(1 - gamma_large["clean"] / total) * 100.)
                else:
                    titer.set_postfix(
                        clean_acc=correct["clean"] / total * 100., 
                        adv_acc=correct["adv"] / total * 100.,
                        clean_gamma=np.mean(gamma_mean["clean"]), 
                        adv_gamma=np.mean(gamma_mean["adv"]))

            # If debug, then run only one batch. 
            # Otherwise, if the full flag is not set, use 500 images (5% of the test set).
            if (debug and ind >= 0) or (
                not full and (ind * self.batch_size_test * 5 // self.num_blocks) >= 500): 
                break

        # Save all adversarial images
        if self.save_path is not None and save_adv_imgs:
            fname = (f"adv_imgs_ep_{self.cur_ep}_ba_{self.glob_batch_num}"
                     f"{'_full' if full else ''}{'_aa' if auto_attack else '_pgd'}.pkl")
            pth = os.path.join(self.save_path, fname)
            with open(pth, 'wb') as wfile:
                pickle.dump(all_adv_images, wfile)

        # Log eval statistics
        return self.log_eval_stats(
            total_loss, correct, gamma_mean, gamma_large, total, clean_only=False, auto_attack=True)

    def evaluate_data(self, debug=False):
        if self.pa_test_data is None:
            print("No pre-attacked data specified. Test with existing data not performed.")
            return

        self.comp_model.eval()
        if self._comp_model.use_policy:
            self._comp_model.set_gamma_scale_bias(*self.gamma_consts["eval"])
        else:
            print(f"Using a constant gamma of {self._comp_model.gamma}.")

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

                images = self.pa_test_data[
                    ba * real_bs: (ba + 1) * real_bs, :, :, :].detach().to(
                        self.device, non_blocking=True)
                labels = self.pa_test_labels[
                    ba * real_bs: (ba + 1) * real_bs].to(self.device, non_blocking=True)

                with torch.no_grad():
                    preds = self.comp_model(images)
                    loss = self.ce_loss(preds, labels).mean()

                total_loss += [loss.item()]
                correct += (preds.argmax(dim=1) == labels).sum().item()
                if debug and ba % 2 == 1: 
                    break

            # Log validation stats with pre-attacked data
            block_loss[ind] = np.mean(total_loss)
            block_acc[ind] = correct / self.n_test * self.num_blocks
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
                 pgd_iters=None, full=True, clean_only=False, auto_attack=True,
                 eval_pa_data=True, save_adv_imgs=False, debug=False):
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
        res = self.evaluate_adap(
            pgd_eps=pgd_eps, pgd_alpha=pgd_alpha, pgd_iters=pgd_iters,
            save_adv_imgs=save_adv_imgs, auto_attack=auto_attack, full=full,
            clean_only=clean_only, verbose=auto_attack, debug=debug)

        if eval_pa_data:
            self.evaluate_data(debug=debug)
        return res

    def get_data_buffer(self, images, labels):
        # Fast loader gives more clean data to the model to stabilize the training.
        if self.use_fast_loader:
            clean_images_2, clean_labels_2 = next(iter(self.trainloader_fast))
            clean_labels_2 = clean_labels_2.to(self.device, non_blocking=True)

        clean_images = images.to(self.device, non_blocking=True)
        clean_labels = labels.to(self.device, non_blocking=True)
        zeros = torch.zeros_like(clean_labels, device=self.device)
        ones = torch.ones_like(clean_labels, device=self.device)

        # Attack before zero_grad so that we don't remove the clean gradients
        self.comp_model.eval(scale_alpha=False)
        if self.use_apgd_training:
            adv_images_list = self.pgd_adversary.randomized_apgd_attack(clean_images, clean_labels)
        else:
            adv_images_list = self.pgd_adversary.randomized_pgd_attack(clean_images, clean_labels)
        self.comp_model.train(scale_alpha=False)
        # print("Done with attacking.")

        if self.pa_train_data is None:  # No pre-attacked data
            # Assemble buffer: [clean images, clean images from fast loader, attacked images]
            all_images = torch.cat(
                [clean_images.cpu()] + ([clean_images_2] if self.use_fast_loader else [])
                + adv_images_list, dim=0)
            all_labels = torch.cat(
                [clean_labels] + ([clean_labels_2] if self.use_fast_loader else []) +
                [clean_labels] * (self.pgd_adversary.n_target_classes + 1), dim=0)
            all_sup_labels = torch.cat(  # Assuming the fast loader has a factor of 9
                [zeros] * (10 if self.use_fast_loader else 1) +
                [ones] * (self.pgd_adversary.n_target_classes + 1), dim=0)
            all_scales = torch.cat(
                [ones * self.comp_loss_params["scale"]["clean"]] *
                (10 if self.use_fast_loader else 1) +  # Assuming the fast loader has a factor of 9
                [ones * self.comp_loss_params["scale"]["ada"]] *
                (self.pgd_adversary.n_target_classes + 1), dim=0
            ) / (self.n_mini_batches if self.accum_grad == -1 else self.accum_grad)

        else:  # There are pre-attacked data
            assert not self.use_fast_loader  # Incompatible
            cur_inds = self.img_inds[
                self.ba * self.batch_size_train: (self.ba + 1) * self.batch_size_train]
            pa_images = self.transform(self.pa_train_data[cur_inds, :, :, :].detach())
            pa_labels = self.pa_train_labels[cur_inds].to(self.device, non_blocking=True)
            pa_sup_labels = self.pa_sup_labels[cur_inds].to(self.device, non_blocking=True)

            # Assemble buffer
            all_images = torch.cat([clean_images.cpu()] + adv_images_list + [pa_images], dim=0)
            all_labels = torch.cat(
                [clean_labels] * (self.pgd_adversary.n_target_classes + 2) + [pa_labels], dim=0)
            all_sup_labels = torch.cat(
                [zeros] + [ones] * (self.pgd_adversary.n_target_classes + 1) + 
                [pa_sup_labels], dim=0)
            all_scales = torch.cat(
                [ones * self.comp_loss_params["scale"]["clean"]] + 
                [ones * self.comp_loss_params["scale"]["ada"]] * 
                (self.pgd_adversary.n_target_classes + 1) + 
                [torch.ones_like(pa_sup_labels, device=self.device) * 
                 self.comp_loss_params["scale"]["pa"]], dim=0)

        return (all_images, all_labels, all_sup_labels.detach().half(), all_scales.detach().half())

    def log_training_stats(self, train_loss, EgammaP, EgammaN, gammaAccP, gammaAccN, 
                           mini_batch_size, batch_num, epoch_length):
        # Tensorboard
        self.writer.add_scalar('Training loss', train_loss, self.glob_batch_num)
        self.writer.add_scalar('Minibatch size', mini_batch_size, self.glob_batch_num)
        self.writer.add_scalar('E[gamma] for sup_labels=1', EgammaP, self.glob_batch_num)
        self.writer.add_scalar('E[gamma] for sup_labels=0', EgammaN, self.glob_batch_num)
        self.writer.add_scalar('P[gamma > 0] for sup_labels=1', gammaAccP, self.glob_batch_num)
        self.writer.add_scalar('P[gamma <= 0] for sup_labels=0', gammaAccN, self.glob_batch_num)
        self.writer.add_scalar(
            'Learning rate', self.optimizer.param_groups[0]['lr'], self.glob_batch_num)
        self.writer.add_scalar('Grad scale', self.grad_scaler.get_scale(), self.glob_batch_num)

        # Write to log file
        with open(os.path.join(self.save_path, "log_train.csv"), 'a') as wfile:
            wfile.write(f"Epoch {self.cur_ep}, "
                        f"Batch {1 + (batch_num if self.pa_train_data is None else self.ba)} "
                        f"of {epoch_length if self.pa_train_data is None else self.no_batches}:\n")
            wfile.write(f"Training loss: {train_loss:.4f}.\n")
            wfile.write(f"E[gamma] = {EgammaP:.3f} for sup_labels=1.\n")
            wfile.write(f"E[gamma] = {EgammaN:.3f} for sup_labels=0.\n")
            wfile.write(f"P[gamma > 0] = {gammaAccP:.3f} for sup_labels=1.\n")
            wfile.write(f"P[gamma <= 0] = {gammaAccN:.3f} for sup_labels=0.\n")

    def get_gradient_batch(self, images, labels, sup_labels, scale):
        preds, gammas = self.comp_model.comp_model(images)
        loss = self.comp_loss(preds, labels, gammas, sup_labels, scale)
        assert not torch.isnan(loss)  # Stop training if loss is NaN
        self.grad_scaler.scale(loss).backward()

        mean_gamma_pos = gammas[sup_labels == 1].mean().item()
        mean_gamma_neg = gammas[sup_labels == 0].mean().item()
        gamma_acc_pos = (gammas[sup_labels == 1] > 0).mean().item()
        gamma_acc_neg = (gammas[sup_labels == 0] <= 0).mean().item()
        return preds, loss, (mean_gamma_pos, mean_gamma_neg, gamma_acc_pos, gamma_acc_neg)

    def step_opt_sch(self):
        # Optimizer step
        self.grad_scaler.unscale_(self.optimizer)
        nn.utils.clip_grad_norm_(self.comp_model.parameters(), max_norm=10.)
        self.grad_scaler.step(self.optimizer)

        # Grad scaler step
        old_scale = self.grad_scaler.get_scale()
        self.grad_scaler.update()
        skip_lr_sched = (old_scale > self.grad_scaler.get_scale())

        # LR scheduler step
        # Skip if grad scaler decreased the scale, which means we skipped the step
        if not skip_lr_sched:
            with self.warmup_scheduler.dampening():
                self.scheduler.step()

    def train_epoch(self, eval_freq_iter=200, debug=False):
        # Set seed here so that we can control the shuffling
        utils.seed_all(20221105 + self.cur_ep)
        if self.pa_train_data is None:
            print("No pre-attacked data specified. Using adaptive attack only.")

        total_loss, correct, total = [], 0, 0  # Epoch-wise information
        tepoch = tqdm(self.trainloader, unit="batch")

        for batch_num, (images, labels) in enumerate(tepoch):
            tepoch.set_description(f"Epoch {self.cur_ep}")

            # Data buffer (to ensure correct distribution in BN)
            images, labels, sup_labels, scales = self.get_data_buffer(images, labels)
            rand_inds = torch.randperm(images.shape[0])
            mini_batch_size = int(np.ceil(images.shape[0] / self.n_mini_batches))

            # Perform training updates
            torch.cuda.empty_cache()
            mean_gamma = np.zeros((2, self.n_mini_batches))
            gamma_acc = np.zeros((2, self.n_mini_batches))
            if self.accum_grad == -1:  # Gradient update after all mini-batches
                self.optimizer.zero_grad()  # Zero grad before all mini-batches

            for ind in range(self.n_mini_batches):
                cur_inds = rand_inds[ind * mini_batch_size: (ind+1) * mini_batch_size]
                images_batch = images[cur_inds, :, :, :].to(self.device, non_blocking=True)
                labels_batch = labels[cur_inds]
                sup_labels_batch = sup_labels[cur_inds]  # Pseudo labels for gamma
                scales_batch = scales[cur_inds]

                if self.accum_grad != -1 and (ind % self.accum_grad) == 0:
                    self.optimizer.zero_grad()

                # The argument "scale" is for compensating for 
                #     the last batch's potentially smaller size
                stats = self.get_gradient_batch(
                    images_batch, labels_batch, sup_labels_batch,
                    scale=(images_batch.shape[0] / mini_batch_size) * scales_batch)
                preds, loss, (mean_gamma[0, ind], mean_gamma[1, ind], 
                              gamma_acc[0, ind], gamma_acc[1, ind]) = stats

                if self.accum_grad != -1 and ((ind + 1) % self.accum_grad) == 0:
                    self.step_opt_sch()  # Optimizer and scheduler updates

                # Print training status
                total_loss += [loss.item()]
                total += (labels_batch.size(0))
                correct += ((preds.argmax(dim=1) == labels_batch).sum().item())

            if self.accum_grad == -1:  # Gradient update after all mini-batches
                self.step_opt_sch()  # Optimizer, scheduler, and scaler updates

            # Global batch count
            self.glob_batch_num = (self.cur_ep - 1) * len(tepoch) + batch_num + 1

            # Average training loss
            train_loss = np.mean(total_loss[-self.n_mini_batches:])
            # TQDM postfix
            tepoch.set_postfix(loss=train_loss.round(decimals=4),
                               lr=self.optimizer.param_groups[0]['lr'])
            # Save training statistics
            self.log_training_stats(train_loss,
                EgammaP=mean_gamma[0, :].mean(), EgammaN=mean_gamma[1, :].mean(),
                gammaAccP=gamma_acc[0, :].mean(), gammaAccN=gamma_acc[1, :].mean(),
                mini_batch_size=mini_batch_size, batch_num=batch_num, epoch_length=len(tepoch))

            # Save and evaluate model at the given interval
            if self.glob_batch_num % eval_freq_iter == 0 and self.glob_batch_num != 0:
                self.save_and_eval(debug=debug)

            # If there is pre-attacked data, this is the batch counter for the big (5-epoch) loop
            # Otherwise, this is the total batch count
            self.ba += 1

            # If there is pre-attacked data, re-shuffle after going through all of them
            if self.pa_train_data is not None:
                if self.ba >= self.no_batches:
                    self.ba -= self.no_batches  # Reset batch counter
                    self.img_inds = torch.randperm(self.n)  # Re-shuffle pre-attacked data

            if debug and self.ba % 2 == 0:  # Only run two iterations when debugging
                break

        print(f"Average training loss: {np.mean(total_loss):.4f}.")
        print(f"Average training accuracy: {(correct / total * 100):.2f} %.")

    def train(self, save_path, load_path=None, eval_freq={"epoch": 1, "iter": 200}, debug=False):

        print(f"Autocast enabled: {self._comp_model.enable_autocast}.")
        self.comp_model.train()
        self.save_path = save_path  # Save folder path
        self.writer = SummaryWriter(self.save_path)  # TensorBoard writer

        # Dump experiment settings
        with open(os.path.join(self.save_path, "settings.json"), 'a') as wfile:
            json.dump(self.setting_dics, wfile)

        # Shuffle pre-attacked data
        if self.pa_train_data is not None:
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
            print(f"Epoch {self.cur_ep}, LR is {self.optimizer.param_groups[0]['lr']}.")

            # Train for an epoch
            self._comp_model.set_gamma_scale_bias(*self.gamma_consts["train"])
            self.train_epoch(eval_freq_iter=eval_freq["iter"], debug=debug)

            # Save current epoch model and evaluate
            if self.cur_ep % eval_freq["epoch"] == 0 or self.cur_ep == self.epochs:
                self.save_and_eval(debug=debug)

    def save_and_eval(self, debug=False):
        torch.cuda.empty_cache()
        self.comp_model.eval()
        # Always save with the eval-mode BN const
        self._comp_model.set_gamma_scale_bias(*self.gamma_consts["eval"])

        os.makedirs(self.save_path, exist_ok=True)
        save_pth = os.path.join(self.save_path, f"epoch_{self.cur_ep}_ba_{self.glob_batch_num}.pt")
        print(f"The path of the saved file is: {save_pth}.")

        torch.save({
            "model": self._comp_model.policy_net.state_dict(), "bn": self._comp_model.bn.state_dict(), 
            "ep": self.cur_ep, "img_inds": self.img_inds, "ba": self.ba,
            "optimizer": self.optimizer.state_dict(), "scheduler": self.scheduler.state_dict(),
            "warmup_scheduler": self.warmup_scheduler.state_dict(), 
            "grad_scaler": self.grad_scaler.state_dict()}, save_pth)

        self.evaluate(
            pgd_eps=self.pgd_adversary.pgd_eps, pgd_alpha=self.pgd_adversary.pgd_alpha_test,
            pgd_iters=self.pgd_adversary.pgd_iters_test, save_adv_imgs=self.save_eval_imgs, 
            auto_attack=self.use_aa, full=False, debug=debug)

        self.comp_model.train()
        self._comp_model.set_gamma_scale_bias(*self.gamma_consts["train"])
        torch.cuda.empty_cache()

    def load_checkpoint(self, load_path, enable_BN=False, reset_scheduler=False):
        ep_start, ba, img_inds = utils.load_ckpt(
            self.comp_model, self.optimizer, self.scheduler, self.warmup_scheduler, self.grad_scaler,
            lr=self.training_settings['lr'], load_path=load_path, batch_per_ep=len(self.trainloader),
            enable_BN=enable_BN, reset_scheduler=reset_scheduler, device=self.device)

        if ep_start % 5 == 1 and self.n is not None:
            self.ba, self.img_inds = 0, torch.randperm(self.n)
        else:
            self.ba, self.img_inds = ba, img_inds
            if self.n is not None and (
                self.img_inds is None or self.img_inds.max().item() >= self.n):
                self.ba, self.img_inds = ep_start * len(self.testloader), torch.randperm(self.n)

        if self.n is not None:
            assert (len(self.img_inds) == self.n) and (self.n == self.img_inds.max() + 1)

        return ep_start
