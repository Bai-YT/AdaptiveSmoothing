import torch
import torch.nn as nn

import numpy as np
import pickle
from tqdm import tqdm

from adaptive_smoothing import attacks
from adaptive_smoothing.losses import DLRLoss, CompLoss

LOSSES = {'ce': nn.CrossEntropyLoss(), 'dlr': DLRLoss(), 'comp': CompLoss()}


def load_data(load_path):
    with open(load_path, 'rb') as readfile:
        data = pickle.load(readfile)
    if len(data["train"]) == 3:
        _, train_imageset, train_labelset = data["train"]
        _, test_imageset, test_labelset = data["test"]
    else:
        train_imageset, train_labelset = data["train"]
        test_imageset, test_labelset = data["test"]
    return (train_imageset if train_imageset is None else train_imageset.detach().cpu(), 
            train_labelset if train_labelset is None else train_labelset.detach().cpu(),
            test_imageset.detach().cpu(), test_labelset.detach().cpu())


def save_data(train_adv_imageset, train_labelset, 
              test_adv_imageset, test_labelset, dump_path):
    train_data = (train_adv_imageset, train_labelset)
    test_data = (test_adv_imageset, test_labelset)
    data = {"train": train_data, "test": test_data}
    with open(dump_path, 'wb') as dumpfile:
        pickle.dump(data, dumpfile, protocol=4)


def assemble_data(imageset, labelset, nested):
    if imageset is None or labelset is None:
        return None, None
    if nested:  # Flatten the list
        imageset = [item for sublist in imageset for item in sublist]
        labelset = [item for sublist in labelset for item in sublist]
    print("Assembling big tensor...")
    imageset = torch.cat(imageset, dim=0)
    labelset = torch.cat(labelset, dim=0)
    return imageset, labelset


def prepare_single_data_worker(model, loader, pgd_settings, device, debug=False):
    adv_imageset, labelset = [], []
    total_accuracy = 0

    titer = tqdm(loader, unit="batch")
    for ind1, (images, labels) in enumerate(titer):
        images, labels = images.to(device), labels.to(device)

        adv_images = attacks.single_pgd_attack(
            model, images, labels, device=device, mom_decay=0,
            pgd_type=pgd_settings["type"], pgd_eps=pgd_settings["eps"],
            pgd_alpha=pgd_settings["alpha_test"], pgd_iters=pgd_settings["iters_test"],
            pgd_loss=LOSSES[pgd_settings["pgd_eval_loss"]])
        adv_imageset += [adv_images.detach().cpu()]
        labelset += [labels.detach().cpu()]

        scores, _ = model(adv_images)
        cur_accuracy = (scores.argmax(dim=1) == labels).double().mean().item()
        total_accuracy += cur_accuracy

        # Update TQDM postfix
        titer.set_postfix(cur_accuracy=cur_accuracy*100.)
        if debug and ind1 == 1: 
            break

    print(f"Total accuracy: {total_accuracy / (ind1 + 1) * 100.:.2f} %.")
    return adv_imageset, labelset


def prepare_single_data(model, pgd_settings, trainloader, testloader,
                        dump_path, device, test_only=False, debug=False):

    print(f"Only prepare the test set: {test_only}")
    test_adv_imageset, test_labelset = prepare_single_data_worker(
        model, testloader, pgd_settings, device=device, debug=debug)
    if test_only:
        train_adv_imageset, train_labelset = None, None
    else:
        train_adv_imageset, train_labelset = prepare_single_data_worker(
            model, trainloader, pgd_settings, device=device, debug=debug)
    
    # Save data
    train_adv_imageset, train_labelset = assemble_data(
        train_adv_imageset, train_labelset, nested=False)
    test_adv_imageset, test_labelset = assemble_data(
        test_adv_imageset, test_labelset, nested=False)
    save_data(train_adv_imageset, train_labelset, test_adv_imageset, 
              test_labelset, dump_path=dump_path)


def prepare_comp_data_worker(comp_model, _comp_model, alphas, loader, 
                             pgd_settings, log_path, device, debug=False):
    adv_imageset, labelset = [[] for _ in alphas], [[] for _ in alphas]
    total_accuracies = [0 for _ in alphas]

    for ind1, alpha in enumerate(alphas):
        titer = tqdm(loader, unit="batch")
        for ind2, (images, labels) in enumerate(titer):
            titer.set_description(f"{ind1}, alpha={alpha:.4f}")
            images, labels = images.to(device), labels.to(device)
            _comp_model.alpha = alpha

            adv_images = attacks.comp_pgd_attack(
                comp_model, images, labels, mom_decay=0,
                pgd_type=pgd_settings["type"], pgd_eps=pgd_settings["eps"],
                pgd_alpha=pgd_settings["alpha_test"], pgd_iters=pgd_settings["iters_test"],
                pgd_loss=LOSSES[pgd_settings["pgd_eval_loss"]])
            adv_imageset[ind1] += [adv_images.detach().cpu()]
            labelset[ind1] += [labels.detach().cpu()]

            outputs, _, _ = comp_model(adv_images)
            cur_accuracy = (outputs.argmax(dim=1) == labels).double().mean().item()
            total_accuracies[ind1] += cur_accuracy
            titer.set_postfix(cur_accuracy=cur_accuracy * 100.)
            if debug and ind2 == 1: 
                break

        if log_path is not None:
            with open(log_path + "_" + str(ind1) + ".pickle", 'wb') as logfile:
                pickle.dump((adv_imageset[ind1], labelset[ind1]), logfile, protocol=4)
        print(f"Total accuracy for alpha={alpha:.3e}: "
              f"{total_accuracies[ind1] / (ind2 + 1) * 100.:.2f} %.")

    print("Total accuracy:", [acc / (ind2 + 1) * 100. for acc in total_accuracies])
    return adv_imageset, labelset


def prepare_comp_data(comp_model, _comp_model, alphas, pgd_settings, 
                      trainloader, testloader, log_path_train, log_path_test, 
                      dump_path, device, test_only=False, debug=False):
    print("alphas:", alphas)
    assert _comp_model.defense_type == pgd_settings["type"]
    print(f"Only prepare the test set: {test_only}")

    test_adv_imageset, test_labelset = prepare_comp_data_worker(
        comp_model, _comp_model, alphas, testloader, pgd_settings, 
        log_path_test, device=device, debug=debug)
    if test_only:
        train_adv_imageset, train_labelset = None, None
    else:
        train_adv_imageset, train_labelset = prepare_comp_data_worker(
            comp_model, _comp_model, alphas, trainloader, pgd_settings, 
            log_path_train, device=device, debug=debug)

    # Save data as list of tensors
    save_data(train_adv_imageset, train_labelset, 
              test_adv_imageset, test_labelset, dump_path=dump_path)


def replace_tensor_list(load_path, dump_path):
    with open(load_path, 'rb') as loadfile:
        data = pickle.load(loadfile)
    train_adv_imageset, train_labelset = data["train"]
    test_adv_imageset, test_labelset = data["test"]
    # Concatenate tensor list
    train_adv_imageset, train_labelset = assemble_data(
        train_adv_imageset, train_labelset, nested=True)
    test_adv_imageset, test_labelset = assemble_data(
        test_adv_imageset, test_labelset, nested=True)
    save_data(train_adv_imageset, train_labelset, 
              test_adv_imageset, test_labelset, dump_path=dump_path)


def assemble_policy_data(load_path_std, load_path_adv, load_path_comp, alphas, dump_path):
    (train_adv_imageset_clean, train_labelset_clean,
     test_adv_imageset_clean, test_labelset_clean) = load_data(load_path_std)
    (train_adv_imageset_adv, train_labelset_adv,
     test_adv_imageset_adv, test_labelset_adv) = load_data(load_path_adv)
    (train_adv_imageset_comp, train_labelset_comp,
     test_adv_imageset_comp, test_labelset_comp) = load_data(load_path_comp)

    if train_adv_imageset_comp is None:
        train_data, train_labels = None, None
    else:
        train_data = torch.cat([train_adv_imageset_clean, train_adv_imageset_adv,
                                train_adv_imageset_comp], dim=0).detach().cpu()
        train_labels = torch.cat([train_labelset_clean, train_labelset_adv,
                                train_labelset_comp], dim=0).detach().cpu()

    test_data = torch.cat([test_adv_imageset_clean, test_adv_imageset_adv,
                           test_adv_imageset_comp], dim=0).detach().cpu()
    test_labels = torch.cat([test_labelset_clean, test_labelset_adv,
                             test_labelset_comp], dim=0).detach().cpu()

    if alphas is not None:
        train_adv_alphas = torch.cat([torch.zeros(50000, ), torch.ones(50000, ) * np.inf] +
                               [torch.ones(50000, ) * alpha for alpha in alphas])
        test_adv_alphas = torch.cat([torch.zeros(10000, ), torch.ones(10000, ) * np.inf] +
                               [torch.ones(10000, ) * alpha for alpha in alphas])
        dataset = {"train_data": train_data, "train_labels": train_labels,
                   "test_data": test_data, "test_labels": test_labels,
                   "train_adv_alphas": train_adv_alphas, "test_adv_alphas": test_adv_alphas}
    else:
        dataset = {"train_data": train_data, "train_labels": train_labels,
                   "test_data": test_data, "test_labels": test_labels}

    with open(dump_path, 'wb') as dumpfile:
        pickle.dump(dataset, dumpfile, protocol=4)
