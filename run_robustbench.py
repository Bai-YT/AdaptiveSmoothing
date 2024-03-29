import torch
import numpy as np  # Import numpy to use np.inf
import click
from robustbench import benchmark
from os.path import join
from os import makedirs

from adaptive_smoothing import utils
from models.comp_model import CompositeModelWrapper
from adaptive_smoothing.utils import seed_all


def get_pretrained_model(
    root_dir, std_load_path, rob_load_path, comp_load_path,
    model_name, dataset, forward_settings
):
    comp_load_path = join(root_dir, comp_load_path) \
        if comp_load_path is not None else None
    std_load_path = join(root_dir, std_load_path)
    rob_load_path = join(root_dir, rob_load_path)

    # Build CompositeModel
    num_classes = 10 if dataset == 'cifar10' else 100
    comp_model = utils.get_comp_model(
        forward_settings=forward_settings, num_classes=num_classes,
        std_load_path=std_load_path, rob_load_path=rob_load_path
    )

    # Load CompositeModel checkpoint
    if comp_load_path is not None:
        state_dict = torch.load(comp_load_path, map_location='cpu')

        # Load policy network
        comp_model.policy_net.load_state_dict(state_dict["model"])
        # Load BN stats
        comp_model.bn.load_state_dict(state_dict["bn"])

    # Set gamma and alpha mean and std div, as well as base model scales.
    if dataset == 'cifar100' and model_name == 'edm':  # EDM
        comp_model.set_gamma_value(gamma=2.5)
        comp_model.set_gamma_scale_bias(gamma_scale=2., gamma_bias=1.3)
        comp_model.set_alpha_scale_bias(alpha_scale=.15, alpha_bias=.84)
        comp_model.set_base_model_scale(std_scale=2., rob_scale=1.)
    elif dataset == 'cifar100' and model_name == 'trades':  # TRADES
        comp_model.set_gamma_value(gamma=2.)
        comp_model.set_gamma_scale_bias(gamma_scale=2., gamma_bias=1.)
        comp_model.set_alpha_scale_bias(alpha_scale=.1, alpha_bias=.815)
        comp_model.set_base_model_scale(std_scale=2., rob_scale=1.)
    elif dataset == 'cifar10' and model_name == 'edm':  # CIFAR-10
        comp_model.set_gamma_value(gamma=3.)
        comp_model.set_gamma_scale_bias(gamma_scale=2., gamma_bias=1.05)
        comp_model.set_alpha_scale_bias(alpha_scale=.04, alpha_bias=.96)
        comp_model.set_base_model_scale(std_scale=1.2, rob_scale=.3)
    else:
        raise ValueError(
            f"Unknown model name and dataset combination: {model_name}."
        )

    # Return wrapped model
    return CompositeModelWrapper(comp_model, parallel=forward_settings["parallel"])


@click.command(context_settings={'show_default': True})
@click.option(
    '--root_dir', default=".", show_default=True,
    help="Path to the root directory that stores the models"
)
@click.option(
    '--model_name', type=click.Choice(['edm', 'trades']), required=True,
    help="Model name (one of {'edm', 'trades'})."
)
@click.option(
    '--dataset', type=click.Choice(['cifar10', 'cifar100', 'imagenet']),
    required=True, help="Dataset (one of {'cifar10', 'cifar100'})."
)
@click.option(
    '--n_examples', type=int, default=10000,
    help="Number of test examples. Default to 10000."
)
@click.option(
    '--fp16/--fp32', default=False, show_default=True,
    help="Use mixed precision (fp16) or not (fp32)."
)

def run_robustbench(root_dir, model_name, dataset, n_examples, fp16):
    threat_model = "Linf"  # one of {"Linf", "L2", "corruptions"}
    assert not (dataset == 'cifar10' and model_name == 'trades')

    std_load_path = f"Base/{dataset}_bit_rn152.pt"
    rob_load_path = f"Base/{dataset}_linf_{model_name}_wrn70-16.pt"
    comp_load_path = f"CompModel/{dataset}_{model_name}_best.pt"

    forward_settings = {
        "std_model_type": 'rn152',
        "rob_model_type": 'wrn7016_silu' if model_name == 'edm' else 'wrn7016',
        "in_planes": (512, 256),
        # alpha = 0.86 gets 80% clean acc for DeepMind's TRADES WRN-70-16.
        # alpha = 0.925 gets 8x% clean acc for EDM WRN-70-16.
        "gamma": 0,  # Overriden by the mixing network.
        "use_policy": True,  # False if no mixing network
        "policy_graph": True,  # False if no mixing network
        "pn_version": 4,
        "parallel": True
    }

    model_full_name = f"bai2023improving_{model_name}_wrn7016"
    model = get_pretrained_model(
        root_dir, std_load_path, rob_load_path, comp_load_path,
        model_name, dataset, forward_settings
    )
    model.eval()
    model.requires_grad_(False)

    # Save state dict
    makedirs(join("model_info", dataset, threat_model), exist_ok=True)
    torch.save(
        model.cpu().state_dict(),
        f"model_info/{dataset}/{threat_model}/{model_full_name}.pt"
    )

    # Run RobustBench benchmark!
    seed_all(20230331)
    model._comp_model.enable_autocast = fp16
    batch_size = 40 * torch.cuda.device_count()
    clean_acc, robust_acc = benchmark(
        model, model_name=model_full_name, n_examples=n_examples,
        dataset=dataset, batch_size=batch_size, threat_model=threat_model,
        eps=8/255, device=torch.device("cuda:0"), to_disk=True
    )


if __name__ == "__main__":
    run_robustbench()
