import torch
from transformers import AutoModelForCausalLM
from typing import Literal, List
from dataclasses import dataclass


@dataclass
class NoiseConfig:
    """Configuration for noise application"""

    noise_type: Literal["pcm", "additive"]
    additive_std_multiplier: float = 0.05
    max_inp_size: int = -1


def polyval(p, x):
    """
    Implementation in torch of the numpy polyval function.
    """

    p = torch.as_tensor(p)
    if isinstance(x, torch.Tensor):
        y = torch.zeros_like(x)
    else:
        x = torch.as_tensor(x)
        y = torch.zeros_like(x)
    for pv in p:
        y = y * x + pv

    y = y.to(torch.float16)

    # check for overflow/underflow in FP16
    if torch.any(torch.isinf(y)) or torch.any(torch.isnan(y)):
        raise RuntimeError(f"Overflow/NaN detected after FP16 conversion: {y}")

    return y


def get_split_sizes(size: int, split_max_size: int) -> List[int]:
    """Computed the split sizes.

    Args:
        size: number of elements of the layer in one dimension
        split_max_size: max size of the split

    Returns:
        List of split sizes
    """
    if split_max_size <= 0:
        return [size]

    n_splits = (size + split_max_size - 1) // split_max_size
    base, extra = divmod(size, n_splits)
    return [base + (i < extra) for i in range(n_splits)]


def add_PCM_noise(weights: torch.Tensor, noise_config: NoiseConfig) -> torch.Tensor:
    """
    Adds PCM noise to the weight tensor.
    Args:
        noise_config.application (str): Controls whether per column or per tile.
    Returns:
        torch.Tensor: Output tensor of the same shape as the input tensor.
    """
    GMAX = 180
    OD_TD_DIVERGENCE = 52.5

    WEIGHT_NOISE_MODEL_OD = torch.tensor(
        [
            2.939751381126213e-05,
            -0.00371581208142707,
            0.22419685954327148,
            2.5797410660159494,
        ],
        dtype=torch.float64,
    )
    WEIGHT_NOISE_MODEL_TD = torch.tensor(
        [
            1.2329755395573392e-05,
            -0.003054993405157235,
            0.2453487119885723,
            2.110295720240385,
        ],
        dtype=torch.float64,
    )

    epsilon = torch.finfo(torch.float16).tiny
    if noise_config.max_inp_size <= 0:
        scale = 1 / (weights.abs().amax(1).view(-1, 1) + epsilon)
    else:
        _, cols = weights.shape
        split_sizes = get_split_sizes(cols, noise_config.max_inp_size)
        weight_tiles = torch.split(weights, split_sizes, dim=1)

        tile_scales = []
        for tile in weight_tiles:
            tile_max = tile.abs().amax(dim=1, keepdim=True)
            tile_scale = 1 / (tile_max + epsilon)
            tile_scale_expanded = tile_scale.expand(-1, tile.size(1))
            tile_scales.append(tile_scale_expanded)

        scale = torch.cat(tile_scales, dim=1)

    weights = weights * scale * GMAX

    noisy_weights = weights + torch.randn_like(weights) * polyval(
        WEIGHT_NOISE_MODEL_OD, weights.abs()
    )

    noisy_weights[torch.abs(weights) < 0.6] = 0

    replace = weights + torch.randn_like(weights) * polyval(
        WEIGHT_NOISE_MODEL_TD, weights.abs()
    )
    noisy_weights[weights.abs() > OD_TD_DIVERGENCE] = replace[
        weights.abs() > OD_TD_DIVERGENCE
    ]
    noisy_weights = noisy_weights / GMAX / scale
    return noisy_weights


def add_additive_noise(
    weights: torch.Tensor, noise_config: NoiseConfig
) -> torch.Tensor:
    """
    Adds additive Gaussian noise per channel to the weight tensor.
    Args:
        noise_config.additive_std_multiplier (float): Scaling factor for noise based on the maximum absolute value per column or per tile.
        noise_config.application (str): Controls whether per column or per tile.
    Returns:
        torch.Tensor: Output tensor of the same shape as the input tensor.
    """

    if noise_config.max_inp_size <= 0:
        amax = weights.abs().amax(dim=1, keepdim=True)
    else:
        _, cols = weights.shape
        split_sizes = get_split_sizes(cols, noise_config.max_inp_size)

        weight_tiles = torch.split(weights, split_sizes, dim=1)

        tile_amaxes = []
        for tile in weight_tiles:
            tile_amax = tile.abs().amax(dim=1, keepdim=True)  # [rows, 1]
            tile_amax_expanded = tile_amax.expand(
                -1, tile.size(1)
            )  # [rows, tile_width]
            tile_amaxes.append(tile_amax_expanded)

        amax = torch.cat(tile_amaxes, dim=1)

    with torch.no_grad():
        noise = noise_config.additive_std_multiplier * amax * torch.randn_like(weights)
    weights += noise

    return weights


def apply_noise(model: torch.nn.Module, noise_config: NoiseConfig):
    """Helper function for applying PCM noise model."""
    print(f"Applying {noise_config.noise_type} noise")
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        if noise_config.noise_type == "pcm":
            module.weight.data = add_PCM_noise(
                weights=module.weight.data, noise_config=noise_config
            )
        elif noise_config.noise_type == "additive":
            module.weight.data = add_additive_noise(
                weights=module.weight.data, noise_config=noise_config
            )
        else:
            raise ValueError("Invalid noise type")
        print(f"Added noise to {name}")


def main():
    # load random model
    model_path = "meta-llama/Llama-3.2-1B-Instruct"
    model = AutoModelForCausalLM.from_pretrained(model_path)

    # choose the noise config
    noise_config = NoiseConfig(noise_type="pcm", max_inp_size=512)

    # additive gaussian noise
    # noise_config = NoiseConfig(
    #     noise_type="additive",
    #     max_inp_size=512,
    #     additive_std_multiplier=0.05  # 5% w.r.t. abs-max
    # )
    
    # apply the noise in-place
    apply_noise(model, noise_config)


if __name__ == "__main__":
    main()