from argparse import Namespace

from aihwkit_lightning.simulator.configs.configs import TorchInferenceRPUConfig
from aihwkit_lightning.simulator.parameters.enums import WeightModifierType, WeightClipType


def create_rpu_config(args: Namespace):
    """
    Create RPU config based on namespace.

    Args:
        args (Namespace): The namespace populated with fields.

    Returns:
        Union[InferenceRPUConfig,TorchInferenceRPUConfig]: The RPUConifg.
    """

    rpu_config = TorchInferenceRPUConfig()

    rpu_config.forward.inp_res = args.forward_inp_res
    rpu_config.forward.out_noise = args.forward_out_noise
    rpu_config.forward.out_noise_per_channel = args.forward_out_noise_per_channel

    # backward compat.
    rpu_config.forward.out_bound = args.forward_out_bound
    rpu_config.forward.out_res = args.forward_out_res

    rpu_config.clip.sigma = args.clip_sigma
    if args.clip_type == "gaussian":
        clip_type = WeightClipType.LAYER_GAUSSIAN
    elif args.clip_type == "gaussian_channel":
        clip_type = WeightClipType.LAYER_GAUSSIAN_PER_CHANNEL
    elif args.clip_type == "none":
        clip_type = WeightClipType.NONE
    else:
        raise Exception("Clip type not supported")
    rpu_config.clip.type = clip_type

    rpu_config.modifier.std_dev = args.modifier_std_dev
    rpu_config.modifier.res = args.modifier_res
    rpu_config.modifier.enable_during_test = args.modifier_enable_during_test
    rpu_config.modifier.offset = args.modifier_offset
    if args.modifier_type == "add_gauss":
        modifier_type = WeightModifierType.ADD_NORMAL
    elif args.modifier_type == "add_gauss_channel":
        modifier_type = WeightModifierType.ADD_NORMAL_PER_CHANNEL
    elif args.modifier_type == "discretize_per_channel":
        modifier_type = WeightModifierType.DISCRETIZE_PER_CHANNEL
    elif args.modifier_type == "multiplicative":
        modifier_type = WeightModifierType.MULTIPLICATIVE
    elif args.modifier_type == "multiplicative_offset":
        modifier_type = WeightModifierType.MULTIPLICATIVE_OFFSET
    elif args.modifier_type == "multiplicative_offset_channel":
        modifier_type = WeightModifierType.MULTIPLICATIVE_OFFSET_PER_CHANNEL
    elif args.modifier_type == "none":
        modifier_type = WeightModifierType.NONE
    else:
        raise Exception("Unknown modifier type")
    rpu_config.modifier.type = modifier_type

    rpu_config.mapping.max_input_size = args.mapping_max_input_size

    rpu_config.pre_post.input_range.enable = args.input_range_enable
    rpu_config.pre_post.input_range.learn_input_range = args.input_range_learn_input_range
    rpu_config.pre_post.input_range.init_value = args.input_range_init_value
    rpu_config.pre_post.input_range.fast_mode = args.input_range_fast_mode
    rpu_config.pre_post.input_range.init_with_max = args.input_range_init_with_max
    rpu_config.pre_post.input_range.init_from_data = args.input_range_init_from_data
    rpu_config.pre_post.input_range.init_std_alpha = args.input_range_init_std_alpha
    rpu_config.pre_post.input_range.decay = args.input_range_decay
    rpu_config.pre_post.input_range.input_min_percentage = args.input_range_input_min_percentage

    return rpu_config
