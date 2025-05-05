import yaml
import argparse
from utils.utils import PrettySafeLoader


def get_args():
    args = create_parser()
    args = parse_args(args)
    args = check_and_eval_args(args)
    return args


def create_parser():
    parser = argparse.ArgumentParser()
    g = parser.add_argument_group("Training Args")
    g.add_argument("--config", dest="config", type=argparse.FileType(mode="r"))
    return parser


def maybe_inf2float(val):
    if isinstance(val, str) and (val == "inf" or val == "-inf"):
        print(f"WARNING: String {val} casted to float")
        return float(val)
    return val


def parse_args(parser: argparse.ArgumentParser):
    args, unknown = parser.parse_known_args()
    if unknown != []:
        assert args.spinquant or args.gptq, "Found unknown arguments but spinquant is not enabled."
    if hasattr(args, "config") and args.config:
        data = yaml.load(args.config, Loader=PrettySafeLoader)
        args.config = args.config.name
        arg_dict = args.__dict__

        # This can be solved more elegantly. Maybe create subfolder for evaluation in config folder
        # and call task evaluation. Then, have own default.yaml in said subfolder
        default_path = "./data/training_files/default.yaml"
        print(f"Setting default configs path at {default_path}")

        with open(default_path, "r") as stream:
            default = yaml.safe_load(stream)
        for key, value in default.items():
            if isinstance(value, dict):
                for subkey, subvalue in value.items():
                    arg_dict[subkey] = maybe_inf2float(subvalue)
            else:
                arg_dict[key] = maybe_inf2float(value)
        for key, value in data.items():
            if isinstance(value, dict):
                for subkey, subvalue in value.items():
                    if subkey not in arg_dict.keys():
                        raise KeyError(f"Trying to override a non-existing parameter {subkey}")
                    arg_dict[subkey] = maybe_inf2float(subvalue)
            else:
                if key not in arg_dict.keys():
                    raise KeyError(f"Trying to override a non-existing parameter {key}")
                arg_dict[key] = maybe_inf2float(value)
    return args


def check_and_eval_args(args):
    for k, v in args.__dict__.items():
        if k == "lr" and not isinstance(v, list):
            args.lr = float(args.lr)
        if isinstance(v, str) and "lambda" in v:
            setattr(args, k, eval(v))
        elif isinstance(v, list):
            new_l = []
            for el in v:
                if isinstance(el, str) and "lambda" in el:
                    new_l.append(eval(el))
                elif k == "lr":
                    new_l.append(float(el))
                else:
                    new_l.append(el)
            setattr(args, k, new_l)
    return args
