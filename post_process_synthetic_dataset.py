import os
import argparse
from datasets import (
    load_from_disk,
    concatenate_datasets
)


def main(
    workers_path: str,
    save_path: str
):
    complete_dataset = None
    i = 0
    for worker_folder in sorted(os.listdir(workers_path)):
        ds_path = os.path.join(workers_path, worker_folder)
        ds = load_from_disk(ds_path)
        if complete_dataset is None:
            complete_dataset = ds
        else:
            complete_dataset = concatenate_datasets([complete_dataset, ds])

        print(i)
        i += 1
    complete_dataset = complete_dataset.train_test_split(test_size=0.01, seed=0)
    complete_dataset.save_to_disk(
        os.path.expanduser(os.path.expanduser(save_path))
    )


def setup_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--workers-path",
        type=str,
        required=True
    )
    parser.add_argument(
        "--save-path",
        type=str,
        required=True
    )
    return parser


if __name__ == "__main__":
    # Setup the argument parser
    parser = setup_arg_parser()

    # Parse the arguments
    args = parser.parse_args()

    main(
        workers_path=args.workers_path,
        save_path=args.save_path
    )