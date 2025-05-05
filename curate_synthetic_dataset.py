import os
import argparse
import torch
from vllm import LLM, SamplingParams
from tqdm import tqdm
import multiprocessing as mp
from datasets import Dataset, Features, Sequence, Value, concatenate_datasets
from transformers import AutoTokenizer


def write_dataset_chunk(
    model_name: str,
    model_base_path: str,
    save_base_path: str,
    node_id: int,
    gpu_id: int,
    number_of_tokens: int,
):
    path_to_model_and_tokenizer = os.path.join(
        os.path.expanduser(model_base_path),
        model_name
    )
    max_seq_len_to_capture = 4096
    start_string = ""

    if model_name == "microsoft/Phi-3-mini-4k-instruct":
        bos_id = 1
        top_k = -1
    elif model_name == "meta-llama/Llama-3.2-1B-Instruct":
        bos_id = 128000
        top_k = 50
    else:
        assert model_name in [
            "microsoft/Phi-3-mini-4k-instruct", "meta-llama/Llama-3.2-1B-Instruct"
        ], "Model not supported."

    tokenizer = AutoTokenizer.from_pretrained(path_to_model_and_tokenizer)

    seed = int(node_id * 8 + gpu_id)

    torch.manual_seed(seed)

    features = Features(
        {"input_ids": Sequence(Value("int32")), "attention_mask": Sequence(Value("int32"))}
    )
    dataset = Dataset.from_dict(
        {"input_ids": [], "attention_mask": []}, features=features, split="train"
    )

    print("Starting on rank", gpu_id)

    current_number_of_tokens = 0
    number_of_sequences_to_generate = 5

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    model = LLM(
        model=path_to_model_and_tokenizer,
        tokenizer=path_to_model_and_tokenizer,
        dtype=torch.float16,
        seed=seed,
        max_seq_len_to_capture=max_seq_len_to_capture,
        gpu_memory_utilization=0.4,
        #for llama 
        enable_chunked_prefill=False, 
        max_model_len=4096,
    )

    print("Loaded on GPU", gpu_id)

    num_starting_tokens = len(tokenizer(start_string)["input_ids"])
    num_tokens = max_seq_len_to_capture - num_starting_tokens

    sampling_params = SamplingParams(
        n=1,
        temperature=1.0,
        top_p=1.0,
        top_k=top_k,
        include_stop_str_in_output=True,
        ignore_eos=True,
        max_tokens=num_tokens,
        min_tokens=num_tokens,
        detokenize=True,
        skip_special_tokens=False,
        spaces_between_special_tokens=False,
    )

    pbar = tqdm(total=number_of_tokens)
    pbar.set_description(f"Rank {gpu_id}")

    # rolling buffers that are filled and stored when they reach 4k in length
    attention_mask_buffer = []
    input_ids_buffer = []

    while current_number_of_tokens < number_of_tokens:
        outputs = model.generate(
            [start_string] * number_of_sequences_to_generate, sampling_params, use_tqdm=False
        )
        outputs = [outputs[i].outputs[0] for i in range(number_of_sequences_to_generate)]

        for output in outputs:
            input_ids = list((bos_id,) + output.token_ids)
            attention_mask = [1] * len(input_ids)
            need_more = max_seq_len_to_capture - len(attention_mask_buffer)
            attention_mask_buffer.extend(attention_mask[:need_more])
            input_ids_buffer.extend(input_ids[:need_more])

            if len(input_ids_buffer) == max_seq_len_to_capture:

                num_new_tokens = len(input_ids_buffer)
                current_number_of_tokens += num_new_tokens
                pbar.update(num_new_tokens)

                # the buffers have the desired length, so now we store
                assert len(attention_mask_buffer) == max_seq_len_to_capture, "Attention mask is not same length as input ids"
                dataset = concatenate_datasets(
                    [
                        dataset,
                        Dataset.from_dict(
                            {"input_ids": [input_ids_buffer], "attention_mask": [attention_mask_buffer]},
                            features=features,
                            split="train",
                        ),
                    ]
                )
                # reset
                attention_mask_buffer = []
                input_ids_buffer = []

    # done, save to disk
    save_base_path = os.path.expanduser(save_base_path)
    if not os.path.isdir(save_base_path):
        os.makedirs(save_base_path, exist_ok=True)

    save_path = os.path.join(save_base_path, f"synthetic-dataset-{model_name}/worker-{seed}")
    dataset.save_to_disk(save_path)
    print(f"Rank {gpu_id} saved to disk...")


def setup_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-name",
        type=str,
        required=True
    )
    parser.add_argument(
        "--model-base-path",
        type=str,
        required=True
    )
    parser.add_argument(
        "--save-base-path",
        type=str,
        required=True
    )
    parser.add_argument(
        "--num-gpus-per-node",
        type=int,
        required=True
    )
    parser.add_argument(
        "--num-nodes",
        type=int,
        required=True
    )
    parser.add_argument(
        "--node-id",
        type=int,
        required=True
    )
    parser.add_argument(
        "--number-of-tokens",
        type=int,
        required=True
    )
    return parser


if __name__ == "__main__":
    # Setup the argument parser
    parser = setup_arg_parser()

    # Parse the arguments
    args = parser.parse_args()

    # Accessing the arguments
    num_gpus_per_node = args.num_gpus_per_node
    num_nodes = args.num_nodes
    number_of_tokens = args.number_of_tokens
    per_device_num_tokens = number_of_tokens // (num_gpus_per_node * num_nodes)
    available_gpus = [*range(num_gpus_per_node)]

    processes = []
    for gpu_id in available_gpus:
        p = mp.Process(
            target=write_dataset_chunk,
            args=(
                args.model_name,
                args.model_base_path,
                args.save_base_path,
                args.node_id,
                gpu_id,
                per_device_num_tokens,
            ),
        )
        processes.append(p)
        p.start()

    for p in processes:
        p: mp.Process
        p.join()  # Wait for all processes to finish
