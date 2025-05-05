import dill
import argparse
import torch
from typing import Any
from aihwkit_lightning.nn.conversion import convert_to_analog
from transformers import (
    Phi3ForCausalLM,
    Phi3Config,
    LlamaForCausalLM,
    LlamaConfig,
    AutoTokenizer,
    TextStreamer
)
from huggingface_hub import hf_hub_download

device = "cuda" if torch.cuda.is_available() else "cpu"


def load_analog_model(
    name: str, fp_model_cls: Any, config_cls: Any, conversion_map: dict = None
):
    """
    Load an analog model from the hub.

    Args:
        name (str): Name for loading from the hub. Must be in the format "organization/model_name".
        fp_model_cls (Any): Class of the model in FP to load.
        config_cls (Any): Configuration class used for model.
        conversion_map (dict, optional): Dict mapping torch layers to analog layers. Defaults to None.

    Returns:
        Model: Loaded analog model.
    """
    if name == "ibm-aimc/analog-Phi-3-mini-4k-instruct":
        path1 = hf_hub_download(repo_id=name, filename="pytorch_model-00001-of-00002.bin")
        path2 = hf_hub_download(repo_id=name, filename="pytorch_model-00002-of-00002.bin")

        model_sd1 = torch.load(path1, weights_only=False)
        model_sd2 = torch.load(path2, weights_only=False)
        model_sd = {**model_sd1, **model_sd2}
    elif name == "ibm-aimc/analog-Llama-3.2-1B-Instruct":
        path = hf_hub_download(repo_id=name, filename="pytorch_model.bin")
        model_sd = torch.load(path, weights_only=False)
    else:
        raise Exception("Unknown model")

    rpu_path = hf_hub_download(repo_id=name, filename="rpu_config")
    with open(rpu_path, "rb") as f:
        rpu_config = dill.load(f)

    model = fp_model_cls(config=config_cls.from_pretrained(name))
    model = convert_to_analog(
        model,
        rpu_config=rpu_config,
        conversion_map=conversion_map,
    )
    model.load_state_dict(model_sd)
    model = model.eval()
    model = model.to(dtype=torch.float16, device=device)
    return model


def query_llm_streaming(prompt: str, model: Phi3ForCausalLM, tokenizer: AutoTokenizer):
    """Queries the model with the given question using token streaming."""

    inputs = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True,
        return_dict=True,
        add_generation_prompt=True,
        return_attention_mask=True,
        return_tensors="pt",
    )

    # Create a streamer to print tokens as they are generated
    streamer = TextStreamer(tokenizer)

    # Generate a response with streaming
    model.generate(
        input_ids=inputs.input_ids.to(device=device),
        attention_mask=inputs.attention_mask.to(device=device),
        max_new_tokens=512,
        streamer=streamer,  # Stream tokens
        tokenizer=tokenizer,
        stop_strings="<|end|>"
    )


def setup_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-name",
        type=str,
        default="microsoft/Phi-3-mini-4k-instruct"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="How to explain Internet to a medieval knight? Keep it short."
    )
    return parser


def main():
    parser = setup_arg_parser()
    args = parser.parse_args()

    if args.model_name == "meta-llama/Llama-3.2-1B-Instruct":
        analog_model_id = "ibm-aimc/analog-Llama-3.2-1B-Instruct"
        config_cls = LlamaConfig
        fp_model_cls = LlamaForCausalLM
    elif args.model_name == "microsoft/Phi-3-mini-4k-instruct":
        analog_model_id = "ibm-aimc/analog-Phi-3-mini-4k-instruct"
        config_cls = Phi3Config
        fp_model_cls = Phi3ForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(analog_model_id)
    model = load_analog_model(analog_model_id, fp_model_cls=fp_model_cls, config_cls=config_cls)
    query_llm_streaming(args.prompt, model, tokenizer)


if __name__ == "__main__":
    main()