import os
import glob
import shutil

from datetime import timedelta
import transformers
import accelerate.logging

import torch

import datasets
from datasets import load_from_disk
from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.utils import set_seed
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from transformers.utils.logging import enable_default_handler, enable_explicit_format

from utils.task_parser import get_args
from utils.train_utils import (
    CustomTrainer,
)

assert torch.cuda.is_available(), "GPU must be available."


def main():
    # set up logger
    log_level = datasets.logging.INFO
    datasets.utils.logging.set_verbosity(log_level)
    transformers.logging.set_verbosity_info()
    accelerate.logging.get_logger(__name__, log_level="INFO")
    enable_default_handler()
    enable_explicit_format()

    # get args and set seed
    args = get_args()
    set_seed(args.seed)

    # Tell the Accelerator object to log with wandb
    kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=1801))
    accelerator = Accelerator(log_with="wandb", kwargs_handlers=[kwargs])

    if accelerator.is_main_process:
        accelerator.init_trackers(
            project_name="ANFM", init_kwargs={"wandb": {"name": args.output_dir.split("/")[-1]}}
        )

    model = AutoModelForCausalLM.from_pretrained(os.path.join("./data", args.base_model))
    tokenizer = AutoTokenizer.from_pretrained(
        os.path.join("./data", args.base_model),
        padding_side="left"
    )
    tokenizer.pad_token = tokenizer.eos_token

    dataset = load_from_disk(args.pre_train_dataset)

    # define TrainingArguments
    metric_name = "loss"
    lr = accelerator.num_processes * args.lr

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=args.overwrite_output_dir,
        do_train=args.do_train,
        do_eval=args.do_eval,
        eval_on_start=args.eval_on_start,
        eval_strategy=args.eval_strategy,
        save_strategy=args.save_strategy,
        save_safetensors=False,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": True},
        learning_rate=lr,
        lr_scheduler_type=args.lr_scheduler_type,
        lr_scheduler_kwargs={"lr_end": args.lr_end} if args.lr_end != None else None,
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        adam_epsilon=args.adam_epsilon,
        report_to=args.report_to,
        weight_decay=args.weight_decay,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        max_grad_norm=args.max_grad_norm,
        eval_steps=args.eval_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        seed=args.seed,
        metric_for_best_model=metric_name,
        load_best_model_at_end=args.load_best_model_at_end,
        warmup_ratio=args.warmup_ratio,
        bf16=False,
        bf16_full_eval=False,
        greater_is_better=args.greater_is_better,
        fp16=True,
        fp16_full_eval=True,
        torch_compile=False,
        deepspeed=args.ds_config_path,
    )

    # define trainer
    validation_key = "test"
    lm_data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False
    )

    device = "cuda" if not hasattr(accelerator, "device") else accelerator.device

    teacher_model = None
    if args.distillation:
        teacher_model = AutoModelForCausalLM.from_pretrained(
            os.path.join("./data", args.base_model)
        ).to(device=device, dtype=torch.float16)

    trainer = CustomTrainer(
        model=model,
        teacher_model=teacher_model,
        eval_medqa=args.eval_medqa,
        temperature=args.distillation_temperature,
        beta=args.distillation_beta,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset[validation_key],
        tokenizer=tokenizer,
        data_collator=lm_data_collator,
    )

    # train the model and save it
    trainer.train(resume_from_checkpoint=False)

    if accelerator.is_main_process:
        training_done = False
        if args.max_steps > 0 and trainer.state.global_step == args.max_steps:
            print("[TRAINING DONE] Saving model...")
            training_done = True
        elif args.max_steps <= 0 and trainer.state.epoch == args.num_train_epochs:
            print("[TRAINING DONE] Saving model...")
            training_done = True
        else:
            print("[TRAINING NOT DONE]")

        if training_done:
            trainer.save_model(args.output_dir)

            # remove checkpoint folders
            print("[TRAINING DONE]: Training has ended. Removing checkpoints")
            pattern = os.path.join(args.output_dir, "checkpoint-*")
            folders = glob.glob(pattern)
            for folder in folders:
                if os.path.isdir(folder):
                    print(f"[TRAINING DONE]: Training has ended. Removing {folder}")
                    shutil.rmtree(folder)


if __name__ == "__main__":
    main()
