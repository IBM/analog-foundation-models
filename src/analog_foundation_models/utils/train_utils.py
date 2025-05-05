from typing import Dict, Any, Union, Optional, List
import math
import os
import torch
import time

import transformers
from datasets import Dataset
from transformers import (
    Trainer,
    Phi3ForCausalLM,
    LlamaForCausalLM,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import (
    is_safetensors_available,
    SAFE_WEIGHTS_NAME,
    WEIGHTS_NAME,
    logging,
)

IS_CUDA = torch.cuda.is_available()
TRANSFORMERS_VERSION = transformers.__version__

if is_safetensors_available():
    import safetensors.torch

logger = logging.get_logger(__name__)


def distillation_loss(student_logits, teacher_logits, temperature):
    # cast the logits. we cannot rely on the model in
    # transformers to cast to fp32 for us
    student_logits = student_logits.float()
    teacher_logits = teacher_logits.float()

    bsz, seq_len, _ = student_logits.shape
    soft_targets = torch.nn.functional.softmax(teacher_logits / temperature, dim=-1)
    soft_prob = torch.nn.functional.log_softmax(student_logits / temperature, dim=-1)
    soft_targets_loss = (
        torch.sum(soft_targets * (soft_targets.log() - soft_prob))
        / (bsz * seq_len)
        * (temperature**2)
    )

    has_nan_inf = not (
        torch.isfinite(student_logits).all()
        and torch.isfinite(teacher_logits).all()
        and torch.isfinite(soft_prob).all()
        and torch.isfinite(soft_targets).all()
    )

    return soft_targets_loss, has_nan_inf


def speed_metrics(split, start_time, num_samples=None, num_steps=None, num_tokens=None):
    """
    Measure and return speed performance metrics.

    This function requires a time snapshot `start_time` before the operation to be measured starts and this function
    should be run immediately after the operation to be measured has completed.

    Args:
    - split: name to prefix metric (like train, eval, test...)
    - start_time: operation start time
    - num_samples: number of samples processed
    - num_steps: number of steps processed
    - num_tokens: number of tokens processed
    """
    runtime = time.time() - start_time
    result = {f"{split}_runtime": round(runtime, 4)}
    if runtime == 0:
        return result
    if num_samples is not None:
        samples_per_second = num_samples / runtime
        result[f"{split}_samples_per_second"] = round(samples_per_second, 3)
    if num_steps is not None:
        steps_per_second = num_steps / runtime
        result[f"{split}_steps_per_second"] = round(steps_per_second, 3)
    if num_tokens is not None:
        tokens_per_second = num_tokens / runtime
        result[f"{split}_tokens_per_second"] = round(tokens_per_second, 3)
    return result


class CustomTrainer(Trainer):
    def __init__(
        self,
        *args,
        eval_medqa: bool = False,
        temperature: float = 1.0,
        beta: float = 0.5,
        teacher_model: Union[Phi3ForCausalLM, LlamaForCausalLM, None] = None,
        **kwargs,
    ):
        self.eval_medqa = eval_medqa
        self.use_distillation = not (teacher_model is None)
        self.teacher_model = teacher_model
        self.temperature = temperature
        self.beta = beta
        self.medqa_step = 0
        super().__init__(*args, **kwargs)

    # overwriting for clipping the weights
    def training_step(
        self,
        model: torch.nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        num_items_in_batch=None,
    ) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to train.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.

        Return:
            `torch.Tensor`: The tensor with training loss on this batch.
        """
        model.train()
        if hasattr(self.optimizer, "train") and callable(self.optimizer.train):
            self.optimizer.train()

        inputs = self._prepare_inputs(inputs)

        # from deepspeed.runtime.utils import see_memory_usage
        # see_memory_usage("Before compute loss", force=True)
        encountered_nan_inf = False
        with self.compute_loss_context_manager():
            if self.use_distillation:
                lm_loss, outputs = self.compute_loss(
                    model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch
                )
                student_logits = outputs.logits
                with torch.no_grad():
                    teacher_outputs = self.teacher_model(**inputs)
                    teacher_logits = teacher_outputs.logits
                distill_loss, has_nan_inf = distillation_loss(
                    student_logits, teacher_logits, temperature=self.temperature
                )
                encountered_nan_inf = encountered_nan_inf or has_nan_inf
                if self.beta == 1.0:
                    loss = distill_loss
                elif self.beta == 0.0:
                    loss = lm_loss
                else:
                    loss = (1 - self.beta) * lm_loss + self.beta * distill_loss
            else:
                loss = self.compute_loss(model, inputs)

        if encountered_nan_inf:
            print(f"WARNING NaN/Inf caused by this input sequence: {self.tokenizer.decode(inputs.input_ids[0])}")

        del inputs
        torch.cuda.empty_cache()

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if hasattr(self.accelerator, "deepspeed_engine_wrapped") and self.accelerator.deepspeed_engine_wrapped is not None:
            made_step = (
                self.accelerator.deepspeed_engine_wrapped.engine.is_gradient_accumulation_boundary()
            )
        else:
            made_step = True

        # see_memory_usage("Before backward", force=True)
        loss *= self.args.gradient_accumulation_steps

        # DeepSpeed direct implementation (skipping accelerate)
        self.accelerator.deepspeed_engine_wrapped.engine.backward(loss)

        if encountered_nan_inf:
            print("WARNING: Zeroing out gradients due to NaN/Inf in logits")
            # sets grad to None. same as overflow procedure
            self.accelerator.deepspeed_engine_wrapped.engine.optimizer.zero_grad()

        self.accelerator.deepspeed_engine_wrapped.engine.step()

        # this is what we changed. we clip the weights when we updated the parameters
        with torch.no_grad():
            if made_step and hasattr(model, "analog_layers"):
                for analog_layer in model.analog_layers():
                    analog_layer.clip_weights()

        return loss.detach() / self.args.gradient_accumulation_steps

    def evaluate(
        self,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:

        # handle multipe eval datasets
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        if isinstance(eval_dataset, dict):
            metrics = {}
            for eval_dataset_name, _eval_dataset in eval_dataset.items():
                dataset_metrics = self.evaluate(
                    eval_dataset=_eval_dataset,
                    ignore_keys=ignore_keys,
                    metric_key_prefix=f"{metric_key_prefix}_{eval_dataset_name}",
                )
                metrics.update(dataset_metrics)
            return metrics

        # memory metrics - must set up as early as possible
        self._memory_tracker.start()
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        start_time = time.time()

        eval_loop = (
            self.prediction_loop if self.args.use_legacy_prediction_loop else self.evaluation_loop
        )
        output = eval_loop(
            eval_dataloader,
            description="Evaluation",
            # No point gathering the predictions if there are no metrics, otherwise we defer to
            # self.args.prediction_loss_only
            prediction_loss_only=True if self.compute_metrics is None else None,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )

        total_batch_size = self.args.eval_batch_size * self.args.world_size
        if f"{metric_key_prefix}_jit_compilation_time" in output.metrics:
            start_time += output.metrics[f"{metric_key_prefix}_jit_compilation_time"]
        output.metrics.update(
            speed_metrics(
                metric_key_prefix,
                start_time,
                num_samples=output.num_samples,
                num_steps=math.ceil(output.num_samples / total_batch_size),
            )
        )
        print(output.metrics)
        self.log(output.metrics)

        self.control = self.callback_handler.on_evaluate(
            self.args, self.state, self.control, output.metrics
        )
        self._memory_tracker.stop_and_update_metrics(output.metrics)
        return output.metrics

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")

        supported_classes = (PreTrainedModel, )
        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        if not isinstance(self.model, supported_classes):
            if state_dict is None:
                state_dict = self.model.state_dict()

            if isinstance(self.accelerator.unwrap_model(self.model), supported_classes):
                self.accelerator.unwrap_model(self.model).save_pretrained(
                    output_dir, state_dict=state_dict, safe_serialization=self.args.save_safetensors
                )
            else:
                logger.info("Trainer.model is not a `PreTrainedModel`, only saving its state dict.")
                if self.args.save_safetensors:
                    safetensors.torch.save_file(
                        state_dict,
                        os.path.join(output_dir, SAFE_WEIGHTS_NAME),
                        metadata={"format": "pt"},
                    )
                else:
                    torch.save(state_dict, os.path.join(output_dir, WEIGHTS_NAME))
        else:
            self.model.save_pretrained(
                output_dir, state_dict=state_dict, safe_serialization=self.args.save_safetensors
            )

        if self.processing_class is not None:
            self.processing_class.save_pretrained(output_dir)
