"""DPO trainer for axolotl"""

import gc
from functools import wraps
from typing import Any, Dict, Union

import torch
import math
import os
from torch import nn
from trl import DPOTrainer
from typing_extensions import override


from axolotl.core.trainers.mixins import (
    DistributedParallelMixin,
    RngLoaderMixin,
    SchedulerMixin,
)
from axolotl.core.trainers.mixins.optimizer import OptimizerInitMixin, OptimizerMixin
from axolotl.core.trainers.utils import (
    sanitize_kwargs_for_ds_tagging,
    sanitize_kwargs_for_tagging,
)
from axolotl.utils import get_not_null
from axolotl.utils.bench import get_gpu_memory_usage
from axolotl.utils.dict import DictDefault
from axolotl.utils.distributed import is_distributed, is_main_process
from axolotl.utils.logging import get_logger
from axolotl.utils.samplers import MultipackBatchSampler, get_dataset_lengths

class AxolotlDPOTrainer(
    RngLoaderMixin,
    SchedulerMixin,
    OptimizerMixin,
    OptimizerInitMixin,
    DPOTrainer,
    DistributedParallelMixin,
):
    """Extend the base DPOTrainer for axolotl helpers."""

    tag_names = ["axolotl", "dpo"]

    def __init__(self, *args, dataset_tags=None, **kwargs):
        super().__init__(*args, **kwargs)

        self.dataset_tags = dataset_tags
        self.optimizer = None
        self.model_accepts_loss_kwargs = False


    @override
    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        # track number of tokens for tokens per second calculation
        if self.args.include_tkps and model.training:
            chosen_inputs_key = "labels" if "labels" in inputs else "chosen_input_ids"
            rejected_inputs_key = "labels" if "labels" in inputs else "rejected_input_ids"
            trainable_tokens = (inputs[chosen_inputs_key] != -100).sum() + (inputs[rejected_inputs_key] != -100).sum()
            total_tokens = inputs["prompt_input_ids"].numel() + trainable_tokens
            total_tokens = torch.tensor(total_tokens, device=inputs[chosen_inputs_key].device)

            if is_distributed():
                torch.distributed.all_reduce(
                    trainable_tokens, op=torch.distributed.ReduceOp.SUM
                )
                torch.distributed.all_reduce(
                    total_tokens, op=torch.distributed.ReduceOp.SUM
                )

            if not hasattr(self.state, "tokens"):
                self.state.tokens = {
                    "trainable": torch.zeros(1),
                    "total": torch.zeros(1),
                }

            # trainable tokens for throughput and total token slots for summaries
            self.state.tokens["trainable"] = (
                self.state.tokens["trainable"] + trainable_tokens.detach().cpu()
            )
            self.state.tokens["total"] = self.state.tokens["total"] + total_tokens.cpu()
            # Store per-step trainable tokens for throughput calculation
            self.state.tokens["trainable_tokens"] = trainable_tokens.detach().cpu()

        return super().compute_loss(
            model,
            inputs,
            return_outputs=return_outputs,
            num_items_in_batch=num_items_in_batch,
        )
    
    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
        """
        Log `logs` on the various objects watching training, including stored metrics.

        Args:
            logs: The values to log.
            start_time: The start of training.
        """
        # logs either has 'loss' or 'eval_loss'
        train_eval = "train" if "loss" in logs else "eval"
        metric_ndigits = int(os.getenv("AXOLOTL_METRIC_NDIGITS", "5"))

        if "loss" in logs:
            try:
                logs["ppl"] = round(math.exp(logs["loss"]), metric_ndigits)
            except OverflowError:
                logs["ppl"] = float("inf")
        if "eval_loss" in logs:
            try:
                logs["eval_ppl"] = round(math.exp(logs["eval_loss"]), metric_ndigits)
            except OverflowError:
                logs["eval_ppl"] = float("inf")

        if is_main_process():
            # Add memory usage
            try:
                active, allocated, reserved = get_gpu_memory_usage()
                logs["memory/max_active (GiB)"] = round(active, 2)
                logs["memory/max_allocated (GiB)"] = round(allocated, 2)
                logs["memory/device_reserved (GiB)"] = round(reserved, 2)
            except (ValueError, TypeError, FileNotFoundError):
                pass

        logs["tokens/train_per_sec_per_gpu"] = round(
            self.state.last_tokens_per_second.item() / self.args.logging_steps, 2
        )

        del self._stored_metrics[train_eval]

        return super().log(logs, start_time)

    @wraps(DPOTrainer.push_to_hub)
    def push_to_hub(self, *args, **kwargs) -> str:
        """
        Overwrite the `push_to_hub` method in order to force-add the tags when pushing
        the model on the Hub. Please refer to `~transformers.Trainer.push_to_hub`
        for more details.
        """
        kwargs = sanitize_kwargs_for_ds_tagging(
            dataset_tags=self.dataset_tags, kwargs=kwargs
        )
        kwargs = sanitize_kwargs_for_tagging(tag_names=self.tag_names, kwargs=kwargs)

        return super().push_to_hub(*args, **kwargs)

    @staticmethod
    def tokenize_row(
        features,
        processing_class,
        max_prompt_length,
        max_completion_length,
        add_special_tokens,
    ) -> Dict:
        res = DPOTrainer.tokenize_row(
            features,
            processing_class,
            max_prompt_length,
            max_completion_length,
            add_special_tokens,
        )
        # fix when the tokenizer doesn't have a bos_token_id, e.g. Qwen
        if processing_class.bos_token is None and res["prompt_input_ids"][0] is None:
            for key in res.keys():
                res[key] = res[key][1:]

        if processing_class.bos_token and processing_class.bos_token_id is not None:
            # dpo trainer may incorrectly prepend the bos_token_id to the dpo outputs
            if res["chosen_input_ids"][0] == processing_class.bos_token_id:
                res["chosen_input_ids"] = res["chosen_input_ids"][1:]
            if res["rejected_input_ids"][0] == processing_class.bos_token_id:
                res["rejected_input_ids"] = res["rejected_input_ids"][1:]

        return res

    def training_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        num_items_in_batch=None,
    ) -> torch.Tensor:
        loss: torch.Tensor = super().training_step(model, inputs, num_items_in_batch)
        gc.collect()
        torch.cuda.empty_cache()
        return loss

    def concatenated_forward(
        self,
        model: nn.Module,
        batch: dict[str, Union[list, torch.LongTensor]],
        is_ref_model: bool = False,
    ) -> dict[str, torch.Tensor]:
        if self.args.dpo_norm_loss:
            # fmt: off
            loss_type: str = self.loss_type  # type: ignore[has-type]
            # fmt: on
            # concatenated_forward handles avg token logprob for ipo case already
            self.loss_type = "ipo"
            res = super().concatenated_forward(model, batch, is_ref_model=is_ref_model)
            self.loss_type = loss_type
            return res
        return super().concatenated_forward(model, batch, is_ref_model=is_ref_model)
