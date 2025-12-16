"""
ddp.py

Core class definition for a strategy implementing Torch native Distributed Data Parallel Training; note that on most
GPU hardware and LLM backbones >= 5-7B parameters, DDP training will OOM, which is why we opt for FSDP.
"""

import shutil
from pathlib import Path
from typing import Optional
from functools import partial

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    apply_activation_checkpointing,
    checkpoint_wrapper,
)
from transformers.optimization import get_constant_schedule, get_cosine_schedule_with_warmup

from prismatic.overwatch import initialize_overwatch
from training.strategies.base_strategy_cogact import TrainingStrategy

# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)


class DDPStrategy(TrainingStrategy):
    @overwatch.rank_zero_only
    def save_checkpoint(
        self,
        run_dir: Path,
        global_step: int,
        epoch: int,
        train_loss: Optional[float] = None,
        only_trainable: bool = True,
    ) -> None:
        """Save a checkpoint to the `run_dir` only containing the state_dicts for trainable parameters by default."""
        assert isinstance(self.vlm, DDP), "save_checkpoint assumes VLM is already wrapped in DDP!"

        # Splinter State Dictionary by Top-Level Submodules (or subset, if `only_trainable`)
        model_state_dicts = {
            mkey: getattr(self.vlm.module, mkey).state_dict()
            for mkey in (self.trainable_module_keys if only_trainable else self.all_module_keys)
        }
        optimizer_state_dict = self.optimizer.state_dict()

        # Set Checkpoint Path =>> Embed *minimal* training statistics!
        checkpoint_dir = run_dir / "checkpoints"
        if train_loss is None:
            checkpoint_path = checkpoint_dir / f"step-{global_step:06d}-epoch-{epoch:02d}-loss=inf.pt"
        else:
            checkpoint_path = checkpoint_dir / f"step-{global_step:06d}-epoch-{epoch:02d}-loss={train_loss:.4f}.pt"

        # Save Checkpoint & Copy Latest to `latest-checkpoint.pt`
        torch.save({"model": model_state_dicts, "optimizer": optimizer_state_dict}, checkpoint_path)
        shutil.copy(checkpoint_path, checkpoint_dir / "latest-checkpoint.pt")

    def run_setup(self, run_dir: Path, n_train_examples: int) -> None:
        # Gradient Checkpointing Setup
        if self.enable_gradient_checkpointing:
            # For Gradient Checkpointing under FSDP --> we make the same assumption as in the DDP/other strategies; the
            #   bulk of activation memory is taken up by the LLM activations. However, unlike other strategies, we
            #   cannot rely on the HF Transformers default `gradient_checkpointing_enable()` --> FSDP breaks semantics!
            #
            # Instead, we need to write our own *NO-REENTRANT* wrapper, and apply it to the LLM's Transformer Layer.
            non_reentrant_wrapper = partial(checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT)

            def check_fn(submodule: nn.Module) -> bool:
                return isinstance(submodule, self.llm_transformer_layer_cls)

            # Note that the terms "activation checkpointing" and "gradient checkpointing" are synonymous!
            apply_activation_checkpointing(self.vlm, checkpoint_wrapper_fn=non_reentrant_wrapper, check_fn=check_fn)


        # Move to Device =>> Note parameters are in full precision (*mixed precision* will only autocast as appropriate)
        overwatch.info("Placing Entire VLM (Vision Backbone, LLM Backbone, Projector Weights) on GPU", ctx_level=1)
        self.vlm.to(self.device_id)

        # Wrap with Distributed Data Parallel
        #   => Note: By default, wrapping naively with DDP(self.vlm) will initialize a *separate* buffer on GPU that
        #            is the same size/dtype as the model parameters; this will *double* GPU memory!
        # - stackoverflow.com/questions/68949954/model-takes-twice-the-memory-footprint-with-distributed-data-parallel
        overwatch.info("Wrapping VLM with Distributed Data Parallel", ctx_level=1)
        self.vlm = DDP(self.vlm, device_ids=[self.device_id], gradient_as_bucket_view=True)

        # Create Optimizer and LR Scheduler =>> note that most of the LR Schedulers we use require `max_steps/epochs`
        #   => Optimizer should only operate on parameters that are *unfrozen* / trainable!
        trainable_params = [param for param in self.vlm.parameters() if param.requires_grad]
        if self.max_steps is None:
            num_training_steps = (n_train_examples * self.epochs) // self.global_batch_size
        else:
            num_training_steps = self.max_steps

        if self.lr_scheduler_type == "linear-warmup+cosine-decay":
            # Set warmup steps (floor) based on `warmup_ratio` (should be 0.03 - 0.05)
            num_warmup_steps = int(num_training_steps * self.warmup_ratio)

            assert self.weight_decay == 0, "DDP training does not currently support `weight_decay` > 0!"
            self.optimizer = AdamW(trainable_params, lr=self.learning_rate, weight_decay=self.weight_decay)
            self.lr_scheduler = get_cosine_schedule_with_warmup(self.optimizer, num_warmup_steps, num_training_steps)
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = 0.0

        elif self.lr_scheduler_type == "constant":
            num_warmup_steps = 0

            assert self.weight_decay == 0, "DDP training does not currently support `weight_decay` > 0!"
            self.optimizer = AdamW(trainable_params, lr=self.learning_rate, weight_decay=self.weight_decay)
            self.lr_scheduler = get_constant_schedule(self.optimizer)

        else:
            raise ValueError(f"Learning Rate Schedule with type `{self.lr_scheduler_type}` is not supported!")

        # Finalize Setup =>> Log
        overwatch.info(
            "DDP Strategy =>> Finalized Training Setup:\n"
            f"         |-> Global (Effective) Batch Size = {self.global_batch_size}\n"
            f"         |-> Per-Device Batch Size = {self.per_device_batch_size}\n"
            f"         |-> Distributed World Size = {overwatch.world_size()}\n"
            f"         |-> Gradient Accumulation Steps = {self.grad_accumulation_steps}\n\n"
            f"         |-> LLM Backbone Gradient Checkpointing = {self.enable_gradient_checkpointing}\n"
            f"         |-> Use Native AMP = {self.enable_mixed_precision_training} ({self.mixed_precision_dtype})\n\n"
            f"         |-> Default AdamW LR = {self.learning_rate}\n"
            f"         |-> AdamW Weight Decay = {self.weight_decay}\n"
            f"         |-> LR Scheduler Type = {self.lr_scheduler_type}\n"
            f"         |-> LR Scheduler Warmup Steps (Ratio) = {num_warmup_steps} ({self.warmup_ratio})\n"
            f"         |-> Dataset Size = {n_train_examples} Examples\n"
            f"         |-> Max Steps = {num_training_steps}\n"
        )

    def clip_grad_norm(self) -> None:
        torch.nn.utils.clip_grad_norm_(self.vlm.parameters(), max_norm=self.max_grad_norm)

    def _get_optimizer_path(self, checkpoint_path: Path) -> Path:
        """Get the path to the optimizer checkpoint file."""
        return checkpoint_path.with_suffix(".optimizer")

    def load_optimizer_and_scheduler(self, checkpoint_path: str) -> None:
        """Load a checkpoint from the specified `checkpoint_path`."""
        assert isinstance(self.vlm, FSDP), "FSDPStrategy.load_optimizer_and_scheduler assumes VLM is already wrapped in FSDP!"
        checkpoint_path = Path(checkpoint_path)
        optimizer_path = self._get_optimizer_path(checkpoint_path)
        if not optimizer_path.exists():
            overwatch.warning(f"Optimizer checkpoint not found at {optimizer_path}!")
            return
        # Load Checkpoint =>> Note that FSDP will automatically handle device placement!
        optim_state_dict = torch.load(optimizer_path, map_location="cpu")
        self.optimizer.load_state_dict(optim_state_dict)
        overwatch.info(f"Loaded optimizer state dict from {optimizer_path}")
