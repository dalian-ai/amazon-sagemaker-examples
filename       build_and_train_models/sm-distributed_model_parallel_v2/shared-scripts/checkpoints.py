"""Export distributed checkpoints."""

import os
import pickle
import statistics
import time
import warnings
from enum import Enum, auto
from typing import Any, Dict, Optional

import numpy

# pylint: disable=import-error,no-name-in-module
import torch
import torch.distributed as dist
import torch.sagemaker.checkpoint.utils as tsm_checkpoint
from pathlib import Path
from data.utils import is_s3_source, parse_s3_address
from logging_utils import get_logger
from torch.distributed import checkpoint
from torch.distributed._shard.api import load_with_process_group
from torch.distributed.checkpoint.optimizer import load_sharded_optimizer_state_dict
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType
from torch.distributed.fsdp.api import FullStateDictConfig, ShardedOptimStateDictConfig
from torch.sagemaker.utils.process_group_utils import get_global_ranks

SUPPORT_ASYNC_SAVE = True
try:
    from megatron.core.dist_checkpointing.mapping import CheckpointingException
    from torch.sagemaker.distributed.checkpoint.state_dict_saver import (
        PyTorchDistSaveShardedStrategy,
        async_save,
    )
except ImportError:
    SUPPORT_ASYNC_SAVE = False
logger = get_logger()


# How to remove extra checkpoints, `regex` and `sort_fn` need to match for correctness.
#   - Sort subdir by the **last** int, right before `steps` as shown in the regex.
_CHECKPOINT_DIR_REGEX = r"^.*\d+steps$"
_CHECKPOINT_SORT_FN = tsm_checkpoint.SORT_BY_LAST_INT
_DEFAULT_STATE_DICT_TYPE = StateDictType.SHARDED_STATE_DICT

_EXPORT_KEYS = (
    "resume_from_sequence_number",
    "start_train_path_index",
    "total_steps",
)

_MAX_ATTEMPTS = 3


class CheckpointingMethod(Enum):
    SHARDED = auto()
    LOCAL = auto()
    FULL = auto()
    ASYNC_SHARDED = auto()
    ASYNC_LOCAL = auto()


def backward_compat_get_val_resume_from_sequence_number(state_dict):
    if "val_resume_from_sequence_number" not in state_dict:

        logger.warn(
            "Did not find validation dataloader's sequence number, validation dataloader will start from batch 0"
        )
        return 0
    return state_dict["val_resume_from_sequence_number"]


def backward_compat_get_resume_from_sequence_number(args, state_dict):
    if "resume_from_sequence_number" not in state_dict:
        return state_dict["start_batch_index"] * args.train_batch_size
    else:
        return state_dict["resume_from_sequence_number"]


def compute_stats_of_metric(metric: float, key: str, group: Optional[Any] = None):
    """Compute metric stats."""
    times = [None for _ in range(dist.get_world_size(group))]
    dist.all_gather_object(times, metric, group=group)

    if dist.get_rank() == 0:
        logger.info(
            "Time taken (min, max, mean, stddev, median, len) = "
            "(%7.2f, %7.2f, %7.2f, %7.2f, %7.2f, %02d): %s.",
            numpy.min(times),
            numpy.max(times),
            statistics.mean(times),
            statistics.stdev(times),
            statistics.median(times),
            len(times),
            key,
        )


def is_action_rank(global_rank):
    from torch.sagemaker import state

    return state.ranker.get_rep_rank(global_rank) == 0


def get_coordinator_rank(process_group):
    model_pg_ranks = get_global_ranks(process_group)
    return min(model_pg_ranks)


def get_current_replication_group(global_rank):
    from torch.sagemaker import state

    current_replication_group = None
    current_replication_ranks = None
    for replication_ranks in state.ranker.get_rep_groups():
        rep_group = dist.new_group(replication_ranks)
        if global_rank in replication_ranks:
            current_replication_group = rep_group
            current_replication_ranks = replication_ranks
    assert current_replication_group and current_replication_ranks, (f"Could not find replication group give {global_rank}")
    return current_replication_ranks, current_replication_group


def _retry_write_to_disk(func, max_attempts=_MAX_ATTEMPTS):
    for retry in range(max_attempts):
        try:
            func()
            return
        except (RuntimeError, pickle.UnpicklingError) as error:
            if isinstance(error, pickle.UnpicklingError) or (
                "unexpected pos" in str(error)
            ):
                logger.error(error)
                logger.error(
                    "Retry [%d/%d] failed to write to disk, in case it was due to transient error.",
                    retry,
                    max_attempts,
                )
                if retry < max_attempts - 1:
                    continue

            raise error


def _save_sharded(  # pylint: disable=too-many-arguments
    model,
    optimizer,
    scheduler,
    user_content,
    save_dir: str,
    checkpointing_pg_metadata,
):
    """Save FSDP checkpoint: Without process groups."""
    with FSDP.state_dict_type(model, _DEFAULT_STATE_DICT_TYPE):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            # pylint: disable=line-too-long
            # torch/distributed/fsdp/_common_utils.py:291: UserWarning:
            # An unexpected prefix is detected. This case should only happen when using DMP with FSDP.
            # prefix = _checkpoint_wrapped_module.gpt_neox.layers.34., submodule_name = _fsdp_wrapped_module
            # pylint: enable=line-too-long

            optim_state_dict = FSDP.optim_state_dict(model, optimizer)

        state_dict = {
            "model": model.state_dict(),
            "optimizer": optim_state_dict,
            "scheduler": scheduler.state_dict(),
        }
        # merge user content to state_dict
        state_dict = state_dict | user_content

    if dist.get_rank() == 0:
        logger.info("Processed state dict to save. Starting write to disk now.")

    process_group, coordinator_rank, action_rank = checkpointing_pg_metadata
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        # torch/distributed/checkpoint/filesystem.py:157: UserWarning: TypedStorage is deprecated.

        if action_rank:
            checkpoint.save_state_dict(
                state_dict=state_dict,
                storage_writer=checkpoint.FileSystemWriter(save_dir),
                planner=checkpoint.DefaultSavePlanner(),
                process_group=process_group,
                coordinator_rank=coordinator_rank,
            )


def _save_full(  # pylint: disable=too-many-arguments
    model,
    save_dir: str,
    user_content: Dict,
):
    """Save FSDP checkpoint: Without process groups."""
    if dist.get_rank() == 0:
        logger.warning("Full checkpoint only saves the model")

    with FSDP.state_dict_type(
        model,
        StateDictType.FULL_STATE_DICT,
        FullStateDictConfig(rank0_only=True, offload_to_cpu=True),
    ):
        state_dict = model.state_dict()
        if dist.get_rank() == 0:
            logger.info("Processed state dict to save. Starting write to disk now.")
            os.makedirs(save_dir, exist_ok=True)
            # this name is needed for HF from_pretrained API to work fine
            torch.save(state_dict, os.path.join(save_dir, "pytorch_model.bin"))
            user_content["model_config"].save_pretrained(save_dir)
        dist.barrier()


def _save_local(  # pylint: disable=too-many-arguments
    model,
    optimizer,
    scheduler,
    user_content,
    save_dir: str,
):
    """Save FSDP checkpoint: Without process groups."""
    os.makedirs(save_dir, exist_ok=True)
    with FSDP.state_dict_type(model, StateDictType.LOCAL_STATE_DICT):
        optim_state_dict = optimizer.state_dict()

        state_dict = {
            "model": model.state_dict(),
            "optimizer": optim_state_dict,
            "scheduler": scheduler.state_dict(),
        }
        # merge user content to state_dict
        state_dict = state_dict | user_content

    if dist.get_rank() == 0:
        logger.info("Processed state dict to save. Starting write to disk now.")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)

        # torch/distributed/checkpoint/filesystem.py:157: UserWarning: TypedStorage is deprecated.
        def write_fn():
            torch.save(state_dict, os.path.join(save_dir, f"{dist.get_rank()}.pt"))

        _retry_write_to_disk(write_fn)




def _delete_old_checkpoints(
    save_dir,
    num_kept_checkpoints,
    sort_fn=_CHECKPOINT_SORT_FN,
    regex=_CHECKPOINT_DIR_REGEX,
):
    if is_s3_source(str(save_dir)):
        return
    rank = dist.get_rank()
    is_rank_zero = rank == 0
    if is_rank_zero:
        tsm_checkpoint.limit_num_subdirs(
            os.path.abspath(save_dir),
            num_kept_checkpoints,
            sort_fn=sort_fn,
            regex=regex,
            log=is_rank_zero,
        )
        logger.info("Finished checkpointing to %s.", save_dir)
    dist.barrier()


def _save_async_sharded(
    model,
    optimizer,
    scheduler,
    user_content,
    save_dir: str,
    async_calls,
    root_dir,
    num_kept_checkpoints,
    checkpointing_pg_metadata,
):

    if not SUPPORT_ASYNC_SAVE:
        raise NotImplementedError("Please upgrade Megatron >= 0.7.0")

    assert async_calls is not None
    process_group, coordinator_rank, action_rank = checkpointing_pg_metadata
    # wait for the previous checkpointing
    async_calls.maybe_finalize_async_calls(
        blocking=True, process_group=process_group
    )
    _delete_old_checkpoints(
        save_dir, num_kept_checkpoints
    )

    with FSDP.state_dict_type(model, _DEFAULT_STATE_DICT_TYPE):
        optim_state_dict = FSDP.optim_state_dict(model, optimizer)
        model_state_dict = model.state_dict()
        scheduler_state_dict = scheduler.state_dict()
        state_dict = {
            "model": model_state_dict,
            "optimizer": optim_state_dict,
            "scheduler": scheduler_state_dict,
        }
        # merge user content to state_dict
        state_dict = state_dict | user_content

    if dist.get_rank(group=process_group) == 0:
        logger.info("Processed state dict to save. Starting write to disk now.")

    if action_rank:
        async_save(
            state_dict,
            checkpoint_id=save_dir,
            process_group=process_group,
            coordinator_rank=coordinator_rank,
            queue=async_calls,
        )


def _save_async_local(
    model,
    optimizer,
    scheduler,
    user_content,
    checkpoint_dir,
    async_calls,
    root_dir,
    num_kept_checkpoints,
    checkpointing_pg_metadata,
):
    if not SUPPORT_ASYNC_SAVE:
        raise NotImplementedError("Please upgrade Megatron >= 0.7.0")
    try:
        from torch.sagemaker.distributed.checkpoint.state_dict_utils import (
            sm_state_dict_type,
            SMStateDictType,
        )
    except ImportError:
        raise NotImplementedError("async save local not implemented")

    assert async_calls is not None
    # process_group, coordinator_rank, action_rank = checkpointing_pg_metadata
    global_rank = dist.get_rank()

    current_replication_ranks, current_replication_group = (
        get_current_replication_group(global_rank)
    )
    coordinator_rank = min(current_replication_ranks)

    # wait for the previous checkpointing
    async_calls.maybe_finalize_async_calls(
        blocking=True, process_group=current_replication_group
    )
    _delete_old_checkpoints(
        checkpoint_dir, num_kept_checkpoints
    )
    with sm_state_dict_type(model, SMStateDictType.SM_LOCAL_STATE_DICT):
        optim_state_dict = optimizer.state_dict()
        model_state_dict = model.state_dict()
        scheduler_state_dict = scheduler.state_dict()
        state_dict = {
            "model": model_state_dict,
            "optimizer": optim_state_dict,
            "scheduler": scheduler_state_dict,
        }
        # merge user content to state_dict
        state_dict = state_dict | user_content

    if global_rank == 0:
        logger.info("Processed state dict to save. Starting write to disk now.")
    async_save(
        state_dict,
        checkpoint_id=checkpoint_dir,
        process_group=current_replication_group,
        coordinator_rank=coordinator_rank,
        queue=async_calls,
     )


def save_checkpoint(  # pylint: disable=too-many-arguments,too-many-locals
    model,
    optimizer,
    scheduler,
    user_content,
    sharding_strategy,
    root_dir: str,
    subdir: str,
    num_kept_checkpoints: int,
    checkpointing_pg_metadata,
    tensor_parallel_degree: int,
    expert_parallel_degree: int,
    checkpoint_type=CheckpointingMethod.LOCAL,
    async_calls=None,
) -> None:
    """Export checkpoint."""
    from torch.sagemaker import state

    # seeing a NCCL crash during broadcast in checkpointing sometimes
    # seems like that happens when cached memory usage is at the limit
    # so clearing cache
    torch.cuda.empty_cache()

    if not root_dir:
        return

    save_dir = os.path.join(root_dir, subdir)
    if is_s3_source(root_dir):
        if (
            checkpoint_type != CheckpointingMethod.ASYNC_SHARDED and
            checkpoint_type != CheckpointingMethod.ASYNC_LOCAL
        ):
            save_dir = os.path.join(f"/tmp/checkpoint_{dist.get_rank()}", subdir)

    if dist.get_rank() == 0:
        logger.info("Checkpointing to %s ...", save_dir)

    if isinstance(checkpoint_type, str):
        checkpoint_type = CheckpointingMethod[checkpoint_type.upper()]

    ckpt_start = time.process_time()
    if checkpoint_type == CheckpointingMethod.SHARDED:
        if tensor_parallel_degree > 1:
            save_dir = os.path.join(
                save_dir, f"tp{tensor_parallel_degree}-{state.tp_rank}"
            )
        if expert_parallel_degree > 1:
            save_dir = os.path.join(
                save_dir, f"ep{expert_parallel_degree}-{state.ep_rank}"
            )
        _save_sharded(
            model,
            optimizer,
            scheduler,
            user_content,
            save_dir,
            checkpointing_pg_metadata,
        )
    elif checkpoint_type == CheckpointingMethod.LOCAL:
        if tensor_parallel_degree > 1 or expert_parallel_degree > 1:
            raise NotImplementedError(
                "Local checkpointing unsupported with tensor/expert parallelism"
            )
        _save_local(model, optimizer, scheduler, user_content, save_dir)
    elif checkpoint_type == CheckpointingMethod.FULL:
        _save_full(model, save_dir, user_content)
    elif checkpoint_type == CheckpointingMethod.ASYNC_SHARDED:
        if tensor_parallel_degree > 1:
            save_dir = os.path.join(
                save_dir, f"tp{tensor_parallel_degree}-{state.tp_rank}"
            )
        if expert_parallel_degree > 1:
            save_dir = os.path.join(
                save_dir, f"ep{expert_parallel_degree}-{state.ep_rank}"
            )
        return _save_async_sharded(
            model,
            optimizer,
            scheduler,
            user_content,
            save_dir,
            async_calls,
            root_dir,
            num_kept_checkpoints,
            checkpointing_pg_metadata,
        )
    elif checkpoint_type == CheckpointingMethod.ASYNC_LOCAL:
        checkpoint_dir = os.path.join(
            save_dir, f"tp{state.tp_rank}_ep{state.ep_rank}_fsdp{model.rank}"
        )
        return _save_async_local(
            model,
            optimizer,
            scheduler,
            user_content,
            checkpoint_dir,
            async_calls,
            root_dir,
            num_kept_checkpoints,
            checkpointing_pg_metadata,
        )

    ckpt_time = time.process_time() - ckpt_start
    dist.barrier()

    process_group = (
        None if checkpointing_pg_metadata is None else checkpointing_pg_metadata[0]
    )
    compute_stats_of_metric(ckpt_time, "saving checkpoint (s)", process_group)

    if dist.get_rank() == 0:
        logger.info("Finished checkpointing to %s.", save_dir)

    if is_s3_source(root_dir):
        s3_start = time.process_time()

        bucket, bucketdir = parse_s3_address(root_dir)
        bucketdir = os.path.join(bucketdir, subdir)
        import boto3

        s3_client = boto3.client("s3")
        for fname in os.listdir(save_dir):
            fpath = os.path.join(save_dir, fname)
            bucketobj = os.path.join(bucketdir, fname)
            s3_client.upload_file(fpath, bucket, bucketobj)

        s3_time = time.process_time() - s3_start
        logger.info(
            "Rank %d: saved to %s in %f sec", dist.get_rank(), bucketdir, s3_time
        )
        dist.barrier()

    # Only limit subdirs when writing intermediate checkpoints, not the final checkpoint.
    if not subdir:
        return

    # Limit checkpoints after writing the latest one.
    tsm_checkpoint.limit_num_subdirs(
        # Need to access the **full** path.
        os.path.abspath(save_dir),
        num_kept_checkpoints,
        sort_fn=_CHECKPOINT_SORT_FN,
        regex=_CHECKPOINT_DIR_REGEX,
        # Both log messages and do the actual remove as needed for one single rank.
        log=dist.get_rank() == 0,
    )


def _load_sharded_with_fn(
    load_fn,
    model,
    optimizer,
    scheduler,
    checkpoint_dir,
    checkpointing_pg_metadata,
    reader,
):
    process_group, coordinator_rank, _ = checkpointing_pg_metadata
    with FSDP.state_dict_type(
        model,
        _DEFAULT_STATE_DICT_TYPE,
        optim_state_dict_config=ShardedOptimStateDictConfig(offload_to_cpu=True),
    ):
        state_dict = {
            "model": model.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": 0,
            "total_steps": 0,
            "start_train_path_index": 0,
            "resume_from_sequence_number": 0,
            # cannot load the optimizer state_dict together with the model state_dict
        }

        def _load_from_disk(state_dict):
            # NOTE: `_{save, load}_sharded` need to be consistent using the `process_group`s.
            load_fn(
                state_dict=state_dict,
                storage_reader=reader(checkpoint_dir),
                process_group=process_group,
                coordinator_rank=coordinator_rank,
                planner=checkpoint.DefaultLoadPlanner(),
            )

        try:
            _load_from_disk(state_dict)
        except KeyError:
            # when loading old checkpoints which had start_batch_index instead of resume_from_sequence_number
            # replace the key in dummy state_dict, and retry
            del state_dict["resume_from_sequence_number"]
            state_dict["start_batch_index"] = 0
            _load_from_disk(state_dict)
        try:
            val_state_dict = {"val_resume_from_sequence_number": 0}
            _load_from_disk(val_state_dict)
            state_dict.update(val_state_dict)
        except:
            pass

        if dist.get_rank() == 0:
            logger.info("Loaded model state from disk")

        model.load_state_dict(state_dict["model"])
        scheduler.load_state_dict(state_dict["scheduler"])
        optim_state = load_sharded_optimizer_state_dict(
            model_state_dict=state_dict["model"],
            optimizer_key="optimizer",
            storage_reader=reader(checkpoint_dir),
            process_group=model.process_group,
        )

        if dist.get_rank() == 0:
            logger.info("Loaded and sharded optimizer state from disk")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            # UserWarning to replace all_gather_base with all_gather_into_tensor floods the logs
            flattened_osd = FSDP.optim_state_dict_to_load(
                model=model,
                optim=optimizer,
                optim_state_dict=optim_state["optimizer"],
            )

        if dist.get_rank() == 0:
            logger.info("Converted optimizer state dict for FSDP")

        optimizer.load_state_dict(flattened_osd)

    return state_dict


def _load_sharded(
    model, optimizer, scheduler, checkpoint_dir, checkpointing_pg_metadata
):
    return _load_sharded_with_fn(
        checkpoint.load_state_dict,
        model,
        optimizer,
        scheduler,
        checkpoint_dir,
        checkpointing_pg_metadata,
        checkpoint.FileSystemReader,
    )


def _load_async_sharded(
    model, optimizer, scheduler, checkpoint_dir, checkpointing_pg_metadata
):
    try:
        from torch.sagemaker.distributed.checkpoint.state_dict_loader import (
            load,
        )
        from torch.sagemaker.distributed.checkpoint.filesystem import (
            DistributedFileSystemReader,
        )
    except ImportError:
        raise NotImplementedError("sharded loader not implemented")

    return _load_sharded_with_fn(
        load,
        model,
        optimizer,
        scheduler,
        checkpoint_dir,
        checkpointing_pg_metadata,
        DistributedFileSystemReader,
    )


def _load_async_local(
    model, optimizer, scheduler, checkpoint_dir, checkpointing_pg_metadata
):
    try:
        from torch.sagemaker.distributed.checkpoint.state_dict_loader import (
            load,
        )
        from torch.sagemaker.distributed.checkpoint.state_dict_utils import (
            sm_state_dict_type,
            SMStateDictType,
            init_optim_state,
        )
        from torch.sagemaker.distributed.checkpoint.filesystem import (
            DistributedFileSystemReader,
        )
    except ImportError:
        raise NotImplementedError("Local loader not implemented")

    global_rank = dist.get_rank()

    current_replication_ranks, current_replication_group = (
        get_current_replication_group(global_rank)
    )
    coordinator_rank = min(current_replication_ranks)

    checkpoint_dir = Path(checkpoint_dir)
    storage_reader = DistributedFileSystemReader(checkpoint_dir)

    with sm_state_dict_type(
        model,
        SMStateDictType.SM_LOCAL_STATE_DICT,
    ):
        state_dict = {
            "model": model.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": 0,
            "total_steps": 0,
            "start_train_path_index": 0,
            "resume_from_sequence_number": 0,
        }

        init_optim_state(optimizer, skip_empty_param=True)
        state_dict["optimizer"] = optimizer.state_dict()

        def _load_from_disk(state_dict):
            load(
                state_dict=state_dict,
                process_group=current_replication_group,
                coordinator_rank=coordinator_rank,
                storage_reader=storage_reader,
            )

        try:
            _load_from_disk(state_dict)
        except KeyError:
            # when loading old checkpoints which had start_batch_index instead of resume_from_sequence_number
            # replace the key in dummy state_dict, and retry
            del state_dict["resume_from_sequence_number"]
            state_dict["start_batch_index"] = 0
            _load_from_disk(state_dict)
        try:
            val_state_dict = {"val_resume_from_sequence_number": 0}
            _load_from_disk(val_state_dict)
            state_dict.update(val_state_dict)
        except:
            pass

        if global_rank == 0:
            logger.info(f"Loaded model and optimizer state from {checkpoint_dir}")

    return state_dict


def gather_and_log_param_buffer_norms(model):
    with FSDP.state_dict_type(
        model,
        StateDictType.FULL_STATE_DICT,
        state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
    ):
        sd = model.state_dict()
        for k, v in sd.items():
            if dist.get_rank() == 0:
                print(k, torch.linalg.norm(v), v.sum())
        for n, m in model.named_buffers():
            if dist.get_rank() == 0:
                print(dist.get_rank(), n, torch.linalg.norm(m), m.sum())


def _load_local(model, optimizer, scheduler, checkpoint_dir):
    with load_with_process_group(model.process_group):
        state_dict = torch.load(os.path.join(checkpoint_dir, f"{dist.get_rank()}.pt"))

    with FSDP.state_dict_type(model, StateDictType.LOCAL_STATE_DICT):
        if dist.get_rank() == 0:
            logger.info("Loaded model state from disk")

        model.load_state_dict(state_dict["model"])
        scheduler.load_state_dict(state_dict["scheduler"])
        optimizer.load_state_dict(state_dict["optimizer"])

    return state_dict


def load_checkpoint(
    args,
    model,
    optimizer,
    scheduler,
    checkpoint_dir: str,
    sharding_strategy,
    checkpointing_pg_metadata,
    tensor_parallel_degree: int,
    expert_parallel_degree: int,
    checkpoint_type=CheckpointingMethod.LOCAL,
):
    """Load checkpoint."""
    from torch.sagemaker import state

    if dist.get_rank() == 0:
        logger.info("Loading checkpoint from %s ...", checkpoint_dir)

    load_start = time.process_time()
    if isinstance(checkpoint_type, str):
        checkpoint_type = CheckpointingMethod[checkpoint_type.upper()]

    if checkpoint_type == CheckpointingMethod.SHARDED:
        if tensor_parallel_degree > 1:
            checkpoint_dir = os.path.join(
                checkpoint_dir, f"tp{tensor_parallel_degree}-{state.tp_rank}"
            )
        if expert_parallel_degree > 1:
            checkpoint_dir = os.path.join(
                checkpoint_dir, f"ep{expert_parallel_degree}-{state.ep_rank}"
            )
        loaded = _load_sharded(
            model, optimizer, scheduler, checkpoint_dir, checkpointing_pg_metadata
        )
    elif checkpoint_type == CheckpointingMethod.ASYNC_SHARDED:
        if tensor_parallel_degree > 1:
            checkpoint_dir = os.path.join(
                checkpoint_dir, f"tp{tensor_parallel_degree}-{state.tp_rank}"
            )
        if expert_parallel_degree > 1:
            checkpoint_dir = os.path.join(
                checkpoint_dir, f"ep{expert_parallel_degree}-{state.ep_rank}"
            )
        loaded = _load_async_sharded(
            model, optimizer, scheduler, checkpoint_dir, checkpointing_pg_metadata
        )
    elif checkpoint_type == CheckpointingMethod.LOCAL:
        if tensor_parallel_degree > 1:
            raise NotImplementedError(
                "Local checkpointing unsupported with tensor parallelism"
            )
        loaded = _load_local(model, optimizer, scheduler, checkpoint_dir)
    elif checkpoint_type == CheckpointingMethod.ASYNC_LOCAL:
        checkpoint_dir = os.path.join(
            checkpoint_dir, f"tp{state.tp_rank}_ep{state.ep_rank}_fsdp{model.rank}"
        )
        loaded = _load_async_local(
            model, optimizer, scheduler, checkpoint_dir, checkpointing_pg_metadata
        )

    else:
        raise NotImplementedError

    load_time = time.process_time() - load_start
    dist.barrier()
    compute_stats_of_metric(load_time, "loading checkpoint (s)")

    if dist.get_rank() == 0:
        logger.info("Checkpoint loaded from %s.", checkpoint_dir)

    state_dict = loaded

    resume_from_sequence_number = backward_compat_get_resume_from_sequence_number(
        args, state_dict
    )
    val_resume_from_sequence_number = (
        backward_compat_get_val_resume_from_sequence_number(state_dict)
    )

    if dist.get_rank() == 0:
        logger.info(
            "Loaded state from disk: epoch %d, start_train_path_index %d, resume_from_sequence_number %d.",
            state_dict["epoch"],
            state_dict["start_train_path_index"],
            resume_from_sequence_number,
        )

    return (
        model,
        optimizer,
        scheduler,
        state_dict["epoch"],
        state_dict["total_steps"],
        state_dict["start_train_path_index"],
        resume_from_sequence_number,
        val_resume_from_sequence_number,
    )
