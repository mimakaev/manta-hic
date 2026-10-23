import copy
import datetime as dt
import gc
import glob
import json
import math
import os
import pickle
import queue
import shutil
import threading

import click
import numpy as np

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Training-recipe keys of the params file's "train" block and their defaults. The defaults reproduce the recipe
# of the first-generation MicroZoi trunks (Adam, constant LR after warm-up, fp16, no clipping, no EMA).
TRAIN_DEFAULTS = {
    "optimizer": "adam",  # "adam" | "adamw"
    "weight_decay": 0.0,  # decoupled (AdamW); never applied to norm gains or biases
    "betas": [0.9, 0.999],
    "schedule": "constant",  # "constant" (after warm-up) | "cosine" (to min_lr_frac * learning_rate at the end)
    "min_lr_frac": 0.0,
    "warmup_steps": None,  # None: ramp_up_epochs worth of steps
    "grad_clip": None,  # global gradient-norm clip, None = off
    "precision": "fp16",  # "fp16" (with GradScaler) | "bf16" | "fp32"
    "compile": False,  # torch.compile the trunk
    "ema_decay": None,  # parameter EMA; when set, validation and ema_<epoch>.pth use the EMA weights
    "log_every": 20,  # steps between progress lines (each one syncs the GPU)
}


def lr_at(step, total_steps, max_lr, warmup_steps, schedule, min_lr_frac):
    """Learning rate at optimizer step ``step`` (0-based): linear warm-up, then constant or cosine decay."""
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if schedule == "constant":
        return max_lr
    if schedule == "cosine":
        progress = min((step - warmup_steps) / max(total_steps - warmup_steps, 1), 1.0)
        return max_lr * (min_lr_frac + (1 - min_lr_frac) * 0.5 * (1 + math.cos(math.pi * progress)))
    raise ValueError(f"Unknown schedule {schedule!r}")


@click.command(name="microzoi", context_settings={"show_default": True})
@click.option("--gpu", "-g", default="0", help="GPU number")
@click.option("--param_file", "-p", help="JSON file with parameters file")
@click.option("--output_folder", "-o", help="Output folder")
@click.option("--continue_training", "-c", is_flag=True, help="Continue training from the last saved epoch")
@click.option("--val_fold", "-v", default=3, help="Validation fold, or `-1` to skip validation")
@click.option("--test_fold", "-t", default=4, help="Test fold")
@click.option(
    "--max_steps",
    default=None,
    type=int,
    help="Cap each epoch's training and validation steps (smoke tests/benchmarks)",
)
def train_microzoi(gpu, param_file, output_folder, continue_training, val_fold, test_fold, max_steps):
    print(f"Training with GPU: {gpu}")
    print(f"Parameters file: {param_file}")
    print(f"Output folder: {output_folder}")
    print(f"Continue training: {continue_training}")
    print(f"Validation fold: {val_fold}")
    print(f"Test fold: {test_fold}")
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    import torch

    if torch.cuda.is_available():
        DEVICE = torch.device("cuda")
    elif torch.backends.mps.is_available():
        DEVICE = torch.device("mps")
    else:
        DEVICE = torch.device("cpu")
    torch.set_float32_matmul_precision("high")
    from torch import GradScaler, autocast

    from manta_hic.nn.microzoi import Microzoi, borzoi_loss, corr, dataGenerator, sequence_table
    from manta_hic.nn.training_meta import get_strand_pair
    from manta_hic.ops.tensor_ops import list_to_tensor_batch

    params = json.load(open(param_file))
    model_params = params["model"]
    storage_params = params["storage"]
    train_params = params["train"]
    unknown = set(train_params) - set(TRAIN_DEFAULTS) - {"num_epochs", "learning_rate", "ramp_up_epochs"}
    unknown -= {"max_shift", "batch_size"}
    if unknown:
        raise ValueError(f"Unknown train parameters: {sorted(unknown)}")
    tp = TRAIN_DEFAULTS | train_params
    print("Train parameters:", tp)

    # -------------- Initializing storage  --------------
    # populate full dataGenerator arguments
    storage_params["batch_size"] = train_params["batch_size"]
    storage_params["val_fold"] = val_fold
    storage_params["test_fold"] = test_fold
    storage_params["max_shift"] = train_params["max_shift"]
    storage_params["mode"] = "train"
    storage_params["seq_length"] = model_params["seq_length"]

    print("storage location", storage_params["fasta_path"])

    validation_params = storage_params.copy()
    validation_params["mode"] = "val"

    table_args = {k: storage_params[k] for k in ("base", "genomes", "seq_length", "batch_size", "val_fold")}
    table_args |= {"test_fold": test_fold}
    steps_per_epoch = len(sequence_table(**table_args, mode="train")) // storage_params["batch_size"]
    if max_steps is not None:
        steps_per_epoch = min(steps_per_epoch, max_steps)
    num_epochs = tp["num_epochs"]
    total_steps = num_epochs * steps_per_epoch
    warmup_steps = tp["warmup_steps"] or tp["ramp_up_epochs"] * steps_per_epoch
    print(f"{steps_per_epoch} steps per epoch, {total_steps} total, {warmup_steps} warm-up")

    folder = output_folder
    if not os.path.exists(folder):
        os.mkdir(folder)

    data_queue = queue.Queue(maxsize=3)  # Adjust maxsize based on memory constraints

    def data_loader_thread(data_queue, params, stop):
        for data in dataGenerator(**params):
            if stop.is_set():
                break
            data_queue.put(data)
        data_queue.put(None)

    # -------------- Initializing model, optimizer and loading saved state  --------------
    model = Microzoi(**model_params).to(DEVICE)
    ema = copy.deepcopy(model).eval().requires_grad_(False) if tp["ema_decay"] else None

    # no weight decay on norm gains and biases
    decay = [p for p in model.parameters() if p.ndim >= 2]
    no_decay = [p for p in model.parameters() if p.ndim < 2]
    groups = [{"params": decay, "weight_decay": tp["weight_decay"]}, {"params": no_decay, "weight_decay": 0.0}]
    opt_class = {"adam": torch.optim.Adam, "adamw": torch.optim.AdamW}[tp["optimizer"]]
    optimizer = opt_class(groups, lr=tp["learning_rate"], betas=tuple(tp["betas"]), fused=DEVICE.type == "cuda")

    amp_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": None}[tp["precision"]]
    scaler = GradScaler(DEVICE.type, enabled=amp_dtype == torch.float16)

    st_epoch = 0
    if continue_training:
        if os.path.exists(f"{folder}/state.pth"):  # full training state: model, optimizer, scaler, EMA
            state = torch.load(f"{folder}/state.pth", map_location=DEVICE)
            model.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
            scaler.load_state_dict(state["scaler"])
            if ema is not None:
                ema.load_state_dict(state["ema"])
            st_epoch = state["epoch"] + 1
        else:  # runs from before state.pth: weights only
            mods = glob.glob(f"{folder}/model_*.pth")
            assert len(mods) > 0, "No models found in output folder"
            mods = sorted(mods, key=lambda x: int(x.split("/")[-1].split("_")[1].split(".")[0]))
            sd = torch.load(mods[-1])
            sd = {k: v for k, v in sd.items() if not k.endswith("freqs_cis")}  # now a non-persistent buffer
            model.load_state_dict(sd, strict=True)
            if ema is not None:
                ema.load_state_dict(sd)
            st_epoch = int(mods[-1].split("/")[-1].split("_")[1].split(".")[0]) + 1
        print(f"Resuming at epoch {st_epoch}")

    if tp["compile"]:
        model.compile_trunk()
        if ema is not None:
            ema.compile_trunk()

    shutil.copy(param_file, f"{folder}/params.json")
    model_params_list = list(model.parameters())
    ema_params_list = list(ema.parameters()) if ema is not None else None
    run_corr = 0
    step = st_epoch * steps_per_epoch

    for epoch in range(st_epoch, num_epochs):
        train_corrs_epoch, val_corrs_epoch, train_losses, val_losses, grad_norms = [], [], [], [], []
        stop = threading.Event()
        loader_thread = threading.Thread(target=data_loader_thread, args=(data_queue, storage_params, stop))
        loader_thread.start()
        model.train()
        epoch_start = dt.datetime.now()
        epoch_step = 0

        while (dta := data_queue.get()) is not None:  # training loop
            if epoch_step >= steps_per_epoch:  # --max_steps: drain the loader
                stop.set()
                continue
            que_len = data_queue.qsize()
            (in_data, target), global_meta, meta = dta
            genome, shift_bins = global_meta["genome"], global_meta["shift_bins"]

            in_data = list_to_tensor_batch(in_data, DEVICE, dtype=torch.float32).permute(0, 2, 1)
            target = list_to_tensor_batch(target, DEVICE, dtype=torch.float32).permute(0, 2, 1)

            if np.random.random() < 0.5:  # random flip
                in_data = in_data.flip(dims=(1, 2))
                target = torch.flip(target[:, get_strand_pair(genome), :], dims=(2,))
                shift_bins = -shift_bins

            lr = lr_at(step, total_steps, tp["learning_rate"], warmup_steps, tp["schedule"], tp["min_lr_frac"])
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            optimizer.zero_grad(set_to_none=True)
            with autocast(DEVICE.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                output = model(in_data, genome=genome, offset=shift_bins)  # [B, C, N]
            output = output.float()
            loss = borzoi_loss(output, target)  # in float32

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            max_norm = tp["grad_clip"] if tp["grad_clip"] is not None else float("inf")
            grad_norms.append(torch.nn.utils.clip_grad_norm_(model_params_list, max_norm).detach())
            scaler.step(optimizer)
            scaler.update()
            if ema is not None:
                torch._foreach_lerp_(ema_params_list, model_params_list, 1 - tp["ema_decay"])

            train_losses.append(loss.detach())
            step += 1
            epoch_step += 1
            if epoch_step % tp["log_every"] == 0:
                train_corrs_epoch.append(mean_corr := float(corr(target, output.detach())))
                run_corr = (1 / (E := 5 * (epoch + 1))) * float(mean_corr) + ((E - 1) / E) * run_corr
                sec_per_step = (dt.datetime.now() - epoch_start) / dt.timedelta(seconds=1) / epoch_step
                print(
                    f"Epoch [{epoch}/{num_epochs}] step {epoch_step}/{steps_per_epoch}, queue len {que_len}, ", end=""
                )
                print(f"{genome} loss={loss.item():.4g} corr={mean_corr:.3g} running {run_corr:.3g} ", end="")
                print(f"lr={lr:.3g} grad_norm={float(grad_norms[-1]):.3g} {sec_per_step:.3f} s/step", end="\r")
        print()

        loader_thread.join()
        train_seconds = (dt.datetime.now() - epoch_start) / dt.timedelta(seconds=1)
        gc.collect()

        # validation loop -- on the EMA weights when EMA is on
        if val_fold != -1:
            print(f"Epoch {epoch} validation")
            val_model = ema if ema is not None else model
            val_model.eval()
            stop = threading.Event()
            loader_thread = threading.Thread(target=data_loader_thread, args=(data_queue, validation_params, stop))
            loader_thread.start()
            val_step = 0
            while (dta := data_queue.get()) is not None:
                if max_steps is not None and val_step >= max_steps:
                    stop.set()
                    continue
                val_step += 1
                (in_data, target), global_meta, meta = dta
                genome, shift_bins = global_meta["genome"], global_meta["shift_bins"]

                in_data = list_to_tensor_batch(in_data, DEVICE, dtype=torch.float32).permute(0, 2, 1)
                target = list_to_tensor_batch(target, DEVICE, dtype=torch.float32).permute(0, 2, 1)

                assert shift_bins == 0
                with torch.no_grad():
                    with autocast(DEVICE.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                        output = val_model(in_data, genome=genome, offset=shift_bins)
                    output = output.float()
                    val_losses.append(float(borzoi_loss(output, target)))
                    val_corrs_epoch.append(mean_corr := float(corr(target, output)))
                print(f"Epoch [{epoch}/{num_epochs}], {genome} val corr={mean_corr:.3g}", end="\r")
            print()
            loader_thread.join()
        else:
            val_corrs_epoch = [0]

        torch.save(model.state_dict(), f"{folder}/model_{epoch}.pth")
        if ema is not None:
            torch.save(ema.state_dict(), f"{folder}/ema_{epoch}.pth")
        state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict()}
        state |= {"ema": ema.state_dict() if ema is not None else None, "epoch": epoch}
        torch.save(state, f"{folder}/state.pth.tmp")
        os.replace(f"{folder}/state.pth.tmp", f"{folder}/state.pth")
        pickle.dump((train_corrs_epoch, val_corrs_epoch), open(f"{folder}/corrs_{epoch}.pkl", "wb"))
        metrics = {
            "epoch": epoch,
            "step": step,
            "lr": lr,
            "train_loss": float(torch.stack(train_losses).mean()),
            "train_corr": float(np.mean(train_corrs_epoch)) if train_corrs_epoch else None,
            "grad_norm_mean": float(torch.stack(grad_norms).mean()),
            "grad_norm_max": float(torch.stack(grad_norms).max()),
            "val_loss": float(np.mean(val_losses)) if val_losses else None,
            "val_corr": float(np.mean(val_corrs_epoch)),
            "val_weights": "ema" if ema is not None else "model",
            "train_seconds": train_seconds,
            "sec_per_step": train_seconds / max(epoch_step, 1),
        }
        json.dump(metrics, open(f"{folder}/metrics_{epoch}.json", "w"), indent=1)
        print(f"Epoch {epoch} done: {metrics}")


if __name__ == "__main__":
    train_microzoi()
