import datetime as dt
import glob
import json
import os
import pickle
import queue
import shutil
import threading
import gc

import click
import numpy as np


@click.command(name="microzoi")
@click.option("--gpu", "-g", default="0", help="GPU number")
@click.option("--param_file", "-p", help="JSON file with parameters file")
@click.option("--output_folder", "-o", help="Output folder")
@click.option("--continue_training", "-c", is_flag=True, help="Continue training from last epoch")
@click.option("--val_fold", "-v", default=3, help="Validation fold")
@click.option("--test_fold", "-t", default=4, help="Test fold")
def train_microzoi(gpu, param_file, output_folder, continue_training, val_fold, test_fold):
    print(f"Training with GPU: {gpu}")
    print(f"Parameters file: {param_file}")
    print(f"Output folder: {output_folder}")
    print(f"Continue training: {continue_training}")
    print(f"Validation fold: {val_fold}")
    print(f"Test fold: {test_fold}")
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    import torch

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("high")
    import torch.optim as optim
    from torch import GradScaler, autocast

    from manta_hic.nn.microzoi import MicroBorzoi, borzoi_loss, corr, dataGenerator
    from manta_hic.ops.tensor_ops import list_to_tensor_batch
    from manta_hic.training_meta import get_strand_pair

    params = json.load(open(param_file))
    model_params = params["model"]
    storage_params = params["storage"]
    train_params = params["train"]

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

    folder = output_folder
    if not os.path.exists(folder):
        os.mkdir(folder)

    data_queue = queue.Queue(maxsize=3)  # Adjust maxsize based on memory constraints

    def data_loader_thread(data_queue, params):
        for data in dataGenerator(**params):
            data_queue.put(data)
        data_queue.put(None)

    # -------------- Initializing model and loading saved model  --------------
    model = MicroBorzoi(**model_params)
    model = model.to(DEVICE)

    # load if continue training, or raise error if folder is not empty

    st_epoch = 0
    if continue_training:  # load last model
        mods = glob.glob(f"{folder}/model_*.pth")
        assert len(mods) > 0, "No models found in output folder"
        mods = sorted(mods, key=lambda x: int(x.split("/")[-1].split("_")[1].split(".")[0]))
        model.load_state_dict(torch.load(mods[-1]))
        st_epoch = int(mods[-1].split("/")[-1].split("_")[1].split(".")[0]) + 1

    shutil.copy(param_file, f"{folder}/params.json")
    max_lr = train_params["learning_rate"]
    ramp_up_epochs = train_params["ramp_up_epochs"]
    num_epochs = train_params["num_epochs"]
    optimizer = optim.Adam(model.parameters(), lr=max_lr)
    scaler = GradScaler("cuda")
    run_corr = 0

    for epoch in range(st_epoch, num_epochs):
        for param_group in optimizer.param_groups:  # set learning rate with ramp up
            param_group["lr"] = max_lr * min((epoch + 1) / ramp_up_epochs, 1)

        train_corrs_epoch, val_corrs_epoch = [], []
        loader_thread = threading.Thread(target=data_loader_thread, args=(data_queue, storage_params))
        loader_thread.start()
        model.train()

        while (dta := data_queue.get()) is not None:  # training loop
            que_len = data_queue.qsize()
            (in_data, target), global_meta, meta = dta
            genome, shift_bins = global_meta["genome"], global_meta["shift_bins"]

            current = dt.datetime.now()
            in_data = list_to_tensor_batch(in_data, DEVICE, dtype=torch.float32).permute(0, 2, 1)
            target = list_to_tensor_batch(target, DEVICE, dtype=torch.float32).permute(0, 2, 1)
            target.requires_grad = False

            if np.random.random() < 0.5:  # random flip
                in_data = in_data.flip(dims=(1, 2))
                target = torch.flip(target[:, get_strand_pair(genome), :], dims=(2,))
                shift_bins = -shift_bins

            optimizer.zero_grad()
            with autocast("cuda"):
                output = model(in_data, genome=genome, offset=shift_bins)  # [B, C, N]
                loss = borzoi_loss(output, target)
            del in_data, target
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_corrs_epoch.append(mean_corr := float(corr(target, output)))
            run_corr = (1 / (E := 100 * (epoch + 1))) * float(mean_corr) + ((E - 1) / E) * run_corr
            print(f"Epoch [{epoch}/{num_epochs}], queue len {que_len}, {genome} loss={loss.item():.3g}", end="")
            print(f"corr={mean_corr:.3g} running {run_corr:.3g} ", end="")
            print("duration: ", (dt.datetime.now() - current) / dt.timedelta(seconds=1))
            del output, loss, mean_corr, run_corr

        loader_thread.join()

        gc.collect()
        # validation loop
        print(f"Epoch {epoch} validation")
        model.eval()
        loader_thread = threading.Thread(target=data_loader_thread, args=(data_queue, validation_params))
        loader_thread.start()
        while (dta := data_queue.get()) is not None:
            (in_data, target), global_meta, meta = dta
            genome, shift_bins = global_meta["genome"], global_meta["shift_bins"]

            in_data = list_to_tensor_batch(in_data, DEVICE, dtype=torch.float32).permute(0, 2, 1)
            target = list_to_tensor_batch(target, DEVICE, dtype=torch.float32).permute(0, 2, 1)
            target.requires_grad = False

            with torch.no_grad(), autocast("cuda"):
                assert shift_bins == 0
                output = model(in_data, genome=genome, offset=shift_bins)
                val_corrs_epoch.append(mean_corr := float(corr(target, output)))
                print(f"Epoch [{epoch}/{num_epochs}], {genome} val corr={mean_corr:.3g}")
            del in_data, target, output
        loader_thread.join()

        torch.save(model.state_dict(), f"{folder}/model_{epoch}.pth")
        pickle.dump((train_corrs_epoch, val_corrs_epoch), open(f"{folder}/corrs_{epoch}.pkl", "wb"))

        print(
            f"Epoch {epoch} done, train corr {np.mean(train_corrs_epoch):.3g}, val corr {np.mean(val_corrs_epoch):.3g}"
        )


if __name__ == "__main__":
    train_microzoi()
