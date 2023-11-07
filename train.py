"""Module providing a function training neural network."""
import os
import torch
import torch.nn as nn
import utils
from model import Classifier
from pathlib import Path
from sys import platform
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm


def parse_args():
    """arguments"""
    if platform == "linux":
        data_path = "./data/Voxceleb2"
    elif platform == "win32":
        data_dir = "Data/Voxceleb2"
        current_dir = os.getcwd()
        current_path = Path(current_dir)
        data_path = current_path.parents[2] / Path(data_dir)

    config = {
        "data_dir": data_path,
        "save_path": "model.ckpt",
        "batch_size": 32,
        "n_workers": 8,
        "valid_steps": 2000,
        "warmup_steps": 1000,
        "save_steps": 10000,
        "total_steps": 400000,
    }

    return config


def main(
        data_dir,
        save_path,
        batch_size,
        n_workers,
        valid_steps,
        warmup_steps,
        total_steps,
        save_steps,
):
    """Main function."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Info]: Use {device} now!")

    train_loader, valid_loader, speaker_num = utils.get_dataloader(
        data_dir, batch_size, n_workers)
    train_iterator = iter(train_loader)
    print(f"[Info]: Finish loading data!", flush=True)

    model = Classifier(n_spks=speaker_num).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=1e-3)
    scheduler = utils.get_cosine_schedule_with_warmup(
        optimizer, warmup_steps, total_steps)
    print(f"[Info]: Finish creating model!", flush=True)

    best_accuracy = -1.0
    best_state_dict = None

    pbar = tqdm(total=valid_steps, ncols=0, desc="Train", unit=" step")

    for step in range(total_steps):
        # Get data
        try:
            batch = next(train_iterator)
        except StopIteration:
            train_iterator = iter(train_loader)
            batch = next(train_iterator)

        loss, accuracy = utils.model_fn(batch, model, criterion, device)
        batch_loss = loss.item()
        batch_accuracy = accuracy.item()

        # Updata model
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        # Log
        pbar.update()
        pbar.set_postfix(
            loss=f"{batch_loss:.2f}",
            accuracy=f"{batch_accuracy:.2f}",
            step=step + 1,
        )

        # Do validation
        if (step + 1) % valid_steps == 0:
            pbar.close()

            valid_accuracy = utils.valid(
                valid_loader, model, criterion, device)

            # keep the best model
            if valid_accuracy > best_accuracy:
                best_accuracy = valid_accuracy
                best_state_dict = model.state_dict()

            pbar = tqdm(total=valid_steps, ncols=0, desc="Train", unit=" step")

        # Save the best model so far.
        if (step + 1) % save_steps == 0 and best_state_dict is not None:
            torch.save(best_state_dict, save_path)
            pbar.write(
                f"\n Step {step + 1}, best model saved. (accuracy={best_accuracy:.4f})")

    pbar.close()


if __name__ == "__main__":
    main(**parse_args())
