"""
Training script for the ROM-native student policy.

Trains a small student model to imitate a teacher (or ground-truth actions)
using the compact RomBattleState representation.

Supports:
- KL distillation from teacher logits
- Behavioral cloning from ground-truth actions
- Mixed objective (lambda_kd * KL + lambda_bc * CE)
- Multiple model sizes (4M, 2M, 1M, 500k)
- Evaluation metrics (top-1 agreement, KL, cross entropy)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from metamon.rom_native_obs.schema import (
    NUM_POKEMON_SLOTS, NUM_ACTIONS,
    POKEMON_CAT_LEN, POKEMON_NUM_LEN, POKEMON_MASK_LEN,
    POKEMON_MOVE_CAT_LEN, POKEMON_MOVE_TYPE_LEN,
    GLOBAL_CAT_LEN, GLOBAL_NUM_LEN,
)
from metamon.rom_native_obs.student_model import RomStudentPolicy, RomStudentGRUPolicy, preset_config, StudentConfig


class RomObsDataset(Dataset):
    """Dataset for ROM-native observations from .npz files."""

    def __init__(self, data_dir: str, max_files: int = -1):
        self.files = sorted(Path(data_dir).glob("*.npz"))
        if max_files > 0:
            self.files = self.files[:max_files]

        # Pre-index all (file, timestep) pairs
        self.index = []
        for fi, f in enumerate(self.files):
            npz = np.load(f, allow_pickle=True)
            T = len(npz["actions"])
            for t in range(T):
                self.index.append((fi, t))
            del npz

        # Cache for current file
        self._cached_file_idx = -1
        self._cached_data = None

    def _load_file(self, idx: int):
        if idx != self._cached_file_idx:
            self._cached_data = np.load(self.files[idx], allow_pickle=True)
            self._cached_file_idx = idx

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        fi, t = self.index[i]
        self._load_file(fi)
        d = self._cached_data

        # Skip missing actions
        action = int(d["actions"][t])
        if action < 0 or action >= NUM_ACTIONS:
            action = -1  # will be filtered in training loop

        return {
            "global_cat": torch.from_numpy(d["global_cat"][t]).long(),
            "global_num": torch.from_numpy(d["global_num"][t]).float(),
            "pokemon_cat": torch.from_numpy(d["pokemon_cat"][t]).long(),
            "pokemon_move_cat": torch.from_numpy(d["pokemon_move_cat"][t]).long(),
            "pokemon_move_type": torch.from_numpy(d["pokemon_move_type"][t]).long(),
            "pokemon_num": torch.from_numpy(d["pokemon_num"][t]).float(),
            "pokemon_mask": torch.from_numpy(d["pokemon_mask"][t]).long(),
            "legal_action_mask": torch.from_numpy(d["legal_action_mask"][t]).float(),
            "action": torch.tensor(action, dtype=torch.long),
            "teacher_logits": torch.from_numpy(d["teacher_logits"][t]).float() if "teacher_logits" in d else None,
        }


def collate_fn(batch):
    """Collate function that handles None teacher_logits."""
    result = {}
    for key in ["global_cat", "global_num", "pokemon_cat", "pokemon_move_cat",
                "pokemon_move_type", "pokemon_num", "pokemon_mask",
                "legal_action_mask", "action"]:
        result[key] = torch.stack([b[key] for b in batch])

    teacher_logits = [b["teacher_logits"] for b in batch]
    if all(tl is not None for tl in teacher_logits):
        result["teacher_logits"] = torch.stack(teacher_logits)
    else:
        result["teacher_logits"] = None
    return result


def train_epoch(model, dataloader, optimizer, device, 
                lambda_kd=1.0, lambda_bc=0.5, max_grad_norm=1.0):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_kd_loss = 0.0
    total_bc_loss = 0.0
    total_correct = 0
    total_examples = 0
    total_legal = 0

    for batch in dataloader:
        # Move to device
        inputs = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}

        # Forward pass
        logits = model(inputs)

        # Apply legal action mask for BC loss only
        legal_mask = inputs["legal_action_mask"].bool()
        logits_masked = logits.masked_fill(~legal_mask, float('-inf'))

        loss = torch.tensor(0.0, device=device)

        # KL distillation loss (on unmasked logits - teacher already handles masking)
        if lambda_kd > 0 and batch.get("teacher_logits") is not None:
            teacher_logits = batch["teacher_logits"].to(device)
            # teacher_logits are log-probs (exp sums to 1)
            teacher_sum = teacher_logits.exp().sum(dim=-1).mean().item()
            if abs(teacher_sum - 1.0) < 0.1:
                teacher_log_probs = teacher_logits
            else:
                teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)
            # Student log-probs (softmax over all actions, not masked)
            student_log_probs = F.log_softmax(logits, dim=-1)

            # Forward KL: KL(teacher || student) = sum(teacher * (log_teacher - log_student))
            kd_loss = F.kl_div(student_log_probs, teacher_log_probs, reduction='batchmean', log_target=True)
            if torch.isfinite(kd_loss):
                loss = loss + lambda_kd * kd_loss
                total_kd_loss += kd_loss.item()

        # Behavioral cloning loss
        if lambda_bc > 0:
            actions = inputs["action"]
            valid = (actions >= 0) & (actions < NUM_ACTIONS)
            if valid.any():
                bc_loss = F.cross_entropy(logits_masked[valid], actions[valid])
                loss = loss + lambda_bc * bc_loss
                total_bc_loss += bc_loss.item()

                # Accuracy
                preds = logits_masked[valid].argmax(dim=-1)
                total_correct += (preds == actions[valid]).sum().item()
                total_examples += valid.sum().item()

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        total_loss += loss.item()
        total_legal += legal_mask.sum().item()

    metrics = {
        "loss": total_loss / len(dataloader),
        "kd_loss": total_kd_loss / len(dataloader),
        "bc_loss": total_bc_loss / len(dataloader),
        "top1_acc": total_correct / max(total_examples, 1),
        "num_examples": total_examples,
    }
    return metrics


def evaluate(model, dataloader, device):
    """Evaluate the model."""
    model.eval()
    total_top1 = 0
    total_top2 = 0
    total_kl = 0.0
    total_ce = 0.0
    total_examples = 0
    total_move_correct = 0
    total_switch_correct = 0
    total_move_total = 0
    total_switch_total = 0

    with torch.no_grad():
        for batch in dataloader:
            inputs = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            logits = model(inputs)
            legal_mask = inputs["legal_action_mask"].bool()
            logits_masked = logits.masked_fill(~legal_mask, float('-inf'))

            actions = inputs["action"]
            valid = (actions >= 0) & (actions < NUM_ACTIONS)

            if not valid.any():
                continue

            logits_valid = logits_masked[valid]  # for accuracy (masked)
            logits_unmasked_valid = logits[valid]  # for KL (unmasked)
            actions_valid = actions[valid]
            legal_valid = legal_mask[valid]

            # Top-1 accuracy
            preds = logits_valid.argmax(dim=-1)
            total_top1 += (preds == actions_valid).sum().item()

            # Top-2 accuracy
            top2 = logits_valid.topk(2, dim=-1).indices
            in_top2 = (top2 == actions_valid.unsqueeze(1)).any(dim=1)
            total_top2 += in_top2.sum().item()

            # KL and CE (using unmasked logits)
            if batch.get("teacher_logits") is not None:
                teacher_logits = batch["teacher_logits"].to(device)[valid]
                # Check if log-probs or raw logits
                teacher_sum = teacher_logits.exp().sum(dim=-1).mean().item()
                if abs(teacher_sum - 1.0) < 0.1:
                    teacher_probs = teacher_logits.exp()
                    teacher_log_probs = teacher_logits
                else:
                    teacher_probs = F.softmax(teacher_logits, dim=-1)
                    teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)
                student_log_probs = F.log_softmax(logits_unmasked_valid, dim=-1)
                kl = F.kl_div(student_log_probs, teacher_log_probs, reduction='sum', log_target=True)
                if torch.isfinite(kl):
                    total_kl += kl.item()
                ce = -(teacher_probs * student_log_probs).sum()
                if torch.isfinite(ce):
                    total_ce += ce.item()

            # Per-class accuracy
            for i in range(len(actions_valid)):
                a = actions_valid[i].item()
                p = preds[i].item()
                if a < 4:
                    total_move_total += 1
                    if a == p:
                        total_move_correct += 1
                else:
                    total_switch_total += 1
                    if a == p:
                        total_switch_correct += 1

            total_examples += len(actions_valid)

    metrics = {
        "top1_acc": total_top1 / max(total_examples, 1),
        "top2_acc": total_top2 / max(total_examples, 1),
        "kl_div": total_kl / max(total_examples, 1),
        "cross_entropy": total_ce / max(total_examples, 1),
        "move_acc": total_move_correct / max(total_move_total, 1),
        "switch_acc": total_switch_correct / max(total_switch_total, 1),
        "num_examples": total_examples,
    }
    return metrics


# Model size configurations (preset names from student_model.py)
MODEL_CONFIGS = {
    "4m": "large",
    "2m": "medium",
    "1m": "small",
    "500k": "tiny",
}


def main():
    parser = argparse.ArgumentParser(description="Train ROM-native student policy")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory of .npz dataset files")
    parser.add_argument("--model_size", type=str, default="1m", choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lambda_kd", type=float, default=0.0, help="KL distillation weight")
    parser.add_argument("--lambda_bc", type=float, default=1.0, help="Behavioral cloning weight")
    parser.add_argument("--output_dir", type=str, default="./student_ckpts")
    parser.add_argument("--max_files", type=int, default=-1)
    parser.add_argument("--eval_every", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Create model
    preset = MODEL_CONFIGS[args.model_size]
    model = RomStudentPolicy(preset=preset).to(args.device)
    n_params = model.count_parameters()
    print(f"Model size: {args.model_size} (preset={preset}), Parameters: {n_params:,}")

    # Create dataset
    dataset = RomObsDataset(args.data_dir, max_files=args.max_files)
    print(f"Dataset: {len(dataset)} examples from {len(dataset.files)} files")

    # Split 90/10
    n_train = int(0.9 * len(dataset))
    n_val = len(dataset) - n_train
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=4, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate_fn, num_workers=4)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_acc = 0.0
    results_log = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_metrics = train_epoch(
            model, train_loader, optimizer, args.device,
            lambda_kd=args.lambda_kd, lambda_bc=args.lambda_bc
        )
        scheduler.step()
        train_time = time.time() - t0

        log_entry = {"epoch": epoch, "train_time_s": train_time, **train_metrics}

        if epoch % args.eval_every == 0 or epoch == args.epochs:
            val_metrics = evaluate(model, val_loader, args.device)
            log_entry.update({f"val_{k}": v for k, v in val_metrics.items()})

            if val_metrics["top1_acc"] > best_val_acc:
                best_val_acc = val_metrics["top1_acc"]
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "config": {"preset": preset},
                    "epoch": epoch,
                    "val_metrics": val_metrics,
                    "n_params": n_params,
                }, os.path.join(args.output_dir, f"best_{args.model_size}.pt"))

            print(f"Epoch {epoch}/{args.epochs} ({train_time:.1f}s) | "
                  f"Loss: {train_metrics['loss']:.4f} | "
                  f"Train Acc: {train_metrics['top1_acc']:.4f} | "
                  f"Val Acc: {val_metrics['top1_acc']:.4f} | "
                  f"Val KL: {val_metrics.get('kl_div', 0):.4f}")
        else:
            print(f"Epoch {epoch}/{args.epochs} ({train_time:.1f}s) | "
                  f"Loss: {train_metrics['loss']:.4f} | "
                  f"Train Acc: {train_metrics['top1_acc']:.4f}")

        results_log.append(log_entry)

    # Save final model and results
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": {"preset": preset},
        "epoch": args.epochs,
        "n_params": n_params,
    }, os.path.join(args.output_dir, f"final_{args.model_size}.pt"))

    with open(os.path.join(args.output_dir, f"results_{args.model_size}.json"), "w") as f:
        json.dump({
            "model_size": args.model_size,
            "n_params": n_params,
            "config": {"preset": preset},
            "best_val_acc": best_val_acc,
            "epochs": args.epochs,
            "log": results_log,
        }, f, indent=2)

    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
