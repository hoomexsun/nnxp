from pathlib import Path
from typing import Tuple, Dict, List
import copy
import heapq
import logging

import torch


def save_models(
    exp_dir: Path,
    logger: logging.Logger,
    current_model: torch.nn.Module,
    model_conf: dict,
    device: torch.device,
    epoch: int,
    max_epoch: int,
    avg_train_loss: float,
    avg_val_loss: float,
    wa_score: float,
    best_train_loss: float,
    best_val_loss: float,
    best_wa: float,
    best_models: list,
    saved_epochs: set,
    n_best: int,
) -> Tuple[float, float, float, list, set]:
    # Save best training loss
    if avg_train_loss < best_train_loss:
        best_train_loss = avg_train_loss
        torch.save(current_model.state_dict(), exp_dir / "train.loss.best.pth")
        logger.info(f"Saved best training loss model at epoch {epoch}.")

    # Save best validation loss
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(current_model.state_dict(), exp_dir / "val.loss.best.pth")
        logger.info(f"Saved best validation loss model at epoch {epoch}.")

    # Save best word accuracy
    if wa_score > best_wa:
        best_wa = wa_score
        torch.save(current_model.state_dict(), exp_dir / "wa.best.pth")
        logger.info(f"Saved best word accuracy model at epoch {epoch}.")

    # Save latest
    torch.save(current_model.state_dict(), exp_dir / "latest.pth")
    logger.info(f"Saved latest model at epoch {epoch}.")

    # Save top-N best by validation loss using max-heap (invert val_loss)
    if epoch not in saved_epochs:
        if len(best_models) < n_best:
            heapq.heappush(
                best_models,
                (-avg_val_loss, epoch, copy.deepcopy(current_model.state_dict())),
            )
            saved_epochs.add(epoch)
            torch.save(current_model.state_dict(), exp_dir / f"{epoch}epoch.pth")
        else:
            worst_neg_loss, worst_epoch, _ = best_models[0]
            if -avg_val_loss > worst_neg_loss:
                removed = heapq.heappushpop(
                    best_models,
                    (-avg_val_loss, epoch, copy.deepcopy(current_model.state_dict())),
                )
                saved_epochs.discard(removed[1])
                saved_epochs.add(epoch)

                # Remove old worst model
                worst_path = exp_dir / f"{removed[1]}epoch.pth"
                if worst_path.exists():
                    worst_path.unlink()
                    logger.info(f"Removed evicted model from epoch {removed[1]}.")

                # Save new top model
                torch.save(current_model.state_dict(), exp_dir / f"{epoch}epoch.pth")

    # Save averaged model at end
    if epoch == max_epoch:
        avg_model = average_model_weights(
            best_models, current_model, model_conf, device
        )
        torch.save(avg_model.state_dict(), exp_dir / "val.loss.ave.pth")
        logger.info("Saved averaged model at the end of training.")

    return best_train_loss, best_val_loss, best_wa, best_models, saved_epochs


def average_model_weights(
    model_heap: list[tuple[float, int, dict]],
    current_model: torch.nn.Module,
    model_conf: Dict,
    device: torch.device,
) -> torch.nn.Module:
    n_models = len(model_heap)
    assert n_models > 0, "No models to average."
    avg_state_dict = copy.deepcopy(model_heap[0][2])

    for key in avg_state_dict.keys():
        for i in range(1, n_models):
            avg_state_dict[key] += model_heap[i][2][key]
        avg_state_dict[key] /= n_models

    avg_model = copy.deepcopy(current_model)
    avg_model.load_state_dict(avg_state_dict)
    avg_model.to(device)
    return avg_model


def save_best_predictions(
    exp_dir: Path, xs: List[str], true_texts: List[str], pred_texts: List[str]
) -> None:
    correct_flags = ["✔" if p == t else "✘" for p, t in zip(pred_texts, true_texts)]
    results = [
        f"{x}\t{t}\t{p}\t{c}"
        for x, t, p, c in zip(xs, true_texts, pred_texts, correct_flags)
    ]
    decode_path = exp_dir / "decode/wa.best.decode"
    decode_path.parent.mkdir(parents=True, exist_ok=True)
    decode_path.write_text("\n".join(results), encoding="utf-8")
