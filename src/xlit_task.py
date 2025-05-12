import time
from pathlib import Path
from typing import Optional, List, Tuple

import torch
import yaml
from jiwer import cer, wer
from torch.utils.tensorboard import SummaryWriter

from .tokenizer import prepare_tokenizers
from .data.loader import load_dataloaders
from .models import load_model
from .utils import (
    set_seed,
    setup_logger,
    plot_metrics,
    save_models,
    save_best_predictions,
)


class XlitTask:
    def __init__(
        self, conf_file: str | Path = "train.yaml", ckpt_file: str | Path = None
    ) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._load_config(Path("conf") / conf_file)
        set_seed(self.conf["seed"])
        self.exp_dir = (
            Path("exp")
            / f"xlit_train_{self.model_name}_{self.token_type}_{self.lang_pair}"
        )
        self.data_dir = self.exp_dir / "data"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.model = self._build_model(ckpt_file).to(self.device)

    @classmethod
    def from_pretrained(cls, ckpt_file: str | Path, conf_file: str | Path = None):
        ckpt_file = Path(ckpt_file)
        if not ckpt_file.exists():
            raise FileNotFoundError(
                f"Checkpoint file not found: {ckpt_file.as_posix()}"
            )
        return cls(conf_file, ckpt_file)

    def __call__(self, x: str, max_len: Optional[int] = None) -> str:
        return self.infer(x, max_len)

    def _load_config(self, conf_path) -> None:
        with open(conf_path, "r") as f:
            content = yaml.safe_load(f)
        self.model_name, self.conf = content["xlit"], content["xlit_conf"]
        self.token_type = self.conf["token_type"]
        self.langx, self.langy = self.conf["langx"], self.conf["langy"]
        self.lang_pair = f"{self.langx}_{self.langy}"

    def _build_model(self, ckpt_file: Optional[str | Path]) -> torch.nn.Module:
        self.x_tokenizer, self.y_tokenizer, self.xs, self.ys = prepare_tokenizers(
            x_tokens_file=self.data_dir / f"{self.langx}_{self.token_type}_tokens.txt",
            y_tokens_file=self.data_dir / f"{self.langy}_{self.token_type}_tokens.txt",
            db_file=self.conf["db_file"],
        )
        self.conf["idim"], self.conf["odim"] = len(self.x_tokenizer), len(
            self.y_tokenizer
        )
        self.conf["pad_token"] = self.y_tokenizer.tok2idx.get("<pad>", 0)
        self.conf["sos_token"] = self.y_tokenizer.tok2idx.get("<sos>", 1)
        self.conf["eos_token"] = self.y_tokenizer.tok2idx.get("<eos>", 2)
        self.conf["max_len"] = self.conf.get("max_len", 100)
        model = load_model(self.model_name, self.conf, device=self.device)
        if ckpt_file:
            model.load_state_dict(torch.load(ckpt_file, map_location=self.device))
        return model

    def infer(self, x: str, max_len: Optional[int] = None) -> str:
        max_len = max_len or self.conf.get("max_len", 100)
        tokenized_x = self.x_tokenizer.encode(x, max_len=max_len)
        input_tensor = torch.tensor(tokenized_x).unsqueeze(0).to(self.device)
        with torch.no_grad():
            y_pred = self.model(
                input_tensor,
                max_len=max_len,
            )
        predicted_ids = y_pred.argmax(dim=2)
        return self.y_tokenizer.decode(predicted_ids[0].tolist())

    def _prepare_data(self) -> None:
        (
            self.train_loader,
            self.val_loader,
        ) = load_dataloaders(
            self.xs,
            self.ys,
            self.x_tokenizer,
            self.y_tokenizer,
            max_len=self.conf["max_len"],
            batch_size=self.conf["batch_size"],
            val_ratio=self.conf["val_ratio"],
            train_file=self.data_dir / f"train_{self.lang_pair}.txt",
            val_file=self.data_dir / f"val_{self.lang_pair}.txt",
            seed=self.conf["seed"],
        )

    def _compute_loss(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        return self.criterion(
            y_pred[:, 1:].reshape(-1, y_pred.shape[2]),
            y_true[:, 1:].reshape(-1),
        )

    def _train_step(self, batch) -> float:
        x, y = batch["input"].to(self.device), batch["target"].to(self.device)
        self.optimizer.zero_grad()
        y_pred = self.model(x, y)
        loss = self._compute_loss(y_pred, y)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def _val_step(self, batch) -> Tuple[float, List[str], List[str]]:
        x, y = batch["input"].to(self.device), batch["target"].to(self.device)
        with torch.no_grad():
            y_pred = self.model(x, y)
            loss = self._compute_loss(y_pred, y)
            pred_texts = [
                self.y_tokenizer.decode(seq.tolist())
                for seq in y_pred.argmax(dim=-1).cpu()
            ]
            true_texts = [self.y_tokenizer.decode(seq.tolist()) for seq in y]
        return loss.item(), pred_texts, true_texts

    def _init_logs(self):
        self.logger.info("Data Information:")
        self.logger.info(f"Batch size: {self.conf['batch_size']}")
        self.logger.info(
            f"[Training] Data size: {len(self.train_loader.dataset)} to {len(self.train_loader)} batches"
        )
        self.logger.info(
            f"[Validation] Data size: {len(self.val_loader.dataset)} to {len(self.val_loader)} batches"
        )
        self.logger.info("Token information")
        self.logger.info(
            f"[{self.langx}] Tokenizer loaded with {len(self.x_tokenizer)} tokens."
        )
        self.logger.info(
            f"[{self.langy}] Tokenizer loaded with {len(self.y_tokenizer)} tokens."
        )
        total_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        self.logger.info(f"Model information:\n{self.model}")
        self.logger.info(f"Total trainable parameters: {total_params}")
        self.logger.info(f"Experiment directory: {self.exp_dir.as_posix()}")
        self.logger.info(f"Optimizer: {self.optimizer.__class__.__name__}")
        self.logger.info(f"Loss criterion: {self.criterion.__class__.__name__}")

    def train(self):
        start_time = time.time()
        image_dir = self.exp_dir / "images"

        self.tb_writer = SummaryWriter(log_dir=self.exp_dir / "tensorboard")
        self.logger = setup_logger(
            log_file=self.exp_dir / "train.log",
            backup_file=self.exp_dir / "train.old.log",
        )
        self._prepare_data()

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.conf["optim_conf"]["lr"]
        )
        self.criterion = torch.nn.CrossEntropyLoss()
        self._init_logs()

        max_epoch = self.conf.get("max_epoch", 100)
        n_best = self.conf.get("keep_nbest_models", 5)

        best_val_loss = float("inf")
        best_wa = -float("inf")
        best_train_loss = float("inf")

        best_models = []
        saved_epochs = set()

        train_losses, val_losses, val_cers, val_accs = [], [], [], []

        self.logger.info(f"Started training {self.model_name} model on [{self.device}]")
        start_epoch = 1
        try:
            for epoch in range(start_epoch, max_epoch + 1):
                epoch_start_time = time.time()
                self.logger.info(f"Epoch {epoch}/{max_epoch}")

                # Training
                self.model.train()
                avg_train_loss = sum(
                    self._train_step(batch) for batch in self.train_loader
                ) / len(self.train_loader)
                train_losses.append(avg_train_loss)

                self.tb_writer.add_scalar("Loss/Train", avg_train_loss, epoch)
                self.logger.info(f"Train Loss: {avg_train_loss:.4f}")

                # Validation
                self.model.eval()
                val_loss = 0.0
                pred_texts, true_texts = [], []

                val_xs = []
                for batch in self.val_loader:
                    val_xs.extend(
                        [self.x_tokenizer.decode(x.tolist()) for x in batch["input"]]
                    )
                    loss, preds, trues = self._val_step(batch)
                    val_loss += loss
                    pred_texts.extend(preds)
                    true_texts.extend(trues)

                avg_val_loss = val_loss / len(self.val_loader)
                cer_score = cer(true_texts, pred_texts)
                wa_score = 1 - wer(true_texts, pred_texts)
                if best_wa < wa_score:
                    save_best_predictions(
                        self.exp_dir,
                        val_xs,
                        true_texts,
                        pred_texts,
                    )
                    self.logger.info(
                        f"Saved best word accuracy predictions at epoch {epoch}."
                    )

                val_losses.append(avg_val_loss)
                val_cers.append(cer_score)
                val_accs.append(wa_score)

                self.tb_writer.add_scalar("Loss/Val", avg_val_loss, epoch)
                self.tb_writer.add_scalar("CER/Val", cer_score, epoch)
                self.tb_writer.add_scalar("Word Accuracy/Val", wa_score, epoch)
                self.logger.info(
                    f"Val Loss: {avg_val_loss:.4f}, CER: {cer_score:.4f}, Word Accuracy: {wa_score:.4f}"
                )

                epoch_duration = time.time() - epoch_start_time
                elapsed = time.time() - start_time
                remaining_epochs = max_epoch - epoch
                eta = epoch_duration * remaining_epochs
                eta_sec = int(eta)
                eta_h, rem = divmod(eta_sec, 3600)
                eta_m, eta_s = divmod(rem, 60)
                self.logger.info(
                    f"Epoch {epoch} duration: {epoch_duration:.2f} sec | ETA: {eta_h:02d}:{eta_m:02d}:{eta_s:02d}"
                )

                best_train_loss, best_val_loss, best_wa, best_models, saved_epochs = (
                    save_models(
                        self.exp_dir,
                        self.logger,
                        self.model,
                        self.conf,
                        self.device,
                        epoch,
                        max_epoch,
                        avg_train_loss,
                        avg_val_loss,
                        wa_score,
                        best_train_loss,
                        best_val_loss,
                        best_wa,
                        best_models,
                        saved_epochs,
                        n_best,
                    )
                )
                plot_metrics(image_dir, train_losses, val_losses, val_cers, val_accs)
                self.logger.info(
                    f"Saved [loss, wa, cer] curves at {image_dir.as_posix()}"
                )
        finally:
            self.tb_writer.close()

        total_time = time.time() - start_time
        h, rem = divmod(int(total_time), 60)
        m, s = divmod(rem, 60)
        self.logger.info(f"Training completed in {h:02d}:{m:02d}:{s:02d}")
