"""
DistilBERT fine-tuning module for political misinformation classification.
Wraps HuggingFace's DistilBERT with early stopping, best-model checkpointing,
cosine learning rate scheduling, and class-weighted loss for imbalanced data.
"""

import copy
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
)
from torch.optim import AdamW
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score
from tqdm import tqdm


class TextDataset(Dataset):

    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "label": torch.tensor(label, dtype=torch.long),
        }


class BertClassifier:

    def __init__(
        self,
        num_labels: int,
        model_name: str = "distilbert-base-uncased",
        max_length: int = 128,
        batch_size: int = 32,
        learning_rate: float = 2e-5,
        num_epochs: int = 5,
        device: str = None,
        use_class_weights: bool = True,
        early_stopping_patience: int = 2,
        warmup_ratio: float = 0.1,
        lr_schedule: str = "cosine",
        gradient_accumulation_steps: int = 1,
    ):
        self.num_labels = num_labels
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.use_class_weights = use_class_weights
        self.early_stopping_patience = early_stopping_patience
        self.warmup_ratio = warmup_ratio
        self.lr_schedule = lr_schedule
        self.gradient_accumulation_steps = gradient_accumulation_steps

        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
                print(f"Using GPU: {torch.cuda.get_device_name(0)}")
            else:
                self.device = torch.device("cpu")
                print("Using CPU (training will be slower)")
        else:
            self.device = torch.device(device)

        print(f"Loading {model_name}...")
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        self.model = DistilBertForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels
        )
        self.model.to(self.device)

    def _create_dataloader(self, texts, labels, shuffle=True):
        dataset = TextDataset(
            texts, labels, self.tokenizer, self.max_length
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=0,
            pin_memory=True if self.device.type == "cuda" else False,
        )

    def train(self, X_train, y_train, X_val=None, y_val=None):
        """
        Fine-tune DistilBERT on the training data.
        New features vs. baseline:
          - Early stopping on validation macro-F1 (patience configurable)
          - Best-model checkpointing: loads the best epoch's weights at the end
          - Cosine or linear LR schedule (cosine default)
          - Configurable warmup ratio
          - Gradient accumulation for effective larger batch sizes
        Args:
            X_train: Array of training texts.
            y_train: Array of training labels (integers).
            X_val:   Optional validation texts (required for early stopping).
            y_val:   Optional validation labels.
        Returns:
            Dictionary with training history including per-epoch metrics.
        """

        train_loader = self._create_dataloader(X_train, y_train, shuffle=True)

        # Build weighted loss to counter class imbalance
        if self.use_class_weights:
            classes = np.arange(self.num_labels)
            weights = compute_class_weight("balanced", classes=classes, y=y_train)
            weight_tensor = torch.tensor(weights, dtype=torch.float).to(self.device)
            print(f"  Class weights: { {i: round(w, 3) for i, w in enumerate(weights)} }")
            loss_fn = nn.CrossEntropyLoss(weight=weight_tensor)
        else:
            loss_fn = nn.CrossEntropyLoss()

        optimizer = AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=0.01,
        )

        # Total optimizer steps accounting for gradient accumulation
        effective_steps_per_epoch = max(1, len(train_loader) // self.gradient_accumulation_steps)
        total_steps = effective_steps_per_epoch * self.num_epochs
        warmup_steps = int(total_steps * self.warmup_ratio)

        if self.lr_schedule == "cosine":
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps,
            )
        else:
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps,
            )

        history = {
            "train_loss": [],
            "val_loss": [],
            "val_accuracy": [],
            "val_f1_macro": [],
        }

        # Early stopping state
        best_val_f1 = -1.0
        best_epoch = 0
        best_model_weights = None
        patience_counter = 0
        use_early_stopping = (X_val is not None and y_val is not None
                               and self.early_stopping_patience > 0)

        print(f"\nTraining DistilBERT for up to {self.num_epochs} epochs "
              f"(LR schedule: {self.lr_schedule}, warmup: {warmup_steps} steps)")
        if use_early_stopping:
            print(f"Early stopping patience: {self.early_stopping_patience} epochs on val F1 macro")
        start_time = time.time()

        for epoch in range(self.num_epochs):
            self.model.train()
            total_loss = 0.0
            optimizer.zero_grad()

            progress = tqdm(
                enumerate(train_loader),
                total=len(train_loader),
                desc=f"Epoch {epoch + 1}/{self.num_epochs}",
                leave=True,
            )

            for step, batch in progress:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["label"].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )

                loss = loss_fn(outputs.logits, labels)

                # Scale loss for gradient accumulation
                if self.gradient_accumulation_steps > 1:
                    loss = loss / self.gradient_accumulation_steps

                total_loss += loss.item() * (self.gradient_accumulation_steps
                                              if self.gradient_accumulation_steps > 1 else 1)
                loss.backward()

                if (step + 1) % self.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                progress.set_postfix({"loss": f"{loss.item() * self.gradient_accumulation_steps:.4f}"})

            avg_loss = total_loss / len(train_loader)
            history["train_loss"].append(avg_loss)
            print(f"  Epoch {epoch + 1} — Train Loss: {avg_loss:.4f}")

            # Validation
            if X_val is not None and y_val is not None:
                val_loss, val_acc, val_f1 = self._evaluate(X_val, y_val)
                history["val_loss"].append(val_loss)
                history["val_accuracy"].append(val_acc)
                history["val_f1_macro"].append(val_f1)
                print(f"  Epoch {epoch + 1} — Val Loss: {val_loss:.4f} | "
                      f"Val Acc: {val_acc:.4f} | Val F1 (macro): {val_f1:.4f}")

                # Early stopping check on macro F1
                if use_early_stopping:
                    if val_f1 > best_val_f1:
                        best_val_f1 = val_f1
                        best_epoch = epoch + 1
                        # Deep copy model weights as the best checkpoint
                        best_model_weights = copy.deepcopy(self.model.state_dict())
                        patience_counter = 0
                        print(f"  ✓ New best val F1: {best_val_f1:.4f} (epoch {best_epoch})")
                    else:
                        patience_counter += 1
                        print(f"  No improvement. Patience: {patience_counter}/{self.early_stopping_patience}")
                        if patience_counter >= self.early_stopping_patience:
                            print(f"\n  Early stopping triggered after epoch {epoch + 1}. "
                                  f"Best epoch was {best_epoch} (F1={best_val_f1:.4f}).")
                            break

        total_time = time.time() - start_time
        print(f"\nTraining completed in {total_time:.1f}s")

        # Restore best model weights
        if best_model_weights is not None:
            self.model.load_state_dict(best_model_weights)
            print(f"Restored best model from epoch {best_epoch} (Val F1={best_val_f1:.4f})")

        history["train_time"] = total_time
        history["best_epoch"] = best_epoch if best_model_weights is not None else self.num_epochs
        history["best_val_f1"] = best_val_f1

        return history

    def _evaluate(self, X, y):
        self.model.eval()
        loader = self._create_dataloader(X, y, shuffle=False)

        total_loss = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["label"].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )

                total_loss += outputs.loss.item()
                preds = torch.argmax(outputs.logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / len(loader)
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        accuracy = (all_preds == all_labels).mean()
        val_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
        return avg_loss, accuracy, val_f1

    def predict(self, X):
        """
        Generate predictions for input texts.

        Args:
            X: Array of input texts.

        Returns:
            Array of predicted label IDs.
        """
        self.model.eval()
        dummy_labels = np.zeros(len(X), dtype=int)
        loader = self._create_dataloader(X, dummy_labels, shuffle=False)

        all_preds = []

        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )

                preds = torch.argmax(outputs.logits, dim=1)
                all_preds.extend(preds.cpu().numpy())

        return np.array(all_preds)

    def save(self, path: str):
        """Save the model and tokenizer."""
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        print(f"Model saved to {path}")

    @classmethod
    def load(cls, path: str, num_labels: int):
        """Load a saved model."""
        instance = cls.__new__(cls)
        instance.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        instance.tokenizer = DistilBertTokenizer.from_pretrained(path)
        instance.model = DistilBertForSequenceClassification.from_pretrained(
            path, num_labels=num_labels
        )
        instance.model.to(instance.device)
        instance.max_length = 128
        instance.batch_size = 32
        return instance