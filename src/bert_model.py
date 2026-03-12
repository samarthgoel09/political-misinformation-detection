"""
DistilBERT fine-tuning module for multi-class text classification.
Uses HuggingFace transformers for tokenization and model training.
"""

import time
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW
from tqdm import tqdm


class TextDataset(Dataset):
    """PyTorch Dataset for text classification."""

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
    """DistilBERT-based text classifier."""

    def __init__(
        self,
        num_labels: int,
        model_name: str = "distilbert-base-uncased",
        max_length: int = 128,
        batch_size: int = 32,
        learning_rate: float = 2e-5,
        num_epochs: int = 3,
        device: str = None,
    ):
        self.num_labels = num_labels
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs

        # Auto-detect device
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
                print(f"Using GPU: {torch.cuda.get_device_name(0)}")
            else:
                self.device = torch.device("cpu")
                print("Using CPU (training will be slower)")
        else:
            self.device = torch.device(device)

        # Load tokenizer and model
        print(f"Loading {model_name}...")
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        self.model = DistilBertForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels
        )
        self.model.to(self.device)

    def _create_dataloader(self, texts, labels, shuffle=True):
        """Create a DataLoader from texts and labels."""
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

        Args:
            X_train: Array of training texts.
            y_train: Array of training labels (integers).
            X_val: Optional validation texts.
            y_val: Optional validation labels.

        Returns:
            Dictionary with training history.
        """
        train_loader = self._create_dataloader(X_train, y_train, shuffle=True)

        # Optimizer and scheduler
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=0.01,
        )
        total_steps = len(train_loader) * self.num_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=total_steps // 10,
            num_training_steps=total_steps,
        )

        history = {"train_loss": [], "val_loss": [], "val_accuracy": []}

        print(f"\nTraining DistilBERT for {self.num_epochs} epochs...")
        start_time = time.time()

        for epoch in range(self.num_epochs):
            # Training phase
            self.model.train()
            total_loss = 0
            progress = tqdm(
                train_loader,
                desc=f"Epoch {epoch + 1}/{self.num_epochs}",
                leave=True,
            )

            for batch in progress:
                optimizer.zero_grad()

                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["label"].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )

                loss = outputs.loss
                total_loss += loss.item()

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

                progress.set_postfix({"loss": f"{loss.item():.4f}"})

            avg_loss = total_loss / len(train_loader)
            history["train_loss"].append(avg_loss)
            print(f"  Epoch {epoch + 1} - Train Loss: {avg_loss:.4f}")

            # Validation phase
            if X_val is not None and y_val is not None:
                val_loss, val_acc = self._evaluate(X_val, y_val)
                history["val_loss"].append(val_loss)
                history["val_accuracy"].append(val_acc)
                print(f"  Epoch {epoch + 1} - Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        total_time = time.time() - start_time
        print(f"\nTraining completed in {total_time:.1f}s")
        history["train_time"] = total_time

        return history

    def _evaluate(self, X, y):
        """Evaluate the model on given data."""
        self.model.eval()
        loader = self._create_dataloader(X, y, shuffle=False)

        total_loss = 0
        correct = 0
        total = 0

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
                correct += (preds == labels).sum().item()
                total += len(labels)

        avg_loss = total_loss / len(loader)
        accuracy = correct / total
        return avg_loss, accuracy

    def predict(self, X):
        """
        Generate predictions for input texts.

        Args:
            X: Array of input texts.

        Returns:
            Array of predicted label IDs.
        """
        self.model.eval()
        # Create dummy labels for DataLoader
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
