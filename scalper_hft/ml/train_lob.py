"""Pipeline для навчання моделей на L2-даних (Order Book).

Очікує підготовлені DataFrame зі сліпками стакану, з яких генерує
тензори для навчання DeepLOB.
"""

import logging
import sys

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset

    from scalper_hft.ml.lob_models import DeepLOB

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

logger = logging.getLogger(__name__)


def train_deeplob(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 10,
    batch_size: int = 64,
    lr: float = 1e-3,
):
    """
    Тренування моделі DeepLOB на підготовлених тензорах.
    X має бути форми (N, 1, seq_len, 40)
    y має бути форми (N,) з мітками класів (напр. 0, 1, 2)
    """
    if not HAS_TORCH:
        logger.error("PyTorch не встановлено! Використайте `uv pip install torch`")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Використовуємо пристрій: %s", device)

    # Створюємо датасети
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Ініціалізуємо модель
    model = DeepLOB(num_classes=len(np.unique(y_train))).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    logger.info("Починаємо навчання DeepLOB...")
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        total_samples = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            train_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)

        avg_train_loss = train_loss / total_samples
        train_acc = train_correct / total_samples

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        total_val = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                val_correct += (predicted == labels).sum().item()
                total_val += labels.size(0)

        avg_val_loss = val_loss / total_val
        val_acc = val_correct / total_val

        logger.info(
            "Epoch %d/%d: Train Loss: %.4f | Train Acc: %.4f | Val Loss: %.4f | Val Acc: %.4f",
            epoch + 1,
            epochs,
            avg_train_loss,
            train_acc,
            avg_val_loss,
            val_acc,
        )

    return model


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("Модуль навчання LOB моделей. Інтегруйте його з CLI для виклику.")
