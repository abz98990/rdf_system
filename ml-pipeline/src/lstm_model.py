"""Phase 3b: the LSTM deep learning model — multi-variate sequence model
with dropout regularisation, per the "Deep Learning Network" component
described in the architecture diagram. Implemented in PyTorch.
"""
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from config import LOOKBACK_DAYS, LSTM_MAX_SERIES, MODELS_DIR, RANDOM_SEED
from src.sequence_utils import build_window_index, group_series_arrays, split_window_index

torch.manual_seed(RANDOM_SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class WindowDataset(Dataset):
    """Lazily slices out one lookback window per __getitem__ instead of
    materialising all of them up front — the difference between O(n_rows)
    and O(n_windows * lookback) memory. See sequence_utils module docstring.
    """

    def __init__(self, arrays: dict, index: list, lookback: int = LOOKBACK_DAYS):
        self.arrays = arrays
        self.index = index
        self.lookback = lookback

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        id_, start, _ = self.index[i]
        features, target, _ = self.arrays[id_]
        X = features[start: start + self.lookback]
        y = target[start + self.lookback]
        return torch.from_numpy(X), torch.tensor(y, dtype=torch.float32)


class DemandLSTM(nn.Module):
    def __init__(self, n_features: int, hidden_size: int = 64, dropout: float = 0.2):
        super().__init__()
        self.lstm1 = nn.LSTM(n_features, hidden_size, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.lstm2 = nn.LSTM(hidden_size, hidden_size // 2, batch_first=True)
        self.dropout2 = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden_size // 2, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        out, _ = self.lstm1(x)
        out = self.dropout1(out)
        out, (h_n, _) = self.lstm2(out)
        out = self.dropout2(h_n[-1])
        return self.head(out).squeeze(-1)


def build_lstm_model(n_features: int) -> DemandLSTM:
    return DemandLSTM(n_features).to(DEVICE)


def subsample_series(arrays: dict, max_series: int | None, seed: int = RANDOM_SEED) -> dict:
    if max_series is None or len(arrays) <= max_series:
        return arrays
    rng = np.random.default_rng(seed)
    keep = rng.choice(sorted(arrays.keys()), size=max_series, replace=False)
    return {k: arrays[k] for k in keep}


def prepare_datasets(df: pd.DataFrame, max_series: int | None = LSTM_MAX_SERIES):
    arrays = group_series_arrays(df)
    arrays = subsample_series(arrays, max_series)
    index = build_window_index(arrays)
    train_idx, val_idx, test_idx = split_window_index(index)
    return (
        WindowDataset(arrays, train_idx),
        WindowDataset(arrays, val_idx),
        WindowDataset(arrays, test_idx),
        len(arrays),
    )


def train_lstm(train_ds: WindowDataset, val_ds: WindowDataset, epochs: int = 20, batch_size: int = 256, patience: int = 3):
    n_features = next(iter(train_ds.arrays.values()))[0].shape[1]
    model = build_lstm_model(n_features=n_features)
    optimizer = torch.optim.Adam(model.parameters())
    loss_fn = nn.L1Loss()  # MAE, matching the metric used to compare models

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(epochs):
        model.train()
        train_losses = []
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()
            preds = model(X_batch)
            loss = loss_fn(preds, y_batch)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                preds = model(X_batch)
                val_losses.append(loss_fn(preds, y_batch).item())

        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        print(f"epoch {epoch + 1}/{epochs} - train_loss: {train_loss:.4f} - val_loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping at epoch {epoch + 1} (no improvement for {patience} epochs).")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


def predict_lstm(model: DemandLSTM, X: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        preds = model(torch.from_numpy(X).to(DEVICE))
    return preds.cpu().numpy()


def predict_dataset(model: DemandLSTM, dataset: WindowDataset, batch_size: int = 256):
    """Run predictions over a whole WindowDataset, returning (y_true, y_pred)
    without materialising every window into one array at once.
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            preds = model(X_batch.to(DEVICE))
            y_true.append(y_batch.numpy())
            y_pred.append(preds.cpu().numpy())
    return np.concatenate(y_true), np.concatenate(y_pred)


def save_lstm(model: DemandLSTM, path):
    torch.save({"state_dict": model.state_dict(), "n_features": model.lstm1.input_size}, path)


def load_lstm(path) -> DemandLSTM:
    checkpoint = torch.load(path, map_location=DEVICE, weights_only=True)
    model = build_lstm_model(n_features=checkpoint["n_features"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


if __name__ == "__main__":
    from config import DATA_PROCESSED_DIR
    from src.evaluate import evaluate_predictions

    featured = pd.read_parquet(DATA_PROCESSED_DIR / "featured.parquet")
    train_ds, val_ds, test_ds, n_series = prepare_datasets(featured)
    print(f"LSTM training on {n_series} series: train={len(train_ds):,} val={len(val_ds):,} test={len(test_ds):,} windows")

    model, _ = train_lstm(train_ds, val_ds)

    y_val, preds = predict_dataset(model, val_ds)
    metrics = evaluate_predictions(y_val, preds)

    print("LSTM validation metrics:")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    save_lstm(model, MODELS_DIR / "lstm_model.pt")
    print(f"Saved model to {MODELS_DIR / 'lstm_model.pt'}")
