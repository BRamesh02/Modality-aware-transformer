import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.models.dataset import FinancialDataset
from src.models.architectures.canonical_transformer import TransformerEDBaselineAR  


def train_one_epoch(model, loader, optimizer, criterion, device, grad_clip=None):
    model.train()
    total_loss, n = 0.0, 0

    for batch in loader:
        x_num  = batch["x_num"].to(device)
        x_sent = batch["x_sent"].to(device)
        x_emb  = batch["x_emb"].to(device)
        y_hist = batch["y_hist"].to(device)
        y_true = batch["y_future"].to(device)

        optimizer.zero_grad(set_to_none=True)
        y_pred = model(x_num, x_sent, x_emb, y_hist)  # [B,H]
        loss = criterion(y_pred, y_true)
        loss.backward()

        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        bs = x_num.size(0)
        total_loss += loss.item() * bs
        n += bs

    return total_loss / max(n, 1)


@torch.no_grad()
def eval_one_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, n = 0.0, 0
    for batch in loader:
        x_num  = batch["x_num"].to(device)
        x_sent = batch["x_sent"].to(device)
        x_emb  = batch["x_emb"].to(device)
        y_hist = batch["y_hist"].to(device)
        y_true = batch["y_future"].to(device)

        y_pred = model(x_num, x_sent, x_emb, y_hist)
        loss = criterion(y_pred, y_true)

        bs = x_num.size(0)
        total_loss += loss.item() * bs
        n += bs

    return total_loss / max(n, 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--num_workers", type=int, default=4)

    p.add_argument("--window_size", type=int, default=60)
    p.add_argument("--H", type=int, default=5)
    p.add_argument("--min_date", type=str, default=None)
    p.add_argument("--max_date", type=str, default=None)

    p.add_argument("--num_input_dim", type=int, required=True)
    p.add_argument("--n_sent", type=int, default=5)
    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--nhead", type=int, default=4)
    p.add_argument("--enc_layers", type=int, default=2)
    p.add_argument("--dec_layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.2)

    p.add_argument("--save_dir", type=str, default="checkpoints/transformer_ar")
    args = p.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device(args.device)

    # --- TODO: tu dois charger tes df train/val ici ---
    raise NotImplementedError("Charge df_train/df_val ici.")

    train_ds = FinancialDataset(df_train, args.window_size, args.min_date, args.max_date, forecast_horizon=args.H)
    val_ds   = FinancialDataset(df_val,   args.window_size, args.min_date, args.max_date, forecast_horizon=args.H)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    model = TransformerEDBaselineAR(
        num_input_dim=args.num_input_dim,
        n_sent=args.n_sent,
        d_model=args.d_model,
        nhead=args.nhead,
        enc_layers=args.enc_layers,
        dec_layers=args.dec_layers,
        dropout=args.dropout,
        forecast_horizon=args.H,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        tr = train_one_epoch(model, train_loader, optimizer, criterion, device, grad_clip=args.grad_clip)
        va = eval_one_epoch(model, val_loader, criterion, device)
        print(f"[Transformer AR] epoch {epoch:03d} | train {tr:.6f} | val {va:.6f}")

        if va < best_val:
            best_val = va
            path = os.path.join(args.save_dir, "best.pt")
            torch.save({"model": model.state_dict(), "epoch": epoch, "val": va}, path)

    print("Done. Best val:", best_val)


if __name__ == "__main__":
    main()