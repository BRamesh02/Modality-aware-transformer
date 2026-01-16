import torch
import numpy as np
from tqdm import tqdm
from torch.amp import GradScaler


def train_epoch(
    model, model_config: dict, dataloader, optimizer, criterion, device, scheduler=None
):
    """
    Training loop optimized for A100 with GradScaler for robust Mixed Precision.
    """
    model.train()
    total_loss = 0.0

    use_amp = device == "cuda"
    dtype = (
        torch.bfloat16
        if (use_amp and torch.cuda.is_bf16_supported())
        else torch.float16
    )
    scaler = GradScaler("cuda", enabled=(use_amp and dtype == torch.float16))

    progress_bar = tqdm(dataloader, desc="Training", leave=False)

    for batch in progress_bar:
        pin = model_config.get("pin_memory", False)

        x_num = batch["x_num"].to(device, non_blocking=pin)
        x_sent = batch["x_sent"].to(device, non_blocking=pin)
        y_hist = batch["y_hist"].to(device, non_blocking=pin)
        y_target = batch["y_future"].to(device, non_blocking=pin)

        x_emb = batch["x_emb"]
        if x_emb is not None:
            x_emb = x_emb.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type="cuda", dtype=dtype, enabled=use_amp):
            preds = model(x_num, x_sent, x_emb, y_hist)
            loss = criterion(preds, y_target)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        progress_bar.set_postfix({"loss": f"{loss.item():.5f}"})

    return total_loss / len(dataloader)


def validate_epoch(model, model_config: dict, dataloader, criterion, device):
    """
    Validation loop.
    Calculates IC only on the 'Actionable' horizons (e.g., first 5 days),
    ignoring the long-term noise for the sake of checkpointing.
    """
    model.eval()
    total_loss = 0.0

    focus_h = model_config.get("validation_horizon", 5)

    all_preds = []
    all_targets = []

    use_amp = device == "cuda"
    dtype = (
        torch.bfloat16
        if (use_amp and torch.cuda.is_bf16_supported())
        else torch.float16
    )

    with torch.no_grad():
        for batch in dataloader:
            pin = model_config.get("pin_memory", False)

            x_num = batch["x_num"].to(device, non_blocking=pin)
            x_sent = batch["x_sent"].to(device, non_blocking=pin)
            y_hist = batch["y_hist"].to(device, non_blocking=pin)
            y_target = batch["y_future"].to(device, non_blocking=pin)  # [Batch, 10]

            x_emb = batch["x_emb"]
            if x_emb is not None:
                x_emb = x_emb.to(device, non_blocking=True)

            with torch.autocast(device_type="cuda", dtype=dtype, enabled=use_amp):
                # Preds shape: [Batch, 10]
                preds = model(x_num, x_sent, x_emb, y_hist)
                loss = criterion(preds, y_target)

            total_loss += loss.item()

            p_slice = preds[:, :focus_h]
            t_slice = y_target[:, :focus_h]

            all_preds.append(p_slice.float().cpu().numpy().flatten())
            all_targets.append(t_slice.float().cpu().numpy().flatten())

    avg_loss = total_loss / len(dataloader)

    if len(all_preds) > 0:
        flat_preds = np.concatenate(all_preds)
        flat_targets = np.concatenate(all_targets)

        if len(flat_preds) > 1 and np.std(flat_preds) > 1e-6:
            val_ic = np.corrcoef(flat_preds, flat_targets)[0, 1]
        else:
            val_ic = 0.0
    else:
        val_ic = 0.0

    return avg_loss, val_ic
