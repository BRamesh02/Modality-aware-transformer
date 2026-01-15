import torch
from tqdm import tqdm

def train_epoch(model, model_config:dict, dataloader, optimizer, criterion, device, scheduler=None):
    """
    Generic training loop optimized for A100 (BF16 Mixed Precision).
    """
    model.train()
    total_loss = 0.0
    
    use_amp = (device == 'cuda')
    dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16

    progress_bar = tqdm(dataloader, desc="Training", leave=False)

    for batch in progress_bar:
        x_num  = batch["x_num"].to(device, non_blocking=model_config["pin_memory"])
        x_sent = batch["x_sent"].to(device, non_blocking=model_config["pin_memory"])
        y_hist = batch["y_hist"].to(device, non_blocking=model_config["pin_memory"])
        y_target = batch["y_future"].to(device, non_blocking=model_config["pin_memory"])

        x_emb = batch["x_emb"]
        if x_emb is not None:
            x_emb = x_emb.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type='cuda', dtype=dtype, enabled=use_amp):
            preds = model(x_num, x_sent, x_emb, y_hist)
            loss = criterion(preds, y_target)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        if scheduler is not None:
            scheduler.step()
        
        total_loss += loss.item()
        progress_bar.set_postfix({"loss": f"{loss.item():.5f}"})

    return total_loss / len(dataloader)


def validate_epoch(model, model_config:dict, dataloader, criterion, device):
    """
    Validation loop using Mixed Precision for faster inference.
    """
    model.eval()
    total_loss = 0.0
    
    use_amp = (device == 'cuda')
    dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16

    with torch.no_grad():
        for batch in dataloader:
            x_num  = batch["x_num"].to(device, non_blocking=model_config["pin_memory"])
            x_sent = batch["x_sent"].to(device, non_blocking=model_config["pin_memory"])
            y_hist = batch["y_hist"].to(device, non_blocking=model_config["pin_memory"])
            y_target = batch["y_future"].to(device, non_blocking=model_config["pin_memory"])

            x_emb = batch["x_emb"]
            if x_emb is not None:
                x_emb = x_emb.to(device, non_blocking=True)

            with torch.autocast(device_type='cuda', dtype=dtype, enabled=use_amp):
                preds = model(x_num, x_sent, x_emb, y_hist)
                loss = criterion(preds, y_target)

            total_loss += loss.item()

    return total_loss / len(dataloader)