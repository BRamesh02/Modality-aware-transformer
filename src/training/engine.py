import torch
from tqdm import tqdm

def train_epoch(model, dataloader, optimizer, criterion, device):
    """
    Generic training loop for both Canonical and MAT models.
    """
    model.train()
    total_loss = 0
    
    progress_bar = tqdm(dataloader, desc="Training", leave=False)
    
    for batch in progress_bar:
        x_num  = batch['x_num'].to(device)
        x_sent = batch['x_sent'].to(device)
        x_emb  = batch['x_emb'].to(device)
        
        y_hist = batch['y_hist'].to(device)
        y_target = batch['y_future'].to(device)

        optimizer.zero_grad()
        
        preds = model(x_num, x_sent, x_emb, y_hist)
        
        loss = criterion(preds, y_target)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        progress_bar.set_postfix({'loss': f"{loss.item():.5f}"})
        
    return total_loss / len(dataloader)


def validate_epoch(model, dataloader, criterion, device):
    """
    Generic validation loop (No Gradient Calculation).
    """
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for batch in dataloader:
            x_num  = batch['x_num'].to(device)
            x_sent = batch['x_sent'].to(device)
            x_emb  = batch['x_emb'].to(device)
            y_hist = batch['y_hist'].to(device)
            y_target = batch['y_future'].to(device)

            preds = model(x_num, x_sent, x_emb, y_hist)
            loss = criterion(preds, y_target)
            total_loss += loss.item()
            
    return total_loss / len(dataloader)