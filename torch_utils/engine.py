"""
Contains training and evaluation loop functionality for PyTorch models."""
import torch
import json
from tqdm.auto import tqdm
from pathlib import Path


# ANSI COLORS
BLUE = "\033[94m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
RESET = "\033[0m"


def save_results(results: dict, save_path: str):
    """Helper to save results dict to a JSON file."""
    # Change extension from .pth to .json
    json_path = Path(save_path).with_suffix(".json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=4)


def train_step(model: torch.nn.Module, 
               dataloader: torch.utils.data.DataLoader, 
               loss_fn: torch.nn.Module, 
               optimizer: torch.optim.Optimizer, 
               accuracy_fn, 
               device: str,
               epoch_index: int,
               total_epochs: int):
    """
    Performs one epoch of training.
    """
    model.train()
    train_loss, train_acc = 0, 0
    
    progress_bar = tqdm(dataloader, desc=f"Epoch [{epoch_index+1}/{total_epochs}]")
    
    for batch_idx, (X, y) in enumerate(progress_bar, 1):
        X, y = X.to(device), y.to(device)

        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        acc = accuracy_fn(y, y_pred.argmax(dim=1))

        train_loss += loss.item()
        train_acc += acc

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss = train_loss / batch_idx
        running_acc = train_acc / batch_idx
        
        progress_bar.set_postfix(loss=f"{running_loss:.4f}", acc=f"{running_acc:.2f}%")

    return train_loss / len(dataloader), train_acc / len(dataloader)


def test_step(model: torch.nn.Module, 
              dataloader: torch.utils.data.DataLoader, 
              loss_fn: torch.nn.Module, 
              accuracy_fn, 
              device: str):
    """
    Performs evaluation on validation data.
    """
    model.eval()
    val_loss, val_acc = 0, 0
    
    with torch.inference_mode():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            
            y_pred = model(X)
            
            val_loss += loss_fn(y_pred, y).item()
            val_acc += accuracy_fn(y, y_pred.argmax(dim=1))

    return val_loss / len(dataloader), val_acc / len(dataloader)


def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          val_dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          accuracy_fn,
          device: str,
          epochs: int,
          scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
          save_path: str = "models/best_model.pth") -> dict:
    
    
    
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    results = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_loss = float('inf') 

    try:
        for epoch in range(epochs):
            
            # --- Modular Steps ---
            train_loss, train_acc = train_step(
                model=model, 
                dataloader=train_dataloader, 
                loss_fn=loss_fn, 
                optimizer=optimizer, 
                accuracy_fn=accuracy_fn, 
                device=device,
                epoch_index=epoch,
                total_epochs=epochs
            )
            
            val_loss, val_acc = test_step(
                model=model, 
                dataloader=val_dataloader, 
                loss_fn=loss_fn, 
                accuracy_fn=accuracy_fn, 
                device=device
            )

            # --- Logging ---
            results["train_loss"].append(train_loss)
            results["train_acc"].append(train_acc)
            results["val_loss"].append(val_loss)
            results["val_acc"].append(val_acc)

            print(f"{CYAN}Train Loss:{RESET} {YELLOW}{train_loss:.4f}{RESET} | "
                  f"{CYAN}Val Loss:{RESET} {YELLOW}{val_loss:.4f}{RESET} | "
                  f"{CYAN}Train Acc:{RESET} {GREEN}{train_acc:.2f}%{RESET} | "
                  f"{CYAN}Val Acc:{RESET} {GREEN}{val_acc:.2f}%{RESET}")

            # --- Checkpointing ---
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                # Save Model
                torch.save(model.state_dict(), save_path)
                # Save Results (JSON)
                save_results(results, save_path)
                print(f"{GREEN}>>> Improved validation loss. Saved model to {save_path}{RESET}")
            
            print() 

            # --- Scheduler ---
            if scheduler:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_loss)
                else:
                    scheduler.step()

    except KeyboardInterrupt:
        print(f"\n{YELLOW}Training interrupted by user!{RESET}")
        print(f"{YELLOW}Saving current state...{RESET}")
        
        Path("models/interrupted_model.pth").parent.mkdir(parents=True, exist_ok=True)
        # Save interrupted work
        torch.save(model.state_dict(), "models/interrupted_model.pth")
        save_results(results, "models/interrupted_model.pth")
        
        print(f"{GREEN}Safe exit performed. Returning logs.{RESET}")

    return results


def eval_model(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               accuracy_fn,
               device: str) -> dict:
    """
    Evaluates a trained model on a test dataset.
    
    Returns:
        dict: Dictionary containing 'loss' and 'accuracy' metrics
    """
    model.eval()
    test_loss, test_acc = 0, 0
    
    print(f"{CYAN}Evaluating model...{RESET}")
    
    with torch.inference_mode():
        for X, y in tqdm(dataloader, desc="Testing"):
            X, y = X.to(device), y.to(device)
            
            y_pred = model(X)
            
            test_loss += loss_fn(y_pred, y).item()
            test_acc += accuracy_fn(y, y_pred.argmax(dim=1))
    
    test_loss /= len(dataloader)
    test_acc /= len(dataloader)
    
    print(f"\n{CYAN}Test Loss:{RESET} {YELLOW}{test_loss:.4f}{RESET} | "
          f"{CYAN}Test Acc:{RESET} {GREEN}{test_acc:.2f}%{RESET}\n")
    
    return {"loss": test_loss, "accuracy": test_acc}