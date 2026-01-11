"""
Contains training and evaluation loop functionality for PyTorch models."""
import torch
import json
from tqdm.auto import tqdm
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

# ANSI COLORS
BLUE = "\033[94m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
RESET = "\033[0m"


def save_results(results: dict, save_path: str | Path):
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


from pathlib import Path
from datetime import datetime
import torch
from torch.utils.tensorboard import SummaryWriter


def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          val_dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          accuracy_fn,
          device: str,
          epochs: int,
          model_name: str,
          scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
          ) -> dict:

    # --- Directories ---
    exp_dir = Path("models/experiments")
    best_dir = Path("models/best_runs")
    interrupted_dir = Path("models/interrupted")
    results_dir = Path("results")

    exp_dir.mkdir(parents=True, exist_ok=True)
    best_dir.mkdir(parents=True, exist_ok=True)
    interrupted_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    results = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_loss = float('inf')
    best_val_acc = 0.0

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"{model_name.replace('.pth', '')}_{timestamp}"

    # --- TensorBoard ---
    log_dir = f"runs/{run_name}"
    writer = SummaryWriter(log_dir=log_dir)

    # Add model graph
    dummy_input = next(iter(train_dataloader))[0].to(device)
    writer.add_graph(model, dummy_input)

    try:
        for epoch in range(epochs):

            # --- Training ---
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

            # --- Validation ---
            val_loss, val_acc = test_step(
                model=model, 
                dataloader=val_dataloader, 
                loss_fn=loss_fn, 
                accuracy_fn=accuracy_fn, 
                device=device
            )

            # --- Store results ---
            results["train_loss"].append(train_loss)
            results["train_acc"].append(train_acc)
            results["val_loss"].append(val_loss)
            results["val_acc"].append(val_acc)

            # --- TensorBoard logging ---
            writer.add_scalars("Loss", {"Train": train_loss, "Validation": val_loss}, epoch)
            writer.add_scalars("Accuracy", {"Train": train_acc, "Validation": val_acc}, epoch)

            print(f"{CYAN}Train Loss:{RESET} {YELLOW}{train_loss:.4f}{RESET} | "
                  f"{CYAN}Val Loss:{RESET} {YELLOW}{val_loss:.4f}{RESET} | "
                  f"{CYAN}Train Acc:{RESET} {GREEN}{train_acc:.2f}%{RESET} | "
                  f"{CYAN}Val Acc:{RESET} {GREEN}{val_acc:.2f}%{RESET}")

            # --- Best Model Checkpointing ---
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_acc = val_acc

                best_path = best_dir / f"{run_name}.pth"
                torch.save(model.state_dict(), best_path)

                print(f"{GREEN}>>> Improved validation loss. Best model updated at {best_path}{RESET}")

            print()

            # --- Scheduler ---
            if scheduler:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_loss)
                else:
                    scheduler.step()

        # --- Save Final Experiment ---
        exp_path = exp_dir / f"{run_name}.pth"
        results_path = results_dir / f"{run_name}.json"
        
        torch.save(model.state_dict(), exp_path)
        save_results(results, results_path)
        
        print(f"{GREEN}Training complete!{RESET}")
        print(f"{GREEN}Final model saved to {exp_path}{RESET}")
        print(f"{GREEN}Results saved to {results_path}{RESET}")
        print()

        # --- HParams Logging ---
        hparams_dict = {
            "lr": optimizer.param_groups[0]['lr'],
            "batch_size": train_dataloader.batch_size,
            "epochs": epochs,
            "optimizer": type(optimizer).__name__
        }

        metrics_dict = {
            "hparam/best_val_loss": best_val_loss,
            "hparam/best_val_acc": best_val_acc,
            "hparam/final_train_loss": results["train_loss"][-1],
            "hparam/final_val_loss": results["val_loss"][-1],
            "hparam/final_val_acc": results["val_acc"][-1]
        }

        writer.add_hparams(hparam_dict=hparams_dict, metric_dict=metrics_dict)
        writer.close()

    except KeyboardInterrupt:
        print(f"\n{YELLOW}Training interrupted by user!{RESET}")
        print(f"{YELLOW}Saving current state...{RESET}")

        interrupted_path = interrupted_dir / f"{run_name}.pth"
        interrupted_results_path = results_dir / f"{run_name}_interrupted.json"

        torch.save(model.state_dict(), interrupted_path)
        save_results(results, interrupted_results_path)

        print(f"{GREEN}Safe exit performed. Checkpoint saved to {interrupted_path}{RESET}")
        print(f"{GREEN}Results saved to {interrupted_results_path}{RESET}")
        
        writer.close()

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