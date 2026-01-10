"""
This file contains various utility functions.
"""
import math
import random
import torch
from torchvision import transforms
from pathlib import Path
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.patches import Rectangle
from typing import List, Tuple, Dict


def accuracy_fn(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """Calculates accuracy between true and predicted labels.

    Args:
        y_true (torch.Tensor): True class labels.
        y_pred (torch.Tensor): Predicted class labels (usually the output of argmax).

    Returns:
        float: Accuracy percentage.
    """
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc


def save_model(model: torch.nn.Module,
               target_dir: str,
               model_name: str):
    """Saves a PyTorch model to a target directory."""

    # Create target directory if it doesn't exist
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True, exist_ok=True)

    # Create the model save path
    model_save_path = target_dir_path / model_name

    # Save the model state dict
    torch.save(obj=model.state_dict(),
               f=model_save_path)
    
    print(f"Model saved to: {model_save_path}")


def load_model(model: torch.nn.Module,
                model_path: str,
                num_classes: int,
                device: str):
    """Loads a PyTorch model from a target path."""
     
    model_path = Path(model_path) # type: ignore
    
    if model_path.is_file(): #type: ignore
          print(f"Loading model from: {model_path}")
          model.load_state_dict(torch.load(f=model_path,
                                          map_location=device))
          model.to(device)
          print("Model loaded successfully!")
          
    else:
        print(f"Model path: {model_path} does not exist!")

    return model


import matplotlib.pyplot as plt
from typing import Dict, List, Optional



def plot_loss_curves(results: Dict[str, List[float]], save_path: Optional[str] = None):
    """
    Plots training curves of a model.
    
    Args:
        results: Dictionary containing list of values, e.g.:
            {'train_loss': [...], 'train_acc': [...], 'val_loss': [...], 'val_acc': [...]}
        save_path: Optional string path to save the figure (e.g., 'plots/results.png')
    """
    
    # Get the loss values of the results dictionary (training and validation)
    loss = results['train_loss']
    test_loss = results['val_loss']

    # Get the accuracy values of the results dictionary (training and validation)
    accuracy = results['train_acc']
    test_accuracy = results['val_acc']

    # Figure out how many epochs there were (start at 1 for the graph)
    epochs = range(1, len(results['train_loss']) + 1)

    # Setup a plot 
    plt.figure(figsize=(15, 7))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label='Train Loss', marker='o') # Markers help see individual epochs
    plt.plot(epochs, test_loss, label='Val Loss', marker='o')
    plt.title('Loss', fontsize=14, fontweight='bold')
    plt.xlabel('Epochs')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, label='Train Accuracy', marker='o')
    plt.plot(epochs, test_accuracy, label='Val Accuracy', marker='o')
    plt.title('Accuracy', fontsize=14, fontweight='bold')
    plt.xlabel('Epochs')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
        
    plt.show()


def make_predictions_grid(test_dir : str,
                           model: torch.nn.Module, 
                           transform: transforms.Compose, 
                           class_names: List[str], 
                           device: str, 
                           n_images=9):
    """
    Visualizes random model predictions on a grid of images from the test directory.
    
    This function samples images from class subfolders, runs inference, and displays
    a grid showing the image, predicted label, confidence score, and true label.
    Incorrect predictions are highlighted with a red overlay.

    :param test_dir: Path to the root test directory containing class subfolders.
    :param model: The trained PyTorch model in evaluation mode.
    :param transform: The torchvision transform pipeline (must match training input).
    :param class_names: List of class strings where index matches model output.
    :param device: The torch device to run inference on (e.g., 'cuda' or 'cpu').
    :param n_images: Number of images to randomly sample and display (default: 9).
    """
    image_path = Path(test_dir)
    
    # improvement: grab multiple extensions
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_paths_list = []
    for ext in extensions:
        image_paths_list.extend(list(image_path.glob(f"*/{ext}")))

    if not image_paths_list:
        raise ValueError(f"No images found in {test_dir} (checked {extensions})")

    rng = random.Random()
    selected_images = rng.sample(image_paths_list, k=min(n_images, len(image_paths_list)))

    model.eval()
    
    cols = 3
    rows = int(math.ceil(len(selected_images) / cols))
    
    # Slightly larger figure to accommodate titles
    plt.figure(figsize=(4 * cols, 4 * rows))

    with torch.no_grad():
        for idx, img_path in enumerate(selected_images):
            # Open image
            image = Image.open(img_path).convert("RGB")
            
            # transform for model (handles resizing/cropping for the logic)
            image_tensor = transform(image).unsqueeze(0).to(device) # type: ignore

            # Inference
            logits = model(image_tensor)
            probs = F.softmax(logits, dim=1)
            pred_idx = torch.argmax(probs, dim=1).item()
            pred_class = class_names[pred_idx] # type: ignore
            confidence = probs[0][pred_idx].item() #type: ignore

            true_class = img_path.parent.name
            is_correct = (pred_class == true_class)

            # Plotting
            ax = plt.subplot(rows, cols, idx + 1)
            
            # improvement: Display original aspect ratio, don't distort it
            ax.imshow(image) 
            ax.axis("off")

            # Dynamic Title Styling
            title_color = "green" if is_correct else "red"
            title_text = (f"True: {true_class}\n"
                          f"Pred: {pred_class} ({confidence:.1%})")
            
            ax.set_title(title_text, color=title_color, fontsize=10, fontweight='bold')

            # The "Confidently Wrong" Overlay

            if not is_correct:
                alpha = min(0.4, max(0.1, confidence))
                # Get current axis limits to ensure mask covers whole image regardless of size
                xlim = ax.get_xlim()
                ylim = ax.get_ylim()
                
                mask = Rectangle(
                    (xlim[0], ylim[1]), # Anchored top-left
                    xlim[1] - xlim[0], 
                    ylim[0] - ylim[1],
                    linewidth=0,
                    facecolor=(1, 0, 0, alpha)
                )
                ax.add_patch(mask)

    plt.tight_layout()
    plt.show()


def predict_single_image(image_path: str, 
                         model: torch.nn.Module, 
                         transform: transforms.Compose, 
                         class_names: List[str], 
                         device: str):
    img_path = Path(image_path)
    if not img_path.exists():
        raise FileNotFoundError(f"Image not found at {img_path}")

    model.eval()

    # Load and Transform
    image = Image.open(img_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device) # type: ignore

    # Inference
    with torch.no_grad():
        logits = model(image_tensor)
        probs = F.softmax(logits, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()
        pred_class = class_names[pred_idx] # type: ignore
        confidence = probs[0][pred_idx].item() # type: ignore

    # Get True Label (assuming parent folder name is label)
    true_class = img_path.parent.name
    is_correct = (pred_class == true_class)
    
    # Plotting
    plt.figure(figsize=(5, 5)) 
    plt.imshow(image)
    plt.axis("off")

    # Text Styling
    title_color = "green" if is_correct else "red"
    info_text = (f"True: {true_class}\n"
                 f"Pred: {pred_class}\n"
                 f"Conf: {confidence:.2%}")

    plt.title(info_text, color=title_color, fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.show()