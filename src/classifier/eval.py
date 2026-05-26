import json
import yaml
import torch
from torch.nn import Module, CrossEntropyLoss
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
from src.utils import paths
from src.classifier.dataset import build_loader, get_subsets
from src.classifier.model import build_model
from src.classifier.transforms import train_transform, eval_transform


def load_checkpoint(path: Path, model: Module, device: torch.device) -> dict:
    """
    Load model weights
    :param path: Checkpoint path   
    :param model: The model 
    :param device: The device to load the model on
    :return: 
    """
    state = torch.load(path, map_location=device)
    model.load_state_dict(state["model"])
    return state


def evaluate(model: Module, loader: DataLoader, device: torch.device) -> dict:
    """
    Run model over loader and collect classification metrics.
    :param model: The model
    :param loader: The dataloader
    :param device: Device model loaded on
    :return: Returns a dict of [loss, top1, top5, preds, targets]
    """
    
    model.eval()
    criterion = CrossEntropyLoss()

    loss_sum = 0.0
    top1_correct = 0
    top5_correct = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            loss_sum += loss.item()

            preds = output.argmax(dim=1)
            top1_correct += (preds == target).sum().item()

            topk = output.topk(5, dim=1).indices
            top5_correct += topk.eq(target.unsqueeze(1)).any(dim=1).sum().item()

            all_preds.append(preds.cpu())
            all_targets.append(target.cpu())
    n = len(loader.dataset)
    
    return {
        "loss": loss_sum / len(loader),
        "top1": top1_correct / n,
        "top5": top5_correct / n,
        "preds": torch.cat(all_preds),
        "targets": torch.cat(all_targets),
    }


def per_class_accuracy(preds: torch.Tensor, targets: torch.Tensor, num_classes: int) -> dict:
    """
    zPer-class top-1 accuracy. Classes not present in `targets` are omitted
    :param preds: Predictions
    :param targets: Targets
    :param num_classes: Numer of classes
    :return: 
    """
    result = {}
    for c in range(num_classes):
        mask = targets == c
        n = mask.sum().item()
        if n == 0:
            continue
        result[c] = (preds[mask] == c).float().mean().item()
    return result


def confusion_matrix(preds: torch.Tensor, targets: torch.Tensor, num_classes: int) -> torch.Tensor:
    """
    Create confusion matrix. Rows = true class, columns = predicted class.
    :param preds: Predictions
    :param targets: Targets
    :param num_classes: Number of classes
    :return: Confusion matrix as a torch.Tensor
    """
    indices = targets * num_classes + preds
    cm = torch.bincount(indices, minlength=num_classes ** 2).reshape(num_classes, num_classes)
    return cm


def save_results(results: dict, per_class: dict, cm: torch.Tensor, output_dir: Path) -> None:
    """
    Persist eval outputs to disk:
        metrics.json — loss, top1, top5, per_class accuracy
        raw.pt       — preds, targets, confusion_matrix as tensors (for later analysis)
        confusion_matrix.png — row-normalized heatmap
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = {
        "loss": results["loss"],
        "top1": results["top1"],
        "top5": results["top5"],
        "per_class": {str(k): v for k, v in per_class.items()},
    }
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    torch.save(
        {
            "preds": results["preds"],
            "targets": results["targets"],
            "confusion_matrix": cm,
        },
        output_dir / "raw.pt",
    )

    cm_float = cm.float()
    row_sums = cm_float.sum(dim=1, keepdim=True).clamp(min=1)
    cm_norm = cm_float / row_sums

    fig, ax = plt.subplots(figsize=(20, 20))
    im = ax.imshow(cm_norm.numpy(), cmap="Blues", vmin=0, vmax=1)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title("Confusion matrix (row-normalized)")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=100)
    plt.close()


def evaluate_split(name: str, model: Module, loader: DataLoader, device: torch.device, num_classes: int, output_dir: Path) -> dict:
    """
    Run the full eval pipeline on a single loader and persist results.
    :param name: Model name
    :param model: The model
    :param loader: The dataloader
    :param device: Deivce everything is loaded on
    :param num_classes: Number of classes
    :param output_dir: The output for metrics and confusion matrix
    :return: Returns the metrics dict (without preds/targets — those are saved to disk).
    """
    print(f"Evaluating {name}: {len(loader.dataset)} samples")
    results = evaluate(model, loader, device)
    pc = per_class_accuracy(results["preds"], results["targets"], num_classes)
    cm = confusion_matrix(results["preds"], results["targets"], num_classes)
    save_results(results, pc, cm, output_dir)
    print(f"{name}: loss={results['loss']:.4f}  top1={results['top1']:.4f}  top5={results['top5']:.4f}")
    return {"loss": results["loss"], "top1": results["top1"], "top5": results["top5"]}


def evaluate_classifier(config: dict) -> None:
    """
    Top-level eval entry point: load checkpoint, run on synthetic test split, run on
    real-world set if present, save per-split results, print summary + domain gap.
    """
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    torch.manual_seed(config["seed"])

    model = build_model(config["model_name"], config["num_classes"])
    model.to(device)

    checkpoint_path = (paths.PROJECT_ROOT / config["model_name"] /
                       config["checkpoint_dir"] / config["checkpoint_name"])
    state = load_checkpoint(checkpoint_path, model, device)
    print(f"Loaded {checkpoint_path.name}: epoch={state.get('epoch')} val_acc={state.get('val_acc'):.4f}")

    output_root = paths.PROJECT_ROOT / config["model_name"] / "eval_results"
    output_root.mkdir(parents=True, exist_ok=True)

    _, _, test_data = get_subsets(
        train_transformer=train_transform(),
        eval_transformer=eval_transform(),
        test_size=config["test_fraction"],
        seed=config["seed"],
    )
    test_loader = build_loader(test_data, batch_size=config["batch_size"], shuffle=False)
    syn_metrics = evaluate_split(
        "synthetic-test", model, test_loader, device,
        config["num_classes"], output_root / "synthetic",
    )

    real_dir = paths.PROJECT_ROOT / config["real_eval_dir"]
    if real_dir.exists() and any(real_dir.iterdir()):
        real_data = ImageFolder(str(real_dir), transform=eval_transform())
        real_loader = build_loader(real_data, batch_size=config["batch_size"], shuffle=False)
        real_metrics = evaluate_split(
            "real-world", model, real_loader, device,
            config["num_classes"], output_root / "real",
        )
        gap = syn_metrics["top1"] - real_metrics["top1"]
        print(f"Domain gap (synthetic top1 - real top1): {gap:+.4f}")
    else:
        print(f"Skipping real-world eval (no data at {real_dir})")


if __name__ == "__main__":
    with open(paths.CONFIGS_DIR / "eval_config.yaml") as f:
        config = yaml.safe_load(f)
    evaluate_classifier(config)
