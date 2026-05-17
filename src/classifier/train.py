import yaml
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from src.utils import paths
import torch
from src.classifier.dataset import build_loader, get_subsets
from src.classifier.model import build_model
from src.classifier.transforms import train_transform, eval_transform
from torch.nn import Module
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

def train_one_epoch(model: Module, loader: DataLoader, optimizer: AdamW, criterion: CrossEntropyLoss, device: torch.device, epoch_idx: int) -> tuple[float, float]:
    model.train()

    loss_sum = 0.0
    acc_sum = 0.0

    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item()
        acc_sum += (output.argmax(dim=1) == target).sum().item()

    return loss_sum / len(loader), acc_sum / len(loader.dataset)


def validate(model: Module, loader: DataLoader, criterion: CrossEntropyLoss, device: torch.device) -> tuple[float, float]:
    model.eval()
    loss_sum = 0.0
    acc_sum = 0.0

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            loss_sum += loss.item()
            acc_sum += (output.argmax(dim=1) == target).sum().item()

    return loss_sum / len(loader), acc_sum / len(loader.dataset)


def save_checkpoint(path, model: Module, optimizer: AdamW, scheduler: CosineAnnealingLR, epoch: int, val_acc: float) -> None:
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
        "val_acc": val_acc
    }
    torch.save(state, path)


def train(config: dict):
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    torch.manual_seed(config["seed"])
    checkpoint_dir = paths.PROJECT_ROOT / config["model_name"] / config["checkpoint_dir"]
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    train_data, val_data, _ = get_subsets(train_transformer=train_transform(), eval_transformer=eval_transform(), seed=config["seed"])
    train_loader = build_loader(train_data, batch_size=config["batch_size"], shuffle=True)
    val_loader = build_loader(val_data, batch_size=config["batch_size"], shuffle=False)
    model = build_model(config['model_name'], config['num_classes'])
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    criterion = CrossEntropyLoss()
    scheduler = CosineAnnealingLR(optimizer, T_max=config["epochs"])

    best_val_acc = 0.0
    for epoch_idx in tqdm(range(1, config["epochs"] + 1), desc="Epoch"):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch_idx)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step()

        print(f"Epoch {epoch_idx}: train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(checkpoint_dir / "best_model.pth", model, optimizer, scheduler, epoch_idx, val_acc)



if __name__ == "__main__":
    with open(paths.CONFIGS_DIR / "train_config.yaml") as f:
        config = yaml.safe_load(f)
    train(config)