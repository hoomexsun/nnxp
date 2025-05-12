from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns


def plot_metrics(image_dir: Path, train_losses, val_losses, val_cers, val_accs):
    image_dir.mkdir(parents=True, exist_ok=True)

    sns.set(style="whitegrid", font_scale=1.4)
    palette = sns.color_palette("Set2")

    # Loss plot
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Train Loss", marker='o', color=palette[0])
    plt.plot(val_losses, label="Val Loss", marker='s', color=palette[1])
    plt.xlabel("Epoch", fontsize=14)
    plt.ylabel("Loss", fontsize=14)
    plt.title("Training and Validation Loss", fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(image_dir / "losses.png", dpi=300)
    plt.close()

    # CER plot
    plt.figure(figsize=(10, 6))
    plt.plot(val_cers, label="Val CER", marker='^', color=palette[2])
    plt.xlabel("Epoch", fontsize=14)
    plt.ylabel("Character Error Rate", fontsize=14)
    plt.title("Validation CER Over Epochs", fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(image_dir / "cer.png", dpi=300)
    plt.close()

    # Word Accuracy plot
    plt.figure(figsize=(10, 6))
    plt.plot(val_accs, label="Val Word Accuracy", marker='D', color=palette[3])
    plt.xlabel("Epoch", fontsize=14)
    plt.ylabel("Word Accuracy", fontsize=14)
    plt.title("Validation Word Accuracy Over Epochs", fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(image_dir / "wa.png", dpi=300)
    plt.close()
