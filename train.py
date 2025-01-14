import torch
import torch.nn as nn
from transformer_new import DecoderOnlyTransformer, Config
from dataloader import train_dataloader, test_dataloader
from tqdm import tqdm
import wandb
import os


def train():
    # Initialize wandb
    wandb.init(project="shakespeare-transformer")

    # Create checkpoint directory if it doesn't exist
    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Model initialization
    config = Config()
    model = DecoderOnlyTransformer(config).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    # Training loop
    num_epochs = 20
    best_loss = float("inf")

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")

        for batch_idx, (inputs, targets) in enumerate(progress_bar):
            # Move data to device
            inputs = inputs.squeeze(1).to(device)  # Remove the extra dimension
            targets = targets.squeeze(1).to(device)  # Remove the extra dimension

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs.view(-1, config.vocab_size), targets.view(-1))

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # Update progress bar
            total_loss += loss.item()
            avg_loss = total_loss / (batch_idx + 1)
            progress_bar.set_postfix({"loss": f"{avg_loss:.4f}"})

            # Log to wandb
            wandb.log({"train_loss": loss.item(), "epoch": epoch, "batch": batch_idx})

        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, targets in test_dataloader:
                inputs = inputs.squeeze(1).to(device)
                targets = targets.squeeze(1).to(device)
                outputs = model(inputs)
                loss = criterion(outputs.view(-1, config.vocab_size), targets.view(-1))
                val_loss += loss.item()

        val_loss /= len(test_dataloader)
        print(
            f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_loss:.4f}, Val Loss: {val_loss:.4f}"
        )

        # Log validation metrics
        wandb.log({"val_loss": val_loss, "epoch": epoch})

        # Save checkpoint for every epoch
        checkpoint_path = os.path.join(
            checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pth"
        )
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": avg_loss,
                "val_loss": val_loss,
            },
            checkpoint_path,
        )

        # Save best model separately
        if val_loss < best_loss:
            best_loss = val_loss
            best_model_path = os.path.join(checkpoint_dir, "best_model.pth")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": best_loss,
                },
                best_model_path,
            )


if __name__ == "__main__":
    train()
