"""Training entry point for the FT-Transformer package."""

from .trainer import Trainer


def train_ft_transformer():
    """Train the FT-Transformer model and return the trainer instance."""
    trainer = Trainer()
    trainer.run()
    return trainer


def main():
    train_ft_transformer()


if __name__ == "__main__":
    main()
