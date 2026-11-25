"""
File: trainer.py

Purpose:
Training pipeline for the FT-Transformer model.
"""

import logging

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from tqdm.auto import tqdm
from typing import cast, Union

import sys
from pathlib import Path as _Path
_PROJECT_ROOT = _Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
from src.utils.evaluation import evaluate_model

from .config import (
    DEVICE,
    EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    BETAS,
    EPS,
    RESUME_TRAINING
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

from .dataset import get_dataloaders
from .model import FTTransformer

from .losses import get_loss_function
from .scheduler import get_scheduler
from .metrics import calculate_metrics

from .checkpoint import ModelCheckpoint
from .early_stopping import EarlyStopping


class Trainer:
    """
    Trainer class for FT-Transformer.
    """

    def __init__(self):

        logger.info("%s", "=" * 60)
        logger.info("INITIALIZING TRAINER")
        logger.info("%s", "=" * 60)

        # -------------------------------------------------
        # Device
        # -------------------------------------------------

        self.device = DEVICE

        logger.info("Device : %s", self.device)

        # -------------------------------------------------
        # Data
        # -------------------------------------------------

        (
            self.train_loader,
            self.validation_loader,
            self.test_loader
        ) = get_dataloaders()

        # -------------------------------------------------
        # Model
        # -------------------------------------------------

        self.model = FTTransformer().to(self.device)

        # -------------------------------------------------
        # Loss
        # -------------------------------------------------

        self.criterion = get_loss_function()

        # -------------------------------------------------
        # Optimizer
        # -------------------------------------------------

        self.optimizer: optim.Optimizer = optim.AdamW(
            self.model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
            betas=BETAS,
            eps=EPS
        )

        # -------------------------------------------------
        # Scheduler
        # -------------------------------------------------

        self.scheduler = get_scheduler(
            self.optimizer
        )

        # -------------------------------------------------
        # Checkpoint
        # -------------------------------------------------

        self.checkpoint = ModelCheckpoint()

        # -------------------------------------------------
        # Early Stopping
        # -------------------------------------------------

        self.early_stopping = EarlyStopping()

        # -------------------------------------------------
        # Training State
        # -------------------------------------------------

        self.start_epoch = 0

        self.best_score = float("-inf")

        self.history = {
            "train_loss": [],
            "validation_loss": [],
            "train_accuracy": [],
            "validation_accuracy": [],
            "train_f1_score": [],
            "validation_f1_score": []
        }

        # -------------------------------------------------
        # Resume Training
        # -------------------------------------------------

        if RESUME_TRAINING:

            (
                self.model,
                optimizer,
                scheduler,
                self.start_epoch,
                self.best_score
            ) = self.checkpoint.load(
                self.model,
                self.optimizer,
                self.scheduler,
                device=str(self.device)
            )

            if optimizer is not None:
                self.optimizer = optimizer

            if scheduler is not None:
                self.scheduler = scheduler

        logger.info("\nTrainer initialized successfully.")
            # =====================================================
    # Train One Epoch
    # =====================================================

    def train_one_epoch(self):
        """
        Train the model for one epoch.

        Returns
        -------
        train_loss : float
            Average training loss.

        metrics : dict
            Training metrics.
        """

        self.model.train()

        running_loss = 0.0

        all_predictions = []

        all_labels = []

        # -------------------------------------------------
        # Iterate Through Training Data
        # -------------------------------------------------

        for features, labels in tqdm(
            self.train_loader,
            desc="Training",
            leave=False,
            unit="batch"
        ):

            # ---------------------------------------------
            # Move Data To Device
            # ---------------------------------------------

            features = features.to(self.device)

            labels = labels.to(self.device)

            # ---------------------------------------------
            # Reset Previous Gradients
            # ---------------------------------------------

            self.optimizer.zero_grad()

            # ---------------------------------------------
            # Forward Pass
            # ---------------------------------------------

            outputs = self.model(features)

            # ---------------------------------------------
            # Loss Calculation
            # ---------------------------------------------

            loss = self.criterion(
                outputs,
                labels
            )

            # ---------------------------------------------
            # Backpropagation
            # ---------------------------------------------

            loss.backward()

            # ---------------------------------------------
            # Gradient Clipping
            # ---------------------------------------------

            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=1.0
            )

            # ---------------------------------------------
            # Update Parameters
            # ---------------------------------------------

            self.optimizer.step()

            # ---------------------------------------------
            # Running Loss
            # ---------------------------------------------

            running_loss += loss.item()

            # ---------------------------------------------
            # Predictions
            # ---------------------------------------------

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            all_predictions.extend(

                predictions.detach()
                .cpu()
                .numpy()

            )

            all_labels.extend(

                labels.detach()
                .cpu()
                .numpy()

            )

        # -------------------------------------------------
        # Average Loss
        # -------------------------------------------------

        train_loss = (

            running_loss /

            len(self.train_loader)

        )

        # -------------------------------------------------
        # Metrics
        # -------------------------------------------------

        metrics = calculate_metrics(

            all_labels,

            all_predictions

        )

        return train_loss, metrics

    # =====================================================
    # Print Training Results
    # =====================================================

    @staticmethod
    def print_train_results(
        epoch,
        loss,
        metrics
    ):
        """
        Print training statistics.
        """

        logger.info("%s", "=" * 60)

        logger.info("Epoch %d", epoch)

        logger.info("%s", "=" * 60)

        logger.info("Training Loss : %.4f", loss)
        logger.info("Accuracy      : %.4f", metrics['accuracy'])
        logger.info("Precision     : %.4f", metrics['precision'])
        logger.info("Recall        : %.4f", metrics['recall'])
        logger.info("F1 Score      : %.4f", metrics['f1_score'])
            # =====================================================
    # Validation
    # =====================================================

    def validate(self):
        """
        Validate the model.

        Returns
        -------
        validation_loss : float
            Average validation loss.

        metrics : dict
            Validation metrics.
        """

        self.model.eval()

        running_loss = 0.0

        all_predictions = []

        all_labels = []

        with torch.no_grad():

            for features, labels in tqdm(
                self.validation_loader,
                desc="Validation",
                leave=False,
                unit="batch"
            ):

                features = features.to(self.device)

                labels = labels.to(self.device)

                outputs = self.model(features)

                loss = self.criterion(
                    outputs,
                    labels
                )

                running_loss += loss.item()

                predictions = torch.argmax(
                    outputs,
                    dim=1
                )

                all_predictions.extend(

                    predictions.cpu().numpy()

                )

                all_labels.extend(

                    labels.cpu().numpy()

                )

        validation_loss = (

            running_loss /

            len(self.validation_loader)

        )

        metrics = calculate_metrics(

            all_labels,

            all_predictions

        )

        return validation_loss, metrics


    # =====================================================
    # Test
    # =====================================================

    def test(self):
        """
        Evaluate the best saved model
        on the test dataset.
        """

        logger.info("\nLoading Best Model...")

        self.model, _, _, _, _ = self.checkpoint.load(

            self.model,

            self.optimizer,

            self.scheduler,

            device=str(self.device)

        )

        self.model.eval()

        all_predictions = []

        all_labels = []

        with torch.no_grad():

            for features, labels in tqdm(
                self.test_loader,
                desc="Testing",
                leave=False,
                unit="batch"
            ):

                features = features.to(self.device)

                labels = labels.to(self.device)

                outputs = self.model(features)

                predictions = torch.argmax(
                    outputs,
                    dim=1
                )

                all_predictions.extend(

                    predictions.cpu().numpy()

                )

                all_labels.extend(

                    labels.cpu().numpy()

                )

        metrics = calculate_metrics(

            all_labels,

            all_predictions

        )

        logger.info("%s", "=" * 60)

        logger.info("FINAL TEST RESULTS")

        logger.info("%s", "=" * 60)

        print(

            f"Accuracy  : {metrics['accuracy']:.4f}"

        )

        print(

            f"Precision : {metrics['precision']:.4f}"

        )

        print(

            f"Recall    : {metrics['recall']:.4f}"

        )

        print(

            f"F1 Score  : {metrics['f1_score']:.4f}"

        )

        # ------------------------------------------------
        # Save results — confusion matrix, classification
        # report (TXT + CSV), metrics JSON, and append
        # to model_comparison.csv (same as all baselines)
        # ------------------------------------------------

        logger.info("\nSaving FT-Transformer evaluation results...")

        evaluate_model(
            "ft_transformer",
            all_labels,
            all_predictions
        )

        return metrics


    # =====================================================
    # Validation Utilities
    # =====================================================

    @staticmethod
    def print_validation_results(
        loss,
        metrics
    ):
        """
        Print validation statistics.
        """

        logger.info("\nValidation")

        logger.info("%s", "-" * 60)

        logger.info("Validation Loss : %.4f", loss)
        logger.info("Accuracy         : %.4f", metrics['accuracy'])
        logger.info("Precision        : %.4f", metrics['precision'])
        logger.info("Recall           : %.4f", metrics['recall'])
        logger.info("F1 Score         : %.4f", metrics['f1_score'])


    # =====================================================
    # Scheduler Helper
    # =====================================================

    def step_scheduler(self, validation_accuracy):
        """
        Step the configured scheduler using the
        correct API for the scheduler type.
        """

        if isinstance(self.scheduler, lr_scheduler.ReduceLROnPlateau):

            self.scheduler.step(
                validation_accuracy
            )

        else:

            self.scheduler.step()

    # =====================================================
    # Update Training Utilities
    # =====================================================

    def update_training(
        self,
        epoch,
        validation_metrics
    ):
        """
        Update scheduler,
        save checkpoint,
        and perform early stopping.

        Returns
        -------
        bool
            True if training should stop.
        """

        validation_accuracy = (

            validation_metrics["accuracy"]

        )

        # Scheduler

        self.step_scheduler(validation_accuracy)

        # Checkpoint

        self.checkpoint.save(

            model=self.model,

            optimizer=self.optimizer,

            scheduler=self.scheduler,

            epoch=epoch,

            best_score=validation_accuracy

        )

        # Early Stopping

        stop = self.early_stopping(

            validation_accuracy

        )

        return stop
        # =====================================================
    # Complete Training Loop
    # =====================================================

    def train(self):
        """
        Train the FT-Transformer model.
        """

        logger.info("%s", "=" * 60)
        logger.info("STARTING TRAINING")
        logger.info("%s", "=" * 60)

        for epoch in range(self.start_epoch, EPOCHS):

            logger.info("\nEpoch [%d/%d]", epoch + 1, EPOCHS)

            # ---------------------------------------------
            # Train
            # ---------------------------------------------

            train_loss, train_metrics = self.train_one_epoch()

            self.history["train_loss"].append(train_loss)
            self.history["train_accuracy"].append(
                train_metrics["accuracy"]
            )
            self.history["train_f1_score"].append(
                train_metrics["f1_score"]
            )

            self.print_train_results(
                epoch + 1,
                train_loss,
                train_metrics
            )

            # ---------------------------------------------
            # Validation
            # ---------------------------------------------

            validation_loss, validation_metrics = self.validate()

            self.history["validation_loss"].append(validation_loss)
            self.history["validation_accuracy"].append(
                validation_metrics["accuracy"]
            )
            self.history["validation_f1_score"].append(
                validation_metrics["f1_score"]
            )

            self.print_validation_results(
                validation_loss,
                validation_metrics
            )

            # ---------------------------------------------
            # Update scheduler, checkpoint, and early stopping
            # ---------------------------------------------

            if self.update_training(
                epoch,
                validation_metrics
            ):
                logger.info("\nEarly stopping triggered.")
                break

        logger.info("\nTraining Completed.")

    # =====================================================
    # Run Complete Pipeline
    # =====================================================

    def run(self):
        """
        Execute complete training pipeline.
        """

        self.train()

        logger.info("\nEvaluating Best Model...")

        self.test()
# =====================================================
# Main
# =====================================================

def main():

    trainer = Trainer()

    trainer.run()


if __name__ == "__main__":

    main()