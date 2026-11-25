"""
File: predictor.py

Purpose:
High-level prediction interface for the FT-Transformer model.

This module loads the trained model, scaler,
and label encoder, then performs crop prediction.
"""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
import torch

# =====================================================
# Imports
# =====================================================

try:
    # Package import
    from .config import (
        DEVICE,
        MODEL_SAVE_PATH,
        SCALER_PATH,
        LABEL_ENCODER_PATH,
    )
    from .model import FTTransformer

except ImportError:
    # Direct execution
    from config import (
        DEVICE,
        MODEL_SAVE_PATH,
        SCALER_PATH,
        LABEL_ENCODER_PATH,
    )
    from model import FTTransformer


class CropPredictor:
    """
    High-level FT-Transformer predictor.
    """

    def __init__(self):

        self.device = DEVICE

        print("=" * 60)
        print("LOADING FT-TRANSFORMER")
        print("=" * 60)

        # -------------------------------------------------
        # Load Model
        # -------------------------------------------------

        self.model = FTTransformer().to(self.device)

        checkpoint = torch.load(
            MODEL_SAVE_PATH,
            map_location=self.device
        )

        if isinstance(checkpoint, dict):

            if "model_state_dict" in checkpoint:

                self.model.load_state_dict(
                    checkpoint["model_state_dict"]
                )

            elif "state_dict" in checkpoint:

                self.model.load_state_dict(
                    checkpoint["state_dict"]
                )

            else:

                self.model.load_state_dict(checkpoint)

        else:

            self.model.load_state_dict(checkpoint)

        self.model.eval()

        print("Model Loaded Successfully.")

        # -------------------------------------------------
        # Load Scaler
        # -------------------------------------------------

        self.scaler = joblib.load(SCALER_PATH)

        print("Scaler Loaded Successfully.")

        # -------------------------------------------------
        # Load Label Encoder
        # -------------------------------------------------

        self.label_encoder = joblib.load(
            LABEL_ENCODER_PATH
        )

        print("Label Encoder Loaded Successfully.")

        print("=" * 60)

    # =====================================================
    # Preprocess
    # =====================================================

    def preprocess(
        self,
        features
    ) -> torch.Tensor:

        if isinstance(features, dict):

            features = pd.DataFrame([features])

        elif isinstance(features, list):

            features = pd.DataFrame([features])

        elif isinstance(features, np.ndarray):

            features = pd.DataFrame(features)

        if features.shape[1] != 7:

            raise ValueError(
                f"Expected 7 input features, got {features.shape[1]}"
            )

        scaled = self.scaler.transform(features)

        tensor = torch.tensor(
            scaled,
            dtype=torch.float32
        )

        return tensor.to(self.device)

    # =====================================================
    # Prediction
    # =====================================================

    def predict(
        self,
        features,
        top_k: int = 3
    ) -> dict:

        inputs = self.preprocess(features)

        with torch.no_grad():

            outputs = self.model(inputs)

            probabilities = torch.softmax(
                outputs,
                dim=1
            )

        confidence, indices = torch.topk(
            probabilities,
            k=top_k
        )

        confidence = confidence.cpu().numpy()[0]

        indices = indices.cpu().numpy()[0]

        crop_names = self.label_encoder.inverse_transform(
            indices
        )

        return {

            "prediction": crop_names[0],

            "confidence": round(
                float(confidence[0]),
                4
            ),

            "top_predictions": [

                {

                    "crop": crop,

                    "confidence": round(
                        float(score),
                        4
                    )

                }

                for crop, score in zip(
                    crop_names,
                    confidence
                )

            ]

        }


# =====================================================
# Testing
# =====================================================

if __name__ == "__main__":

    predictor = CropPredictor()

    sample = {

        "N": 90,
        "P": 42,
        "K": 43,
        "temperature": 21.5,
        "humidity": 82,
        "ph": 6.5,
        "rainfall": 202

    }

    prediction = predictor.predict(sample)

    print("\nPrediction Result")

    print("-" * 60)

    print(f"Recommended Crop : {prediction['prediction']}")

    print(f"Confidence       : {prediction['confidence']:.4f}")

    print("\nTop Predictions")

    print("-" * 60)

    for i, item in enumerate(
        prediction["top_predictions"],
        start=1
    ):

        print(
            f"{i}. {item['crop']} ({item['confidence']:.4f})"
        )