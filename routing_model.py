"""Routing model that dispatches predictions to distance-specialized sub-models.
This class mimics the LightGBM Booster.predict(X) interface so it works
transparently with app.py without any modifications.
"""
import numpy as np


class RoutingModel:
    """Routes predictions to specialized models based on distance feature value."""

    def __init__(self, models, distance_feature_idx, fallback_model=None):
        """
        models: dict mapping distance_group -> lgb.Booster
            e.g. {'sprint': lgb_sprint, 'middle': lgb_middle, 'long': lgb_long}
        distance_feature_idx: int, index of 'distance' feature in the feature array
        fallback_model: lgb.Booster to use when no specialized model matches
        """
        self.models = models
        self.distance_feature_idx = distance_feature_idx
        self.fallback_model = fallback_model
        # Distance thresholds
        self.groups = [
            ('sprint', 0, 1400),
            ('middle', 1401, 2000),
            ('long', 2001, 9999),
        ]

    def _get_group(self, distance):
        for name, lo, hi in self.groups:
            if lo <= distance <= hi:
                return name
        return None

    def predict(self, data, **kwargs):
        """Predict using distance-specialized models."""
        if not isinstance(data, np.ndarray):
            data = np.array(data)

        n = data.shape[0]
        preds = np.zeros(n)

        # Get distance values
        distances = data[:, self.distance_feature_idx]

        for group_name, lo, hi in self.groups:
            mask = (distances >= lo) & (distances <= hi)
            if not mask.any():
                continue
            model = self.models.get(group_name, self.fallback_model)
            if model is not None:
                preds[mask] = model.predict(data[mask], **kwargs)

        # Fallback for any unmatched rows
        unmatched = preds == 0
        if unmatched.any() and self.fallback_model is not None:
            preds[unmatched] = self.fallback_model.predict(data[unmatched], **kwargs)

        return preds

    def feature_importance(self, importance_type='gain'):
        """Return feature importance from fallback (global) model."""
        if self.fallback_model is not None:
            return self.fallback_model.feature_importance(importance_type=importance_type)
        # Return from first available model
        for m in self.models.values():
            if m is not None:
                return m.feature_importance(importance_type=importance_type)
        return np.array([])
