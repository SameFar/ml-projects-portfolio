import logging
import torch


def predict_preferences(model, dataloader, threshold: float = None):
    """
    Evaluates an entire batch collection from a DataLoader against your trained
    One-Class cluster. Respects the model's native device state.

    Args:
        model: Trained PannGru instance (must contain a valid model.center attribute)
        dataloader: A DataLoader instance yielding batches of processed waveforms
        threshold: Optional manual threshold. If None, defaults to the dynamic
                   model.threshold if available, falling back to 0.065.

    Returns:
        list of dict: Distance profiles and verdicts for every track processed.
    """
    center = getattr(model, "center", None)
    if center is None:
        raise ValueError(
            "The provided model instance hasn't been calibrated with an SVDD center yet."
        )

    # Prioritize threshold: 1. Argument value -> 2. Saved model value -> 3. Hardcoded fallback
    if threshold is None:
        threshold = getattr(model, "threshold", 0.065)

    model.eval()
    predictions = []

    # Track the model's natural device placement
    model_device = next(model.parameters()).device
    # Ensure our reference center matches the model's environment space
    working_center = center.to(model_device)

    logging.info(
        f"Evaluating dataloader tracks on native device: {model_device} (Threshold: {threshold:.4f})..."
    )

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            waveforms = batch[0].to(model_device)

            # If the waveforms array is missing batch dims, make it 2D [Batch, Samples]
            if len(waveforms.shape) == 1:
                waveforms = waveforms.unsqueeze(0)

            # Extract the 128-dimensional embeddings from the projection head
            final_embeddings = model(waveforms)  # Shape: [batch_size, 128]

            # Calculate Deep SVDD Anomaly Scores (Squared Euclidean Distance) for the entire batch
            distances = torch.sum((final_embeddings - working_center) ** 2, dim=1)

            # Loop through batch elements to compile structural outputs
            for idx, dist_tensor in enumerate(distances):
                distance_score = dist_tensor.item()
                is_favorite = distance_score <= threshold
                verdict = "Liked" if is_favorite else "Disliked"

                predictions.append(
                    {
                        "batch_idx": i,
                        "sample_idx": idx,
                        "distance_score": round(
                            distance_score, 6
                        ),  # Increased precision for small variance tracking
                        "verdict": verdict,
                        "is_favorite": is_favorite,
                    }
                )

    logging.info(f"Completed evaluation for {len(predictions)} track profiles.")
    return predictions
