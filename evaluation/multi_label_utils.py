import numpy as np
from scipy.special import expit


def fix_multi_label_scores(original_predictions, original_label_ids, unpad_sequences=False, flatten_sequences=False):

    if flatten_sequences:
        predictions = original_predictions.reshape((-1, original_predictions.shape[-1]))
        label_ids = original_label_ids.reshape((-1, original_predictions.shape[-1]))
    else:
        predictions = original_predictions
        label_ids = original_label_ids

    if unpad_sequences:
        predictions = np.asarray([pred for pred, label in zip(predictions, label_ids) if label[0] != -1])
        label_ids = np.asarray([label for label in label_ids if label[0] != -1])

    # Fix gold labels
    y_true = np.zeros((len(label_ids), len(label_ids[0]) + 1), dtype=np.int32)
    y_true[:, :-1] = label_ids
    y_true[:, -1] = (np.sum(label_ids, axis=1) == 0).astype('int32')
    # Fix predictions
    logits = predictions[0] if isinstance(predictions, tuple) else predictions
    preds = (expit(logits) > 0.5).astype('int32')
    y_pred = np.zeros((len(label_ids), len(label_ids[0]) + 1), dtype=np.int32)
    y_pred[:, :-1] = preds
    y_pred[:, -1] = (np.sum(preds, axis=1) == 0).astype('int32')

    return y_true, y_pred
