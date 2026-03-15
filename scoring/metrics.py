"""Audio comparison metrics for the feedback optimization loop.

Each function takes two audio signals (or file paths) and returns
a similarity score in [0, 1] where 1 = perfect match.
"""
import librosa
import numpy as np


def _load_if_needed(audio, sr=22050):
    """Accept either a file path or (y, sr) tuple."""
    if isinstance(audio, str):
        return librosa.load(audio, sr=sr)
    return audio


def chroma_similarity(audio_a, audio_b, sr=22050):
    """Compare pitch content via chroma features.

    Computes time-averaged chroma vectors and returns cosine similarity.
    This is the most important metric — tells you if the right notes
    are present regardless of timbre differences.
    """
    y_a, sr_a = _load_if_needed(audio_a, sr)
    y_b, sr_b = _load_if_needed(audio_b, sr)

    chroma_a = np.mean(librosa.feature.chroma_cqt(y=y_a, sr=sr_a), axis=1)
    chroma_b = np.mean(librosa.feature.chroma_cqt(y=y_b, sr=sr_b), axis=1)

    norm_a = np.linalg.norm(chroma_a)
    norm_b = np.linalg.norm(chroma_b)
    if norm_a < 1e-10 or norm_b < 1e-10:
        return 0.0

    return float(np.dot(chroma_a, chroma_b) / (norm_a * norm_b))


def onset_alignment(audio_a, audio_b, sr=22050, tolerance_ms=50):
    """Compare timing of note attacks via onset detection.

    Returns F1 score of onset matches within the tolerance window.
    """
    y_a, sr_a = _load_if_needed(audio_a, sr)
    y_b, sr_b = _load_if_needed(audio_b, sr)

    onsets_a = librosa.onset.onset_detect(y=y_a, sr=sr_a, units='time')
    onsets_b = librosa.onset.onset_detect(y=y_b, sr=sr_b, units='time')

    if len(onsets_a) == 0 or len(onsets_b) == 0:
        return 0.0

    tolerance = tolerance_ms / 1000.0
    matched_b = set()
    true_positives = 0

    for oa in onsets_a:
        for j, ob in enumerate(onsets_b):
            if j not in matched_b and abs(oa - ob) <= tolerance:
                true_positives += 1
                matched_b.add(j)
                break

    precision = true_positives / len(onsets_b) if len(onsets_b) > 0 else 0
    recall = true_positives / len(onsets_a) if len(onsets_a) > 0 else 0

    if precision + recall == 0:
        return 0.0
    return float(2 * precision * recall / (precision + recall))


def spectral_distance(audio_a, audio_b, sr=22050):
    """Compare overall spectral shape via MFCCs.

    Returns similarity (1 - normalized distance).
    """
    y_a, sr_a = _load_if_needed(audio_a, sr)
    y_b, sr_b = _load_if_needed(audio_b, sr)

    mfcc_a = np.mean(librosa.feature.mfcc(y=y_a, sr=sr_a, n_mfcc=13), axis=1)
    mfcc_b = np.mean(librosa.feature.mfcc(y=y_b, sr=sr_b, n_mfcc=13), axis=1)

    dist = np.linalg.norm(mfcc_a - mfcc_b)
    # Normalize: typical MFCC distances range 0-200
    similarity = max(0.0, 1.0 - dist / 200.0)
    return float(similarity)


def pitch_histogram_similarity(audio_a, audio_b, sr=22050):
    """Compare pitch class distributions.

    Builds 12-bin pitch class histograms and computes cosine similarity.
    """
    y_a, sr_a = _load_if_needed(audio_a, sr)
    y_b, sr_b = _load_if_needed(audio_b, sr)

    chroma_a = librosa.feature.chroma_cqt(y=y_a, sr=sr_a)
    chroma_b = librosa.feature.chroma_cqt(y=y_b, sr=sr_b)

    hist_a = np.sum(chroma_a, axis=1)
    hist_b = np.sum(chroma_b, axis=1)

    norm_a = np.linalg.norm(hist_a)
    norm_b = np.linalg.norm(hist_b)
    if norm_a < 1e-10 or norm_b < 1e-10:
        return 0.0

    return float(np.dot(hist_a, hist_b) / (norm_a * norm_b))


def combined_score(audio_a, audio_b, sr=22050, weights=None):
    """Compute weighted combination of all metrics.

    Args:
        weights: dict with keys 'chroma', 'onset', 'spectral', 'pitch_histogram'
    """
    if weights is None:
        from config import SCORING_WEIGHTS
        weights = SCORING_WEIGHTS

    scores = {
        "chroma": chroma_similarity(audio_a, audio_b, sr),
        "onset": onset_alignment(audio_a, audio_b, sr),
        "spectral": spectral_distance(audio_a, audio_b, sr),
        "pitch_histogram": pitch_histogram_similarity(audio_a, audio_b, sr),
    }

    total = sum(scores[k] * weights.get(k, 0.25) for k in scores)
    scores["total"] = float(total)
    return scores
