from rapidfuzz import fuzz


def _normalize_text(text: str) -> str:
    """Normalize text for comparison."""
    if text is None:
        return ""
    return text.strip().lower()


def calc_nid(gt_text: str, pred_text: str) -> float:
    """Calculate the Normalized Indel score between the gt and pred text.
    Args:
        gt_text (str): The string of gt text to compare.
        pred_text (str): The string of pred text to compare.
    Returns:
        float: The nid score between gt and pred text. [0., 1.]
    """
    gt_text = _normalize_text(gt_text)
    pred_text = _normalize_text(pred_text)

    # if gt and pred is empty, return 1
    if len(gt_text) == 0 and len(pred_text) == 0:
        score = 1
    # if pred is empty while gt is not, return 0
    elif len(gt_text) > 0 and len(pred_text) == 0:
        score = 0
    else:
        score = fuzz.ratio(gt_text, pred_text)

    return score
