

def predict_verification(question: str, selected_answer_text: str, correct_answer_text: str) -> dict:
    """Stable dict for quiz tab until real Model A is wired."""
    sel = str(selected_answer_text).strip().lower()
    gold = str(correct_answer_text).strip().lower()
    correct = sel == gold and bool(sel)

    return {
        "is_correct": bool(correct),
        "confidence": 0.85 if correct else 0.75,
        "predicted_label": "CORRECT_VERIFIED" if correct else "INCORRECT",
    }
