# hrf/prompts.py
from __future__ import annotations

SENTENCE = {
    "p1": (
        "1. [Topic Reasoning]: Given S and I_CLIP, Based on common sense, "
        "what topic might it be discussing? Return ONLY [R1].\n\n"
        "S: {S}\nI_CLIP: {I_CLIP}\n"
    ),
    "p2": (
        "2. [Opinion Reasoning]: Given S, I_CLIP and R1, what is the implicit opinion "
        "on the aforementioned topic and why? Return ONLY [R2].\n\n"
        "S: {S}\nI_CLIP: {I_CLIP}\nR1: {R1}\n"
    ),
    "p3": (
        "3. [Sentiment Reasoning]: Given S, I_CLIP and R2, Based on such an opinion, "
        "is the overall sentiment positive/neutral/negative? Return ONLY one word [R3].\n\n"
        "S: {S}\nI_CLIP: {I_CLIP}\nR2: {R2}\n"
    ),
}

ASPECT = {
    "p1": (
        "1. [Aspect Reasoning]: Given S, I_CLIP and A, which specific aspect of A is mentioned? "
        "Return ONLY [R1].\n\n"
        "S: {S}\nI_CLIP: {I_CLIP}\nA: {A}\n"
    ),
    "p2": (
        "2. [Opinion Reasoning]: Given S, I_CLIP, A and R1, Based on common sense, "
        "what is the implicit opinion towards the mentioned aspect of A, and why? "
        "Return ONLY [R2].\n\n"
        "S: {S}\nI_CLIP: {I_CLIP}\nA: {A}\nR1: {R1}\n"
    ),
    "p3": (
        "3. [Sentiment Reasoning]: Given S, I_CLIP, A and R2, Based on such opinion what is sentiment polarity towards A? "
        "Return ONLY one word [R3].\n\n"
        "S: {S}\nI_CLIP: {I_CLIP}\nA: {A}\nR2: {R2}\n"
    ),
}


def get_prompts(mode: str):
    if mode == "sentence":
        return SENTENCE
    if mode == "aspect":
        return ASPECT
    raise ValueError("mode must be 'sentence' or 'aspect'")
