"""Prompts for the LLM-as-judge evaluator."""


def build_judge_prompt(
    question: str,
    gold_answer: str,
    predicted_answer: str,
    category: str,
) -> list[dict]:
    """Build the prompt for the LLM judge to score a predicted answer.

    The judge scores on multiple dimensions and returns structured JSON.
    """
    system = (
        "You are an expert evaluator for a multi-user memory management benchmark. "
        "Your task is to compare a predicted answer against a gold (reference) answer "
        "and score the prediction on multiple dimensions.\n\n"
        "Score each dimension from 1-5:\n"
        "  1 = Completely wrong or missing\n"
        "  2 = Mostly wrong with minor correct elements\n"
        "  3 = Partially correct but with significant gaps or errors\n"
        "  4 = Mostly correct with minor gaps\n"
        "  5 = Fully correct and complete\n\n"
        "Scoring dimensions:\n"
        "  - correctness: Does the predicted answer contain the right facts?\n"
        "  - completeness: Does it cover all aspects of the gold answer?\n"
        "  - attribution: Does it correctly attribute information to the right users?\n"
        "  - hallucination: Does it avoid fabricating facts not in the gold answer? "
        "(5 = no hallucinations, 1 = severe hallucinations)\n\n"
        "Respond with ONLY valid JSON in this format:\n"
        "{\n"
        '  "correctness": <1-5>,\n'
        '  "completeness": <1-5>,\n'
        '  "attribution": <1-5>,\n'
        '  "hallucination": <1-5>,\n'
        '  "reasoning": "<brief explanation of scores>"\n'
        "}"
    )

    user_msg = (
        f"## Evaluation Category: {category}\n\n"
        f"## Question\n{question}\n\n"
        f"## Gold Answer (Reference)\n{gold_answer}\n\n"
        f"## Predicted Answer\n{predicted_answer}\n\n"
        f"Score the predicted answer against the gold answer. Respond with JSON only."
    )

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user_msg},
    ]


def build_binary_judge_prompt(
    question: str,
    gold_answer: str,
    predicted_answer: str,
) -> list[dict]:
    """Build prompt for binary (correct/incorrect) evaluation.

    Used for categories like adversarial_confusion where the answer is binary.
    """
    system = (
        "You are an expert evaluator. Compare the predicted answer to the gold answer "
        "and determine if the prediction is correct.\n\n"
        "Respond with ONLY valid JSON:\n"
        "{\n"
        '  "correct": true or false,\n'
        '  "reasoning": "<brief explanation>"\n'
        "}"
    )

    user_msg = (
        f"## Question\n{question}\n\n"
        f"## Gold Answer\n{gold_answer}\n\n"
        f"## Predicted Answer\n{predicted_answer}\n\n"
        f"Is the predicted answer correct?"
    )

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user_msg},
    ]
