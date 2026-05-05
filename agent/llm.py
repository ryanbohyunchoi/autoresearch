import os
import anthropic

_client = None

def get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    return _client


def propose_modification(
    program_md: str,
    current_code: str,
    history: list[dict],
    model: str = "claude-opus-4-7",
) -> str:
    """Ask Claude to propose a modified version of the training script."""
    system = (
        "You are an expert ML researcher. You will be given a training script and "
        "a history of past experiments with their validation metrics. "
        "Propose ONE concrete, focused modification to improve the metric. "
        "Return ONLY the complete modified Python file, no explanation, no markdown fences."
    )

    history_text = ""
    if history:
        history_text = "\n\n## Experiment history (oldest → newest)\n"
        for i, h in enumerate(history):
            history_text += f"\n### Experiment {i+1} (metric={h['metric']:.4f})\n"
            history_text += f"Change made: {h['description']}\n"

    user = (
        f"## Instructions\n{program_md}\n"
        f"{history_text}\n"
        f"## Current train.py\n```python\n{current_code}\n```\n\n"
        "Return the full modified train.py."
    )

    resp = get_client().messages.create(
        model=model,
        max_tokens=8192,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return resp.content[0].text.strip()


def describe_change(original: str, modified: str, model: str = "claude-haiku-4-5-20251001") -> str:
    """One-line description of what changed between two versions of train.py."""
    user = (
        "In one short sentence, describe the key change made to train.py.\n\n"
        f"ORIGINAL:\n```python\n{original}\n```\n\n"
        f"MODIFIED:\n```python\n{modified}\n```"
    )
    resp = get_client().messages.create(
        model=model,
        max_tokens=100,
        messages=[{"role": "user", "content": user}],
    )
    return resp.content[0].text.strip()
