"""
Observer LLM: reads recent dialog and outputs 3 structured scores.

Output JSON:
{
  "deal_probability": float in [0, 1],
  "breakdown_risk": float in [0, 1],
  "opponent_emotion": one of the active emotion labels (taxonomy-dependent)
}

Called once per turn. Failures fall back to neutral defaults.
"""

import json
import re
from typing import Dict, List, Tuple
from langchain_core.messages import HumanMessage
from llm.llm_wrapper import LLMWrapper

from EmoDistill.emotions import (
    emotion_to_idx,
    get_emotions,
    n_emotions,
    parse_emotion_str,
)


def _build_prompt_template() -> str:
    emo_list = ", ".join(get_emotions())
    return (
        "You are an impartial negotiation observer. Read the recent dialog between Creditor and Debtor "
        "and assess the current state.\n\n"
        "NEGOTIATION CONTEXT:\n"
        "- Creditor target: {creditor_target} {time_unit}\n"
        "- Debtor target: {debtor_target} {time_unit}\n"
        "- Current turn: {turn} of {max_turn}\n\n"
        "RECENT DIALOG:\n"
        "{dialog_excerpt}\n\n"
        "Output ONLY a JSON object (no prose, no markdown fences) with exactly these fields:\n"
        "{{\n"
        '  "deal_probability": <float 0-1, likelihood the two sides will reach agreement in remaining turns>,\n'
        '  "breakdown_risk": <float 0-1, likelihood the negotiation will break down without agreement>,\n'
        f'  "opponent_emotion": <one of: {emo_list}>\n'
        "}}\n\n"
        "JSON:"
    )


class NegotiationObserver:
    """Single-call LLM observer producing 3 state features."""

    def __init__(self, model_name: str = "gpt-4o-mini", history_window: int = 4):
        self.llm = LLMWrapper(model_name, role="observer")
        self.history_window = history_window
        self.call_count = 0
        self.fail_count = 0

    def observe(
        self,
        dialog_history: List[Tuple[str, str]],
        creditor_target: int,
        debtor_target: int,
        time_unit: str,
        turn: int,
        max_turn: int,
    ) -> Dict[str, float]:
        excerpt_lines = []
        for speaker, msg in dialog_history[-self.history_window:]:
            label = "Creditor" if speaker == "seller" else "Debtor"
            truncated = msg[:300]
            excerpt_lines.append(f"{label}: {truncated}")
        dialog_excerpt = "\n".join(excerpt_lines) if excerpt_lines else "(no dialog yet)"

        prompt = _build_prompt_template().format(
            creditor_target=creditor_target,
            debtor_target=debtor_target,
            time_unit=time_unit,
            turn=turn,
            max_turn=max_turn,
            dialog_excerpt=dialog_excerpt,
        )

        try:
            self.call_count += 1
            response = self.llm.invoke([HumanMessage(content=prompt)], temperature=0.2)
            return self._parse(response.content.strip())
        except Exception as e:
            self.fail_count += 1
            print(f"        ⚠️  Observer call failed ({e}); using defaults")
            return self._default()

    def _parse(self, text: str) -> Dict[str, float]:
        match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
        if not match:
            self.fail_count += 1
            return self._default()

        try:
            obj = json.loads(match.group(0))
        except json.JSONDecodeError:
            self.fail_count += 1
            return self._default()

        deal_prob = self._clip01(obj.get("deal_probability", 0.5))
        breakdown = self._clip01(obj.get("breakdown_risk", 0.3))
        emotion_str = str(obj.get("opponent_emotion", "neutral")).strip().lower()
        emo_idx = parse_emotion_str(emotion_str)
        # Use canonical label from active taxonomy
        canonical = get_emotions()[emo_idx]

        return {
            "deal_probability": deal_prob,
            "breakdown_risk": breakdown,
            "opponent_emotion": canonical,
            "opponent_emotion_idx": emo_idx,
        }

    @staticmethod
    def _clip01(x) -> float:
        try:
            v = float(x)
        except (TypeError, ValueError):
            return 0.5
        return max(0.0, min(1.0, v))

    @staticmethod
    def _default() -> Dict[str, float]:
        e2i = emotion_to_idx()
        neutral_idx = e2i.get("neutral", 0)
        return {
            "deal_probability": 0.5,
            "breakdown_risk": 0.3,
            "opponent_emotion": get_emotions()[neutral_idx],
            "opponent_emotion_idx": neutral_idx,
        }

    def get_stats(self) -> Dict[str, int]:
        return {
            "observer_calls": self.call_count,
            "observer_failures": self.fail_count,
            "failure_rate": self.fail_count / max(1, self.call_count),
        }
