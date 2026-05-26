"""
NegotiatorNew: extends DebtNegotiator to drive DQN-new.

Key differences from DebtNegotiator:
1. On each creditor turn, builds a 16-dim state vector from dialog + offers,
   calls the observer LLM (if enabled), and passes the state to DQN-new via
   `state_vec` in model_state.
2. After the negotiation ends, walks the recorded (state, action, observer)
   per-turn records and synthesizes per-step transitions with DENSE rewards
   grounded in Δopponent_offer, then hands them to DQN-new via record_step().
3. update_model is left to the caller (run_dqn_new_experiment), so the
   negotiator does not auto-train.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from langchain_core.messages import HumanMessage

from llm.negotiator import DebtNegotiator, GameState
from llm.prompt_templates import PromptTemplates
from EmoDistill.observer import NegotiationObserver
from EmoDistill.emotions import emotion_to_idx, get_emotions, parse_emotion_str
from EmoDistill.state_builder import build_state, extract_offer_trajectory
from EmoDistill.reward import compute_step_reward


class NegotiatorNew(DebtNegotiator):
    """Negotiator with enriched state + observer + dense reward bookkeeping."""

    def __init__(
        self,
        config: Dict[str, Any],
        emotion_model,
        model_creditor: str = "gpt-4o-mini",
        model_debtor: str = "gpt-4o-mini",
        debtor_emotion: str = "neutral",
        debtor_model_type: str = "vanilla",
        observer_model: str = "gpt-4o-mini",
        use_observer: bool = True,
        max_dialog_len: int = 30,
    ):
        super().__init__(
            config=config,
            emotion_model=emotion_model,
            model_creditor=model_creditor,
            model_debtor=model_debtor,
            debtor_emotion=debtor_emotion,
            debtor_model_type=debtor_model_type,
        )
        self.max_dialog_len = max_dialog_len
        self.use_observer = use_observer
        self.observer = NegotiationObserver(model_name=observer_model) if use_observer else None

        self.creditor_target = int(self.config["seller"]["target_price"])
        self.debtor_target = int(self.config.get("buyer", {}).get("target_price", self.creditor_target * 3))
        self.debtor_initial = self.debtor_target  # snapshot at episode start

        # Per-turn records: each entry = {turn, state_vec, action, observer, last_emotion}
        self.turn_records: List[Dict[str, Any]] = []

    # -------- Override creditor_node --------

    def creditor_node(self, state: GameState):
        """Override to enrich model_state with full state_vec + observer features."""
        self.negotiation_round += 1

        history: List[Tuple[str, str]] = state.get("history", [])
        last_emotion = self.emotion_sequence[-1] if self.emotion_sequence else "neutral"

        # Observer call (if enabled)
        obs_features = None
        if self.use_observer and self.observer is not None:
            obs_features = self.observer.observe(
                dialog_history=history,
                creditor_target=self.creditor_target,
                debtor_target=self.debtor_target,
                time_unit=self.get_time_unit(),
                turn=self.negotiation_round,
                max_turn=self.max_dialog_len,
            )

        # Build 16-dim state
        state_vec = build_state(
            dialog=history,
            days_extractor=self.extract_days,
            last_emotion=last_emotion,
            turn=self.negotiation_round,
            max_turn=self.max_dialog_len,
            creditor_target=self.creditor_target,
            debtor_target=self.debtor_target,
            observer_features=obs_features,
        )

        # Ask DQN-new for action
        model_state = {
            "state_vec": state_vec,
            "round": self.negotiation_round,
            "current_emotion": last_emotion,
            "debtor_emotion": self.debtor_emotion,
        }
        emotion_config = self.emotion_model.select_emotion(model_state)
        creditor_emotion = emotion_config["emotion"]
        action_idx = int(emotion_config.get("action_idx", parse_emotion_str(creditor_emotion)))
        self.emotion_sequence.append(creditor_emotion)

        # Record per-turn data for reward computation later
        self.turn_records.append(
            {
                "turn": self.negotiation_round,
                "state_vec": state_vec,
                "action": action_idx,
                "observer": obs_features,
                "last_emotion": last_emotion,
            }
        )

        # Build creditor prompt — reuse parent logic
        config = self.config.get("seller_config", self.config["seller"])
        debt_info = self.config.get("metadata", {})

        timeline_text = self._build_timeline_text(history)
        prompt = PromptTemplates.get_creditor_prompt(
            scenario_type=self.scenario_type,
            config=config,
            emotion_config=emotion_config,
            timeline_text=timeline_text,
            debt_info=debt_info,
        )

        response = self.llm_creditor.invoke(
            [HumanMessage(content=prompt)],
            temperature=emotion_config.get("temperature", 0.7),
        )

        print(
            f"        🧠 DQN-new emotion: {creditor_emotion} (Round {self.negotiation_round}, "
            f"ε={emotion_config.get('exploration_rate', 0):.3f}, conf={emotion_config.get('confidence', 0):.2f})"
        )
        if obs_features is not None:
            print(
                f"           Observer: deal={obs_features['deal_probability']:.2f}, "
                f"breakdown={obs_features['breakdown_risk']:.2f}, opp_emo={obs_features['opponent_emotion']}"
            )
        print(f"        💬 Creditor says: \"{response.content}\"")

        new_history = state["history"] + [("seller", response.content)]

        # Reuse parent's simple agreement-check logic (within ±5)
        current_state = "offer"
        if len(new_history) >= 2:
            last_creditor_days = self.extract_days(response.content)
            debtor_days_value = None
            for speaker, message in reversed(new_history[:-1]):
                if speaker == "buyer":
                    debtor_days_value = self.extract_days(message)
                    if debtor_days_value:
                        break
            if last_creditor_days and debtor_days_value:
                diff = abs(last_creditor_days - debtor_days_value)
                if diff <= 5:
                    current_state = "accept"
                    print(f"        🎯 AGREEMENT: {last_creditor_days} vs {debtor_days_value} (diff: {diff})")

        return {
            "messages": [response],
            "turn": "buyer",
            "current_state": current_state,
            "history": new_history,
            "creditor_emotion": creditor_emotion,
        }

    # -------- Override run_negotiation to post-process rewards --------

    def run_negotiation(self, max_dialog_len: int = 30) -> Dict[str, Any]:
        self.max_dialog_len = max_dialog_len
        result = super().run_negotiation(max_dialog_len=max_dialog_len)
        # Build transitions from turn_records + final dialog
        self._flush_transitions_to_model(result)
        return result

    # -------- Helpers --------

    def _build_timeline_text(self, history: List[Tuple[str, str]]) -> str:
        # Delegate to the full-transcript builder defined on the parent class:
        # creditor sees every prior utterance verbatim, with its own past turns
        # labeled "You (Creditor):".
        return self._build_full_transcript(history, self_role="seller")

    def _flush_transitions_to_model(self, neg_result: Dict[str, Any]) -> None:
        """Walk turn_records + final dialog, synthesize transitions, push to model."""
        dialog_pairs: List[Tuple[str, str]] = [(d["speaker"], d["message"]) for d in neg_result.get("dialog", [])]
        creditor_offers, debtor_offers = extract_offer_trajectory(dialog_pairs, self.extract_days)

        initial_gap = max(1, abs(self.debtor_target - self.creditor_target))

        # Index each dialog entry to mark creditor turn positions
        creditor_dialog_indices = [i for i, d in enumerate(neg_result.get("dialog", [])) if d.get("speaker") == "seller"]
        n_records = min(len(self.turn_records), len(creditor_dialog_indices))

        debtor_offer_history: List[int] = [
            self.debtor_target,  # seed with initial debtor target
        ]
        # Extend with actual debtor offers seen during episode
        for entry in neg_result.get("dialog", []):
            if entry.get("speaker") == "buyer":
                d_offer = entry.get("requested_days")
                if d_offer is not None:
                    debtor_offer_history.append(int(d_offer))

        for i in range(n_records):
            rec = self.turn_records[i]
            dialog_idx = creditor_dialog_indices[i]

            # The debtor message that responded to this creditor action lives at dialog_idx+1
            debtor_idx = dialog_idx + 1
            debtor_msg = ""
            new_debtor_offer: Optional[int] = None
            if debtor_idx < len(neg_result.get("dialog", [])):
                debtor_entry = neg_result["dialog"][debtor_idx]
                debtor_msg = debtor_entry.get("message", "")
                new_debtor_offer = debtor_entry.get("requested_days")
                if new_debtor_offer is not None:
                    new_debtor_offer = int(new_debtor_offer)

            # The previous debtor offer = the most recent debtor offer BEFORE this creditor action
            prev_debtor_offer = self._most_recent_debtor_offer_before(neg_result.get("dialog", []), dialog_idx)
            if prev_debtor_offer is None:
                prev_debtor_offer = self.debtor_target  # fallback to initial

            obs_breakdown = float(rec["observer"]["breakdown_risk"]) if rec.get("observer") else 0.0

            # Trim local debtor_offer_history to just what's known up through this step
            partial_debtor_hist = self._partial_debtor_history_up_to(neg_result.get("dialog", []), debtor_idx)

            step_r, _ = compute_step_reward(
                prev_debtor_offer=prev_debtor_offer,
                new_debtor_offer=new_debtor_offer,
                debtor_offer_history=partial_debtor_hist,
                initial_gap=initial_gap,
                debtor_message=debtor_msg,
                observer_breakdown_risk=obs_breakdown,
            )

            # next_state = state_vec from the next creditor turn record (or this one if last)
            if i + 1 < n_records:
                next_state_vec = self.turn_records[i + 1]["state_vec"]
            else:
                # Episode ended — use a degraded copy of this state (model.update_model
                # will mark done=True and add final bonus regardless)
                next_state_vec = rec["state_vec"]

            done = (i == n_records - 1)
            self.emotion_model.record_step(
                state=rec["state_vec"],
                action=rec["action"],
                reward=step_r,
                next_state=next_state_vec,
                done=done,
                debtor_initial=self.debtor_initial,
            )

    @staticmethod
    def _most_recent_debtor_offer_before(dialog: List[Dict[str, Any]], idx: int) -> Optional[int]:
        for j in range(idx - 1, -1, -1):
            entry = dialog[j]
            if entry.get("speaker") == "buyer" and entry.get("requested_days") is not None:
                return int(entry["requested_days"])
        return None

    @staticmethod
    def _partial_debtor_history_up_to(dialog: List[Dict[str, Any]], up_to_idx: int) -> List[int]:
        hist = []
        for j in range(min(up_to_idx + 1, len(dialog))):
            entry = dialog[j]
            if entry.get("speaker") == "buyer" and entry.get("requested_days") is not None:
                hist.append(int(entry["requested_days"]))
        return hist
