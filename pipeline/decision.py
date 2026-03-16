# pipeline/decision.py
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple
import threading
import time


class DecisionLevel(str, Enum):
    NORMAL = "normal"
    POSSIBLE = "possible"
    CONFIRMED = "confirmed"


@dataclass
class DecisionEvent:
    track_id: int
    level: DecisionLevel
    raw_prob: float
    smooth_prob: float
    ts: float  # monotonic time


@dataclass
class TrackDecisionState:
    # Smoothing
    smooth_prob: float = 0.0
    initialized: bool = False

    # Hysteresis state
    level: DecisionLevel = DecisionLevel.NORMAL

    # Hold-time bookkeeping (for confirming)
    above_confirm_since: Optional[float] = None

    # Cooldown bookkeeping (block repeated alerts)
    cooldown_until: float = 0.0

    # For housekeeping
    last_update_ts: float = 0.0


class DecisionManager:
    """
    Stage 6: turns per-track probabilities into stable alerts.

    Features:
      - EMA smoothing
      - Hysteresis (possible vs confirmed thresholds)
      - Hold-time (must stay above confirm threshold for X seconds)
      - Cooldown after confirmed alert
      - Thread-safe state
    """

    def __init__(
        self,
        ema_alpha: float = 0.30,
        possible_th: float = 0.60,
        confirm_th: float = 0.80,
        hold_seconds: float = 1.0,
        cooldown_seconds: float = 5.0,
        stale_track_seconds: float = 3.0,
    ):
        assert 0.0 < ema_alpha <= 1.0
        assert 0.0 <= possible_th < confirm_th <= 1.0
        assert hold_seconds >= 0.0
        assert cooldown_seconds >= 0.0
        assert stale_track_seconds > 0.0

        self.ema_alpha = ema_alpha
        self.possible_th = possible_th
        self.confirm_th = confirm_th
        self.hold_seconds = hold_seconds
        self.cooldown_seconds = cooldown_seconds
        self.stale_track_seconds = stale_track_seconds

        self._lock = threading.Lock()
        self._states: Dict[int, TrackDecisionState] = {}

    def update_batch(
        self,
        probs_by_track: Dict[int, float],
        ts: Optional[float] = None,
    ) -> List[DecisionEvent]:
        """
        Update decision state for all tracks that have new probs.
        Returns decision events (including confirmed).
        """
        now = ts if ts is not None else time.monotonic()
        events: List[DecisionEvent] = []

        with self._lock:
            # Update each track with new probability
            for tid, raw in probs_by_track.items():
                raw_clamped = float(max(0.0, min(1.0, raw)))
                st = self._states.get(tid)
                if st is None:
                    st = TrackDecisionState()
                    self._states[tid] = st

                # EMA smoothing
                if not st.initialized:
                    st.smooth_prob = raw_clamped
                    st.initialized = True
                else:
                    a = self.ema_alpha
                    st.smooth_prob = a * raw_clamped + (1.0 - a) * st.smooth_prob

                st.last_update_ts = now

                # Cooldown gate: while in cooldown, we still update smoothing,
                # but we do NOT escalate to CONFIRMED again.
                in_cooldown = now < st.cooldown_until

                prev_level = st.level
                new_level = prev_level

                # Hysteresis transitions:
                # - NORMAL -> POSSIBLE when smooth >= possible_th
                # - POSSIBLE -> NORMAL when smooth < possible_th (falls back)
                # - POSSIBLE -> CONFIRMED only if smooth >= confirm_th for hold_seconds,
                #   and NOT in cooldown
                # - CONFIRMED -> POSSIBLE when smooth < confirm_th (de-escalate),
                #   and -> NORMAL when smooth < possible_th
                sp = st.smooth_prob

                if prev_level == DecisionLevel.NORMAL:
                    if sp >= self.possible_th:
                        new_level = DecisionLevel.POSSIBLE
                        st.above_confirm_since = None  # reset

                elif prev_level == DecisionLevel.POSSIBLE:
                    if sp < self.possible_th:
                        new_level = DecisionLevel.NORMAL
                        st.above_confirm_since = None
                    else:
                        # candidate for confirm
                        if sp >= self.confirm_th and not in_cooldown:
                            if st.above_confirm_since is None:
                                st.above_confirm_since = now
                            elif (now - st.above_confirm_since) >= self.hold_seconds:
                                new_level = DecisionLevel.CONFIRMED
                        else:
                            st.above_confirm_since = None

                elif prev_level == DecisionLevel.CONFIRMED:
                    # Once confirmed, we allow it to de-escalate naturally
                    if sp < self.possible_th:
                        new_level = DecisionLevel.NORMAL
                        st.above_confirm_since = None
                    elif sp < self.confirm_th:
                        new_level = DecisionLevel.POSSIBLE
                        st.above_confirm_since = None
                    else:
                        # still high; keep confirmed
                        pass

                # Apply transition & emit events on change OR on confirmed
                st.level = new_level

                if new_level != prev_level:
                    events.append(
                        DecisionEvent(
                            track_id=tid,
                            level=new_level,
                            raw_prob=raw_clamped,
                            smooth_prob=st.smooth_prob,
                            ts=now,
                        )
                    )

                # If we just entered CONFIRMED, start cooldown immediately
                if prev_level != DecisionLevel.CONFIRMED and new_level == DecisionLevel.CONFIRMED:
                    st.cooldown_until = now + self.cooldown_seconds
                

            # Cleanup stale tracks (prevents memory growth when IDs disappear)
            stale_cutoff = now - self.stale_track_seconds
            to_delete = [tid for tid, st in self._states.items() if st.last_update_ts < stale_cutoff]
            for tid in to_delete:
                del self._states[tid]

        return events

    def snapshot(self) -> Dict[int, Tuple[DecisionLevel, float, float, float]]:
        """
        For UI overlay/debug: returns {track_id: (level, smooth_prob, last_update_ts, cooldown_until)}
        """
        with self._lock:
            out: Dict[int, Tuple[DecisionLevel, float, float, float]] = {}
            for tid, st in self._states.items():
                out[tid] = (st.level, st.smooth_prob, st.last_update_ts, st.cooldown_until)
            return out