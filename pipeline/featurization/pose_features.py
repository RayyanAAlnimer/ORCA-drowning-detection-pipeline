from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any

import numpy as np

# COCO keypoint indices (Ultralytics YOLO pose uses COCO order)
# 0:nose 1:l_eye 2:r_eye 3:l_ear 4:r_ear 5:l_sh 6:r_sh 7:l_el 8:r_el
# 9:l_wr 10:r_wr 11:l_hip 12:r_hip 13:l_kn 14:r_kn 15:l_an 16:r_an
L_HIP, R_HIP = 11, 12
L_SHOULDER, R_SHOULDER = 5, 6


@dataclass
class FeatureBatch:
    X: np.ndarray          # (T, D) float32
    mask: np.ndarray       # (T,) float32
    meta: Dict[str, Any]   # debug info


class PoseFeatureExtractor:
    """
    Converts a Stage-3 buffered sequence of FrameFeature objects into
    fixed-size model-ready tensors for Stage 5.
    """
    def __init__(
        self,
        num_keypoints: int = 17,
        use_velocity: bool = True,
        include_bbox: bool = True,
        include_conf: bool = True,
        frame_size: Optional[Tuple[int, int]] = None,  # (w, h) for bbox normalization
    ):
        self.K = int(num_keypoints)
        self.use_velocity = bool(use_velocity)
        self.include_bbox = bool(include_bbox)
        self.include_conf = bool(include_conf)
        self.frame_size = frame_size  # (w, h) or None

    def _safe_kpts_xy(self, kpts_xyc) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns:
          xy: (K, 2) float32
          kc: (K,) float32 keypoint conf
        If missing/invalid, returns zeros.
        """
        xy = np.zeros((self.K, 2), dtype=np.float32)
        kc = np.zeros((self.K,), dtype=np.float32)

        if kpts_xyc is None:
            return xy, kc

        arr = np.asarray(kpts_xyc, dtype=np.float32)
        # Expect (K,3). If flattened, try reshape.
        if arr.ndim == 1 and arr.size == self.K * 3:
            arr = arr.reshape(self.K, 3)

        if arr.ndim != 2 or arr.shape[0] != self.K or arr.shape[1] < 2:
            return xy, kc

        xy[:, 0] = arr[:, 0]
        xy[:, 1] = arr[:, 1]
        if arr.shape[1] >= 3:
            kc[:] = arr[:, 2]
        return xy, kc

    def _bbox_to_cxcywh(self, bbox_xyxy: List[float]) -> np.ndarray:
        x1, y1, x2, y2 = [float(v) for v in bbox_xyxy]
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)
        w = max(0.0, x2 - x1)
        h = max(0.0, y2 - y1)
        v = np.array([cx, cy, w, h], dtype=np.float32)

        # normalize by frame size if provided
        if self.frame_size is not None:
            fw, fh = self.frame_size
            if fw > 0 and fh > 0:
                v[0] /= fw
                v[2] /= fw
                v[1] /= fh
                v[3] /= fh
        return v

    def _choose_center_and_scale(
        self,
        xy: np.ndarray,
        bbox_cxcywh: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """
        Center: mid-hip if available else bbox center.
        Scale: shoulder-to-hip distance if available else bbox h (or 1).
        """
        # bbox center
        bbox_center = np.array([bbox_cxcywh[0], bbox_cxcywh[1]], dtype=np.float32)

        # mid-hip
        lh, rh = xy[L_HIP], xy[R_HIP]
        has_hips = not (np.all(lh == 0) or np.all(rh == 0))
        if has_hips:
            center = 0.5 * (lh + rh)
        else:
            center = bbox_center

        # scale: torso length using mid-shoulder to mid-hip
        ls, rs = xy[L_SHOULDER], xy[R_SHOULDER]
        has_shoulders = not (np.all(ls == 0) or np.all(rs == 0))
        if has_shoulders and has_hips:
            mid_sh = 0.5 * (ls + rs)
            mid_hip = 0.5 * (lh + rh)
            scale = float(np.linalg.norm(mid_sh - mid_hip))
        else:
            # fallback: bbox height (already normalized if frame_size provided)
            scale = float(max(bbox_cxcywh[3], 1e-3))

        scale = max(scale, 1e-3)
        return center, scale

    def extract(self, seq: List[Any]) -> FeatureBatch:
        """
        seq: list of FrameFeature from Stage 3
        Returns: FeatureBatch(X, mask, meta)
        """
        T = len(seq)
        if T == 0:
            return FeatureBatch(
                X=np.zeros((0, 0), dtype=np.float32),
                mask=np.zeros((0,), dtype=np.float32),
                meta={"reason": "empty_seq"},
            )

        feats = []
        mask = np.ones((T,), dtype=np.float32)

        prev_bbox = None
        prev_prev_bbox = None
        recent_speeds: List[float] = []
        stationary_count = 0

        for t, ff in enumerate(seq):
            bbox = self._bbox_to_cxcywh(ff.bbox_xyxy)
            xy, kc = self._safe_kpts_xy(getattr(ff, "kpts_xyc", None))

            center, scale = self._choose_center_and_scale(xy, bbox)
            lh, rh = xy[L_HIP], xy[R_HIP]
            ls, rs = xy[L_SHOULDER], xy[R_SHOULDER]
            has_hips = not (np.all(lh == 0) or np.all(rh == 0))
            has_shoulders = not (np.all(ls == 0) or np.all(rs == 0))

            # Normalize pose: (x,y)-> (x-center)/scale
            xy_norm = (xy - center[None, :]) / scale  # (K,2)
            pose_flat = xy_norm.reshape(-1).astype(np.float32)  # (2K,)

            # Optional stats
            kpt_conf_mean = float(kc.mean()) if kc is not None else 0.0

            f_parts = [pose_flat]

            if self.include_bbox:
                f_parts.append(bbox)  # (4,)

            if self.include_conf:
                conf = float(getattr(ff, "conf", 0.0))
                f_parts.append(np.array([conf, kpt_conf_mean], dtype=np.float32))  # (2,)

            if self.use_velocity:
                if prev_bbox is None:
                    vel = np.zeros((2,), dtype=np.float32)
                else:
                    vel = (bbox[:2] - prev_bbox[:2]).astype(np.float32)  # Δcx, Δcy
                f_parts.append(vel)
            else:
                vel = np.zeros((2,), dtype=np.float32)

            # Speed captures motion intensity; distress can appear as low or erratic motion.
            speed = float(np.linalg.norm(vel))

            # Acceleration captures sudden starts/stops or bursts in motion.
            if prev_bbox is not None and prev_prev_bbox is not None:
                prev_vel = (prev_bbox[:2] - prev_prev_bbox[:2]).astype(np.float32)
                accel = float(np.linalg.norm(vel - prev_vel))
            else:
                accel = 0.0

            recent_speeds.append(speed)
            if len(recent_speeds) > 5:
                recent_speeds.pop(0)

            # Motion variance summarizes short-term movement stability or chaos.
            motion_var = float(np.var(recent_speeds)) if len(recent_speeds) > 1 else 0.0

            # Vertical velocity highlights bobbing or unstable up/down movement in water.
            vy = float(vel[1])

            # Area change approximates submersion cues as the visible body shrinks or grows.
            area = float(bbox[2] * bbox[3])
            if prev_bbox is not None:
                prev_area = float(prev_bbox[2] * prev_bbox[3])
                area_change = area - prev_area
            else:
                area_change = 0.0

            # Visibility ratio tracks partial disappearance when keypoints drop out.
            visible_kpts = int(np.sum(kc > 0.2))
            visibility_ratio = float(visible_kpts / float(self.K))

            # Body angle estimates torso orientation using shoulders and hips.
            if has_shoulders and has_hips:
                mid_sh = 0.5 * (ls + rs)
                mid_hip = 0.5 * (lh + rh)
                dx = float(mid_sh[0] - mid_hip[0])
                dy = float(mid_sh[1] - mid_hip[1])
                body_angle = float(np.arctan2(dy, dx))
            else:
                body_angle = 0.0

            # Stationary duration counts consecutive low-motion frames.
            if speed < 0.01:
                stationary_count += 1
            else:
                stationary_count = 0
            stationary_duration = float(stationary_count)

            f_parts.append(
                np.array(
                    [
                        speed,
                        accel,
                        motion_var,
                        vy,
                        area_change,
                        visibility_ratio,
                        body_angle,
                        stationary_duration,
                    ],
                    dtype=np.float32,
                )
            )

            feats.append(np.concatenate(f_parts, axis=0))
            prev_prev_bbox = prev_bbox
            prev_bbox = bbox

        X = np.stack(feats, axis=0).astype(np.float32)  # (T, D)

        meta = {
            "T": T,
            "D": int(X.shape[1]),
            "frame_size": self.frame_size,
            "use_velocity": self.use_velocity,
            "include_bbox": self.include_bbox,
            "include_conf": self.include_conf,
        }
        return FeatureBatch(X=X, mask=mask, meta=meta)
