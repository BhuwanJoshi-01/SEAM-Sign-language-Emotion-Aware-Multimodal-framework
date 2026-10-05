"""Both hands, in three dimensions, for every frame of a clip.

`seam.perception.tasks_api` keeps MediaPipe's *image* landmarks: x and y in the picture and
a relative depth. For driving an avatar's fingers the hand model's *world* landmarks are
the better input - 21 points in metres, in a frame centred on the hand, whose depth is
predicted as part of the hand's shape instead of being read off pixel size. This module
runs the same hand model once more over a clip's frames and keeps those.

Which detected hand is the signer's left is decided exactly as in `tasks_api`: by which of
the body's own wrists it is nearer (`tasks_api.assign_hands`).
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from seam.perception import tasks_api as T

SIDES = ("left", "right")


@dataclass(frozen=True, slots=True)
class HandTracks:
    """Per-frame hand points in the camera's frame: x right, y down, z away, metres."""

    left: list[np.ndarray | None]
    right: list[np.ndarray | None]

    def side(self, name: str) -> list[np.ndarray | None]:
        return self.left if name == "left" else self.right

    def coverage(self) -> dict[str, float]:
        n = max(len(self.left), 1)
        return {s: round(sum(p is not None for p in self.side(s)) / n, 4) for s in SIDES}


def track(frames_rgb: Iterable[np.ndarray], fps: float, *, delegate: str = "CPU") -> HandTracks:
    """Track both hands through a sequence of RGB frames.

    One hand graph asked for two hands and one pose graph, both in video mode so a hand
    that was found is followed instead of re-detected. Frames must arrive in order.
    """
    import mediapipe as mp
    from mediapipe.tasks.python import vision

    task = T.landmarker(delegate=delegate)
    hand = vision.HandLandmarker.create_from_options(
        vision.HandLandmarkerOptions(
            base_options=T._base_options(task._resolve("hand"), delegate),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.3,
            min_hand_presence_confidence=0.3,
            min_tracking_confidence=0.3,
        )
    )
    pose = vision.PoseLandmarker.create_from_options(
        vision.PoseLandmarkerOptions(
            base_options=T._base_options(task._resolve("pose"), delegate),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
        )
    )
    left: list[np.ndarray | None] = []
    right: list[np.ndarray | None] = []
    step_ms = 1000.0 / fps if fps > 0 else 40.0
    try:
        for i, rgb in enumerate(frames_rgb):
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
            ts = round(i * step_ms)
            found = hand.detect_for_video(image, ts)
            body = pose.detect_for_video(image, ts)
            slots = assign(found, body)
            world = getattr(found, "hand_world_landmarks", None) or []
            for name, store in (("left_hand", left), ("right_hand", right)):
                index = slots.get(name)
                if index is None or index >= len(world):
                    store.append(None)
                else:
                    store.append(
                        np.array([[p.x, p.y, p.z] for p in world[index]], dtype=np.float64)
                    )
    finally:
        hand.close()
        pose.close()
    return HandTracks(left=left, right=right)


def assign(hand_result: object, pose_result: object) -> dict[str, int]:
    """Slot each detected hand by the body's wrists, from raw MediaPipe results."""
    hands = getattr(hand_result, "hand_landmarks", None) or []
    sides = getattr(hand_result, "handedness", None) or []
    wrists = [np.array([h[0].x, h[0].y]) for h in hands]
    labels = [
        str(sides[i][0].category_name) if i < len(sides) and sides[i] else None
        for i in range(len(hands))
    ]
    body = getattr(pose_result, "pose_landmarks", None) or []
    pose_wrists = None
    if body:
        lw, rw = body[0][T.POSE_LEFT_WRIST], body[0][T.POSE_RIGHT_WRIST]
        pose_wrists = (np.array([lw.x, lw.y]), np.array([rw.x, rw.y]))
    return T.assign_hands(wrists, labels, pose_wrists)
