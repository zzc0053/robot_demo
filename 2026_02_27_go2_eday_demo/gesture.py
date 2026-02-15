import math
import numpy as np
import cv2
import mediapipe as mp


def _to_np(lm, w, h):
    return np.array([lm.x * w, lm.y * h, lm.z], dtype=float)


def angle_deg(a, b, c):
    """ ∠ABC """
    ba = a - b
    bc = c - b
    nba = ba / (np.linalg.norm(ba) + 1e-9)
    nbc = bc / (np.linalg.norm(bc) + 1e-9)
    cosang = np.clip(np.dot(nba, nbc), -1.0, 1.0)
    return math.degrees(math.acos(cosang))


def finger_mean_angle(pts, mcp, pip, dip, tip):
    a1 = angle_deg(pts[mcp], pts[pip], pts[dip])
    a2 = angle_deg(pts[pip], pts[dip], pts[tip])
    return (a1 + a2) / 2.0

def four_finger_angles(pts):
    return {
        "idx": finger_mean_angle(pts, 5, 6, 7, 8),
        "mid": finger_mean_angle(pts, 9, 10, 11, 12),
        "ring": finger_mean_angle(pts, 13, 14, 15, 16),
        "pinky": finger_mean_angle(pts, 17, 18, 19, 20),
    }

def pointing_direction(idx_mcp, idx_tip):
    """
    angle in image plane:
      0:right, 90:up, -90:down, 180/-180:left
    """
    v = idx_tip - idx_mcp
    dx, dy = v[0], v[1]
    ang = math.degrees(math.atan2(-dy, dx))
    return ang

class HandGestureRecognizer:
    def __init__(
        self,
        max_num_hands=2,
        model_complexity=0,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
        draw_landmarks=True,
        # 直/弯阈值（可调）
        idx_straight_th=155.0,
        other_straight_th=150.0,
        # 方向角阈值（更严格，避免水平误判）
        up_range=(45.0, 135.0),
        down_range=(-135.0, -45.0),
        # tip 比 pip 更远（避免折叠但角度“看着直”）
        tip_farther_ratio=1.05,
    ):
        self.draw_landmarks = draw_landmarks

        self.idx_straight_th = idx_straight_th
        self.other_straight_th = other_straight_th
        self.up_range = up_range
        self.down_range = down_range
        self.tip_farther_ratio = tip_farther_ratio

        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils

        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def close(self):
        self.hands.close()


    def _hand_pts(self, hand_landmarks, w, h):
        lms = hand_landmarks.landmark
        return [_to_np(lms[i], w, h) for i in range(21)]

    def _is_index_pointing_strict(self, pts):
        wrist = pts[0]
        # index mean angle
        idx_mean = finger_mean_angle(pts, 5, 6, 7, 8)
        idx_straight = idx_mean >= self.idx_straight_th

        # tip farther than pip from wrist
        d_tip = np.linalg.norm(pts[8][:2] - wrist[:2])
        d_pip = np.linalg.norm(pts[6][:2] - wrist[:2])
        farther = d_tip > d_pip * self.tip_farther_ratio

        # other fingers should NOT be straight
        mid_mean = finger_mean_angle(pts, 9, 10, 11, 12)
        ring_mean = finger_mean_angle(pts, 13, 14, 15, 16)
        pinky_mean = finger_mean_angle(pts, 17, 18, 19, 20)

        others_not_straight = (
            mid_mean < self.other_straight_th and
            ring_mean < self.other_straight_th and
            pinky_mean < self.other_straight_th
        )

        dbg = {
            "idx_mean": idx_mean,
            "mid_mean": mid_mean,
            "ring_mean": ring_mean,
            "pinky_mean": pinky_mean,
            "idx_straight": idx_straight,
            "farther": farther,
            "others_not_straight": others_not_straight,
        }
        return (idx_straight and farther and others_not_straight), dbg

    def is_fist(self, pts, fist_angle_th=140.0):
        dbg = four_finger_angles(pts)
        ok = all(v < fist_angle_th for v in dbg.values())
        return ok, dbg

    def is_open(self, pts, open_angle_th=155.0):
        dbg = four_finger_angles(pts)
        ok = all(v > open_angle_th for v in dbg.values())
        return ok, dbg

    def is_two_fingers_up(self, pts, straight_th=155.0, bent_th=140.0):
        dbg = four_finger_angles(pts)

        idx_ok = dbg["idx"] > straight_th
        mid_ok = dbg["mid"] > straight_th
        ring_ok = dbg["ring"] < bent_th
        pinky_ok = dbg["pinky"] < bent_th

        ok = idx_ok and mid_ok and ring_ok and pinky_ok
        return ok, dbg

    def detect(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        res = self.hands.process(rgb)

        if not res.multi_hand_landmarks:
            return None

        hand_lms = res.multi_hand_landmarks[0]

        if self.draw_landmarks:
            self.mp_draw.draw_landmarks(frame_bgr, hand_lms, self.mp_hands.HAND_CONNECTIONS)

        pts = self._hand_pts(hand_lms, w, h)

        fist_ok, fist_dbg = self.is_fist(pts)
        if fist_ok:
            return "fist"

        open_ok, _ = self.is_open(pts)
        if open_ok:
            return "open"

        two_ok, _ = self.is_two_fingers_up(pts)
        if two_ok:
            return "yeah"

        ok, dbg = self._is_index_pointing_strict(pts)
        if not ok:
            return None

        ang = pointing_direction(pts[5], pts[8])  # mcp->tip

        lo, hi = self.up_range
        if lo <= ang <= hi:
            return "index_up"

        lo, hi = self.down_range
        if lo <= ang <= hi:
            return "index_down"

        return None
