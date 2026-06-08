"""
R1Lite LeRobot 로봇 구현 — ROS2 토픽 기반 Async Inference 클라이언트

세 가지 모드:
  "full"    : 전체 observation (64-dim state, 4 cameras) / action (26-dim) — v2.1 서브키 형태
  "partial" : 부분 observation (14-dim 개별 서브키, 3 cameras) / action (14-dim 개별 서브키) — diffusion용
  "flat"    : flat observation (14-dim observation.state, 3 cameras) / action (14-dim action) — multi_task_dit (v3.0)용

observation.state 차원 구성
  full (64):    left_arm.pos(6) + left_arm.vel(6) + right_arm.pos(6) + right_arm.vel(6)
                + imu(10) + chassis.pos(3) + chassis.vel(3) + torso.pos(4) + torso.vel(4)
                + left_gripper(1) + right_gripper(1) + left_ee(7) + right_ee(7)
  partial (14): left_gripper(1) + right_gripper(1) + left_arm(6) + right_arm(6) — 개별 서브키
  flat (14):    left_arm(6) + right_arm(6) + left_gripper(1) + right_gripper(1)
                → "observation.state" 단일 키로 반환 (v3.0 flat 형태)

action 차원 구성
  full (26):    left_gripper(1) + right_gripper(1) + chassis.vel(6) + torso.vel(6)
                + left_arm(6) + right_arm(6)   (modality.json 순서, 서브키 형태)
  partial (14): left_gripper(1) + right_gripper(1) + left_arm(6) + right_arm(6) — 개별 서브키
  flat (14):    left_arm(6) + right_arm(6) + left_gripper(1) + right_gripper(1)
                → "action" 단일 키로 수신 (v3.0 flat 형태, 학습 dataset 순서와 일치)
"""

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, CompressedImage, Imu
from geometry_msgs.msg import PoseStamped, Twist

from lerobot.robots import Robot, RobotConfig
from lerobot.types import RobotAction, RobotObservation
# from lerobot.processor import RobotAction, RobotObservation


# ─────────────────────────────────────────────
# 상수: action 차원 슬라이스
# ─────────────────────────────────────────────
_FULL_ACT_SLICES = {
    "left_gripper":  (0, 1),
    "right_gripper": (1, 2),
    "chassis_vel":   (2, 8),
    "torso_vel":     (8, 14),
    "left_arm":      (14, 20),
    "right_arm":     (20, 26),
}

# partial 모드: left_gripper(0,1) + right_gripper(1,2) + left_arm(2,8) + right_arm(8,14)
_PARTIAL_ACT_SLICES = {
    "left_gripper":  (0, 1),
    "right_gripper": (1, 2),
    "left_arm":      (2, 8),
    "right_arm":     (8, 14),
}

# flat 모드 (v3.0): left_arm(0,6) + right_arm(6,12) + left_gripper(12,13) + right_gripper(13,14)
_FLAT_ACT_SLICES = {
    "left_arm":      (0, 6),
    "right_arm":     (6, 12),
    "left_gripper":  (12, 13),
    "right_gripper": (13, 14),
}

FLAT_STATE_DIM  = 14  # left_arm(6) + right_arm(6) + left_gripper(1) + right_gripper(1)
FLAT_ACTION_DIM = 14


# ─────────────────────────────────────────────
# TimeBuffer: 헤더 타임스탬프 기반 센서 동기화
# ─────────────────────────────────────────────
class _TimeBuffer:
    def __init__(self, maxlen: int = 200):
        self._buf: deque = deque(maxlen=maxlen)

    def push(self, t: float, data: object) -> None:
        self._buf.append((t, data))

    def latest(self) -> Optional[tuple]:
        return self._buf[-1] if self._buf else None

    def nearest(self, t_ref: float) -> Optional[tuple]:
        if not self._buf:
            return None
        return min(self._buf, key=lambda x: abs(x[0] - t_ref))


def _stamp_to_sec(stamp) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


# ─────────────────────────────────────────────
# feature key 생성 헬퍼 (observation_features용)
# ─────────────────────────────────────────────
def _vec_keys(prefix: str, n: int) -> dict:
    return {f"{prefix}.{i}": float for i in range(n)}


def _full_obs_features(head_h, head_w, wrist_h, wrist_w) -> dict:
    """full 모드 observation_features: 스칼라 64개 + 이미지 4개"""
    f = {}
    f.update(_vec_keys("left_arm.pos", 6))      # [0:6]
    f.update(_vec_keys("left_arm.vel", 6))      # [6:12]
    f.update(_vec_keys("right_arm.pos", 6))     # [12:18]
    f.update(_vec_keys("right_arm.vel", 6))     # [18:24]
    f.update(_vec_keys("imu", 10))              # [24:34]
    f.update(_vec_keys("chassis.pos", 3))       # [34:37]
    f.update(_vec_keys("chassis.vel", 3))       # [37:40]
    f.update(_vec_keys("torso.pos", 4))         # [40:44]
    f.update(_vec_keys("torso.vel", 4))         # [44:48]
    f["left_gripper.pos"] = float               # [48]
    f["right_gripper.pos"] = float              # [49]
    f.update(_vec_keys("left_ee", 7))           # [50:57]
    f.update(_vec_keys("right_ee", 7))          # [57:64]
    f["head_rgb"]        = (head_h,  head_w,  3)
    f["head_right_rgb"]  = (head_h,  head_w,  3)
    f["left_wrist_rgb"]  = (wrist_h, wrist_w, 3)
    f["right_wrist_rgb"] = (wrist_h, wrist_w, 3)
    return f


def _partial_obs_features(head_h, head_w, wrist_h, wrist_w) -> dict:
    """partial 모드 observation_features: 스칼라 14개 개별 서브키 + 이미지 3개 (diffusion용)"""
    f = {}
    f.update(_vec_keys("left_arm.pos", 6))      # [0:6]
    f.update(_vec_keys("right_arm.pos", 6))     # [6:12]
    f["left_gripper.pos"] = float               # [12]
    f["right_gripper.pos"] = float              # [13]
    f["head_rgb"]        = (head_h,  head_w,  3)
    f["left_wrist_rgb"]  = (wrist_h, wrist_w, 3)
    f["right_wrist_rgb"] = (wrist_h, wrist_w, 3)
    return f


def _flat_obs_features(head_h, head_w, wrist_h, wrist_w) -> dict:
    """flat 모드 observation_features: 스칼라 14개 개별 서브키 + 이미지 3개 (multi_task_dit용)
    hw_to_dataset_features가 float 값만 state 스칼라로 인식하므로 개별 float 키를 사용해야 함.
    obs 순서: left_arm.pos(6) + right_arm.pos(6) + left_gripper(1) + right_gripper(1) → observation.state(14)
    """
    f = {}
    f.update(_vec_keys("left_arm.pos", 6))      # [0:6]
    f.update(_vec_keys("right_arm.pos", 6))     # [6:12]
    f["left_gripper.pos"] = float               # [12]
    f["right_gripper.pos"] = float              # [13]
    f["head_rgb"]        = (head_h,  head_w,  3)
    f["left_wrist_rgb"]  = (wrist_h, wrist_w, 3)
    f["right_wrist_rgb"] = (wrist_h, wrist_w, 3)
    return f


def _full_action_features() -> dict:
    """full 모드 action_features: 스칼라 26개"""
    f = {}
    f["left_gripper"]  = float                  # [0]
    f["right_gripper"] = float                  # [1]
    f.update(_vec_keys("chassis.vel", 6))       # [2:8]
    f.update(_vec_keys("torso.vel", 6))         # [8:14]
    f.update(_vec_keys("left_arm", 6))          # [14:20]
    f.update(_vec_keys("right_arm", 6))         # [20:26]
    return f


def _partial_action_features() -> dict:
    """partial 모드 action_features: 스칼라 14개 개별 서브키 (diffusion용)
    순서: left_gripper(0) + right_gripper(1) + left_arm(2:8) + right_arm(8:14)
    """
    f = {}
    f["left_gripper"]  = float                  # [0]
    f["right_gripper"] = float                  # [1]
    f.update(_vec_keys("left_arm", 6))          # [2:8]
    f.update(_vec_keys("right_arm", 6))         # [8:14]
    return f


def _flat_action_features() -> dict:
    """flat 모드 action_features: 스칼라 14개 개별 서브키 (multi_task_dit / v3.0 dataset 순서)
    _action_tensor_to_action_dict가 키 수:텐서 원소 1:1 매핑이므로 개별 float 키를 사용해야 함.
    순서: left_arm(0:6) + right_arm(6:12) + left_gripper(12:13) + right_gripper(13:14)
    """
    f = {}
    f.update(_vec_keys("left_arm", 6))    # [0:6]
    f.update(_vec_keys("right_arm", 6))   # [6:12]
    f["left_gripper"]  = float            # [12]
    f["right_gripper"] = float            # [13]
    return f


# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
@RobotConfig.register_subclass("r1lite")
@dataclass
class R1LiteConfig(RobotConfig):
    # "full", "partial", or "flat"
    mode: str = "partial"

    # observation 토픽
    topic_arm_left_fb:      str = "/hdas/feedback_arm_left"
    topic_arm_right_fb:     str = "/hdas/feedback_arm_right"
    topic_chassis_fb:       str = "/hdas/feedback_chassis"
    topic_torso_fb:         str = "/hdas/feedback_torso"
    topic_gripper_left_fb:  str = "/hdas/feedback_gripper_left"
    topic_gripper_right_fb: str = "/hdas/feedback_gripper_right"
    topic_imu_fb:           str = "/hdas/imu_chassis"
    topic_ee_left:          str = "/motion_control/pose_ee_arm_left"
    topic_ee_right:         str = "/motion_control/pose_ee_arm_right"

    # 카메라 토픽
    topic_head_rgb:        str = "/hdas/camera_head/left_raw/image_raw_color/compressed"
    topic_head_right_rgb:  str = "/hdas/camera_head/right_raw/image_raw_color/compressed"
    topic_left_wrist_rgb:  str = "/hdas/camera_wrist_left/color/image_raw/compressed"
    topic_right_wrist_rgb: str = "/hdas/camera_wrist_right/color/image_raw/compressed"

    # action 토픽
    topic_cmd_arm_left:      str = "/motion_target/target_joint_state_arm_left"
    topic_cmd_arm_right:     str = "/motion_target/target_joint_state_arm_right"
    topic_cmd_gripper_left:  str = "/motion_target/target_position_gripper_left"
    topic_cmd_gripper_right: str = "/motion_target/target_position_gripper_right"
    topic_cmd_chassis:       str = "/motion_target/target_speed_chassis"
    topic_cmd_torso:         str = "/motion_target/target_speed_torso"

    # 이미지 해상도
    head_h:  int = 720
    head_w:  int = 1280
    wrist_h: int = 360
    wrist_w: int = 640

    # 센서 동기화 허용 시간차 (초)
    max_sync_dt: float = 1.0


# ─────────────────────────────────────────────
# Robot
# ─────────────────────────────────────────────
class R1LiteRobot(Robot):
    config_class = R1LiteConfig
    name = "r1lite"

    def __init__(self, config: R1LiteConfig):
        super().__init__(config)
        self.config = config

        if config.mode not in ("full", "partial", "flat"):
            raise ValueError(f"mode must be 'full', 'partial', or 'flat', got '{config.mode}'")

        self._connected = False
        self._spin_thread: Optional[threading.Thread] = None
        self.node: Optional[Node] = None

        # 센서 버퍼
        self._buf_arm_l   = _TimeBuffer()
        self._buf_arm_r   = _TimeBuffer()
        self._buf_chassis = _TimeBuffer()
        self._buf_torso   = _TimeBuffer()
        self._buf_grip_l  = _TimeBuffer()
        self._buf_grip_r  = _TimeBuffer()
        self._buf_imu     = _TimeBuffer()
        self._buf_ee_l    = _TimeBuffer()
        self._buf_ee_r    = _TimeBuffer()
        self._buf_head    = _TimeBuffer()
        self._buf_head_r  = _TimeBuffer()
        self._buf_wrist_l = _TimeBuffer()
        self._buf_wrist_r = _TimeBuffer()

    # ── features ──────────────────────────────
    @property
    def observation_features(self) -> dict:
        h, w = self.config.head_h, self.config.head_w
        wh, ww = self.config.wrist_h, self.config.wrist_w
        if self.config.mode == "full":
            return _full_obs_features(h, w, wh, ww)
        elif self.config.mode == "partial":
            return _partial_obs_features(h, w, wh, ww)
        else:  # flat
            return _flat_obs_features(h, w, wh, ww)

    @property
    def action_features(self) -> dict:
        if self.config.mode == "full":
            return _full_action_features()
        elif self.config.mode == "partial":
            return _partial_action_features()
        else:  # flat
            return _flat_action_features()

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def is_calibrated(self) -> bool:
        return True

    # ── 연결 / 해제 ───────────────────────────
    def connect(self, calibrate: bool = True) -> None:
        if self._connected:
            return

        rclpy.init()
        cfg = self.config
        self.node = Node("lerobot_r1lite")

        # subscriptions (공통: partial/flat/full 모두)
        self.node.create_subscription(JointState,      cfg.topic_arm_left_fb,      self._cb_arm_l,   10)
        self.node.create_subscription(JointState,      cfg.topic_arm_right_fb,     self._cb_arm_r,   10)
        self.node.create_subscription(JointState,      cfg.topic_gripper_left_fb,  self._cb_grip_l,  10)
        self.node.create_subscription(JointState,      cfg.topic_gripper_right_fb, self._cb_grip_r,  10)
        self.node.create_subscription(CompressedImage, cfg.topic_head_rgb,         self._cb_head,    10)
        self.node.create_subscription(CompressedImage, cfg.topic_left_wrist_rgb,   self._cb_wrist_l, 10)
        self.node.create_subscription(CompressedImage, cfg.topic_right_wrist_rgb,  self._cb_wrist_r, 10)

        # subscriptions (full 전용)
        if cfg.mode == "full":
            self.node.create_subscription(JointState,      cfg.topic_chassis_fb,     self._cb_chassis, 10)
            self.node.create_subscription(JointState,      cfg.topic_torso_fb,        self._cb_torso,   10)
            self.node.create_subscription(Imu,             cfg.topic_imu_fb,          self._cb_imu,     10)
            self.node.create_subscription(PoseStamped,     cfg.topic_ee_left,         self._cb_ee_l,    10)
            self.node.create_subscription(PoseStamped,     cfg.topic_ee_right,        self._cb_ee_r,    10)
            self.node.create_subscription(CompressedImage, cfg.topic_head_right_rgb,  self._cb_head_r,  10)

        # publishers (공통)
        self._pub_arm_l  = self.node.create_publisher(JointState, cfg.topic_cmd_arm_left,      10)
        self._pub_arm_r  = self.node.create_publisher(JointState, cfg.topic_cmd_arm_right,     10)
        self._pub_grip_l = self.node.create_publisher(JointState, cfg.topic_cmd_gripper_left,  10)
        self._pub_grip_r = self.node.create_publisher(JointState, cfg.topic_cmd_gripper_right, 10)

        # publishers (full 전용)
        if cfg.mode == "full":
            self._pub_chassis = self.node.create_publisher(Twist, cfg.topic_cmd_chassis, 10)
            self._pub_torso   = self.node.create_publisher(Twist, cfg.topic_cmd_torso,   10)

        # ROS2 spin → daemon 스레드
        self._spin_thread = threading.Thread(
            target=rclpy.spin, args=(self.node,), daemon=True, name="r1lite_ros2_spin"
        )
        self._spin_thread.start()
        self._connected = True

        # 첫 head_rgb 수신까지 대기 (최대 10초)
        deadline = time.time() + 10.0
        while self._buf_head.latest() is None and time.time() < deadline:
            time.sleep(0.05)
        if self._buf_head.latest() is None:
            raise RuntimeError("head_rgb 토픽 수신 실패 (10초 타임아웃). 토픽 발행 여부를 확인하세요.")

    def disconnect(self) -> None:
        if not self._connected:
            return
        self._connected = False
        self.node.destroy_node()
        rclpy.shutdown()
        if self._spin_thread:
            self._spin_thread.join(timeout=2.0)

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    # ── 콜백: 상태 ────────────────────────────
    def _cb_arm_l(self, msg: JointState):
        t = _stamp_to_sec(msg.header.stamp) if msg.header.stamp.sec else time.time()
        pos = np.array(msg.position[:6], np.float32) if len(msg.position) > 0 else np.zeros(6, np.float32)
        vel = np.array(msg.velocity[:6], np.float32) if len(msg.velocity) > 0 else np.zeros(6, np.float32)
        self._buf_arm_l.push(t, (pos, vel))

    def _cb_arm_r(self, msg: JointState):
        t = _stamp_to_sec(msg.header.stamp) if msg.header.stamp.sec else time.time()
        pos = np.array(msg.position[:6], np.float32) if len(msg.position) > 0 else np.zeros(6, np.float32)
        vel = np.array(msg.velocity[:6], np.float32) if len(msg.velocity) > 0 else np.zeros(6, np.float32)
        self._buf_arm_r.push(t, (pos, vel))

    def _cb_chassis(self, msg: JointState):
        t = _stamp_to_sec(msg.header.stamp) if msg.header.stamp.sec else time.time()
        pos = np.array(msg.position[:3], np.float32) if len(msg.position) > 0 else np.zeros(3, np.float32)
        vel = np.array(msg.velocity[:3], np.float32) if len(msg.velocity) > 0 else np.zeros(3, np.float32)
        self._buf_chassis.push(t, (pos, vel))

    def _cb_torso(self, msg: JointState):
        t = _stamp_to_sec(msg.header.stamp) if msg.header.stamp.sec else time.time()
        pos = np.array(msg.position[:4], np.float32) if len(msg.position) > 0 else np.zeros(4, np.float32)
        vel = np.array(msg.velocity[:4], np.float32) if len(msg.velocity) > 0 else np.zeros(4, np.float32)
        self._buf_torso.push(t, (pos, vel))

    def _cb_grip_l(self, msg: JointState):
        t = _stamp_to_sec(msg.header.stamp) if msg.header.stamp.sec else time.time()
        self._buf_grip_l.push(t, float(msg.position[0]) if len(msg.position) > 0 else 0.0)

    def _cb_grip_r(self, msg: JointState):
        t = _stamp_to_sec(msg.header.stamp) if msg.header.stamp.sec else time.time()
        self._buf_grip_r.push(t, float(msg.position[0]) if len(msg.position) > 0 else 0.0)

    def _cb_imu(self, msg: Imu):
        t = _stamp_to_sec(msg.header.stamp) if msg.header.stamp else time.time()
        data = np.array([
            msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w,
            msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z,
            msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z,
        ], np.float32)
        self._buf_imu.push(t, data)

    def _cb_ee_l(self, msg: PoseStamped):
        t = _stamp_to_sec(msg.header.stamp) if msg.header.stamp else time.time()
        p, q = msg.pose.position, msg.pose.orientation
        self._buf_ee_l.push(t, np.array([p.x, p.y, p.z, q.x, q.y, q.z, q.w], np.float32))

    def _cb_ee_r(self, msg: PoseStamped):
        t = _stamp_to_sec(msg.header.stamp) if msg.header.stamp else time.time()
        p, q = msg.pose.position, msg.pose.orientation
        self._buf_ee_r.push(t, np.array([p.x, p.y, p.z, q.x, q.y, q.z, q.w], np.float32))

    # ── 콜백: 이미지 ──────────────────────────
    def _decode_compressed(self, msg: CompressedImage, h: int, w: int) -> Optional[np.ndarray]:
        try:
            arr = np.frombuffer(msg.data, dtype=np.uint8)
            bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if bgr is None:
                return None
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            if rgb.shape[0] != h or rgb.shape[1] != w:
                rgb = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA)
            return rgb.astype(np.uint8)
        except Exception:
            return None

    def _cb_head(self, msg: CompressedImage):
        t = _stamp_to_sec(msg.header.stamp) if msg.header.stamp else time.time()
        img = self._decode_compressed(msg, self.config.head_h, self.config.head_w)
        if img is not None:
            self._buf_head.push(t, img)

    def _cb_head_r(self, msg: CompressedImage):
        t = _stamp_to_sec(msg.header.stamp) if msg.header.stamp else time.time()
        img = self._decode_compressed(msg, self.config.head_h, self.config.head_w)
        if img is not None:
            self._buf_head_r.push(t, img)

    def _cb_wrist_l(self, msg: CompressedImage):
        t = _stamp_to_sec(msg.header.stamp) if msg.header.stamp else time.time()
        img = self._decode_compressed(msg, self.config.wrist_h, self.config.wrist_w)
        if img is not None:
            self._buf_wrist_l.push(t, img)

    def _cb_wrist_r(self, msg: CompressedImage):
        t = _stamp_to_sec(msg.header.stamp) if msg.header.stamp else time.time()
        img = self._decode_compressed(msg, self.config.wrist_h, self.config.wrist_w)
        if img is not None:
            self._buf_wrist_r.push(t, img)

    # ── observation 구성 ──────────────────────
    def _get_near(self, buf: _TimeBuffer, t_ref: float) -> Optional[object]:
        item = buf.nearest(t_ref)
        if item is None:
            return None
        t, data = item
        return data if abs(t - t_ref) <= self.config.max_sync_dt else None

    def get_observation(self) -> RobotObservation:
        if not self._connected:
            raise RuntimeError("Robot is not connected.")

        # 헤드 카메라를 기준 타임스탬프로 사용
        head_item = self._buf_head.latest()
        if head_item is None:
            raise RuntimeError("head_rgb 미수신. 토픽 발행 여부를 확인하세요.")

        t_ref, head_img = head_item

        arm_l  = self._get_near(self._buf_arm_l,  t_ref)
        arm_r  = self._get_near(self._buf_arm_r,  t_ref)
        grip_l = self._get_near(self._buf_grip_l, t_ref)
        grip_r = self._get_near(self._buf_grip_r, t_ref)

        for name, val in [("arm_left", arm_l), ("arm_right", arm_r),
                          ("gripper_left", grip_l), ("gripper_right", grip_r)]:
            if val is None:
                raise RuntimeError(f"필수 토픽 미수신: {name}")

        wrist_l = self._get_near(self._buf_wrist_l, t_ref)
        wrist_r = self._get_near(self._buf_wrist_r, t_ref)
        if wrist_l is None or wrist_r is None:
            raise RuntimeError("wrist 카메라 토픽 미수신.")

        arm_l_pos, arm_l_vel = arm_l
        arm_r_pos, arm_r_vel = arm_r

        obs: RobotObservation = {}

        if self.config.mode == "partial":
            # diffusion용: 개별 서브키 반환
            for i, v in enumerate(arm_l_pos):   obs[f"left_arm.pos.{i}"]  = float(v)
            for i, v in enumerate(arm_r_pos):   obs[f"right_arm.pos.{i}"] = float(v)
            obs["left_gripper.pos"]  = float(grip_l)
            obs["right_gripper.pos"] = float(grip_r)
            obs["head_rgb"]        = head_img
            obs["left_wrist_rgb"]  = wrist_l
            obs["right_wrist_rgb"] = wrist_r

        elif self.config.mode == "flat":
            # multi_task_dit/v3.0용: 개별 서브키 반환
            # hw_to_dataset_features가 float 키들을 모아 observation.state(14) 를 자동 구성
            # 순서: left_arm.pos(6) + right_arm.pos(6) + left_gripper(1) + right_gripper(1)
            for i, v in enumerate(arm_l_pos):   obs[f"left_arm.pos.{i}"]  = float(v)
            for i, v in enumerate(arm_r_pos):   obs[f"right_arm.pos.{i}"] = float(v)
            obs["left_gripper.pos"]  = float(grip_l)
            obs["right_gripper.pos"] = float(grip_r)
            obs["head_rgb"]        = head_img
            obs["left_wrist_rgb"]  = wrist_l
            obs["right_wrist_rgb"] = wrist_r

        else:  # full
            chassis = self._get_near(self._buf_chassis, t_ref)
            torso   = self._get_near(self._buf_torso,   t_ref)
            imu     = self._get_near(self._buf_imu,     t_ref)
            ee_l    = self._get_near(self._buf_ee_l,    t_ref)
            ee_r    = self._get_near(self._buf_ee_r,    t_ref)
            _head_r = self._get_near(self._buf_head_r,  t_ref)
            head_r  = _head_r if _head_r is not None else head_img

            ch_pos, ch_vel = chassis if chassis is not None else (np.zeros(3), np.zeros(3))
            to_pos, to_vel = torso   if torso   is not None else (np.zeros(4), np.zeros(4))
            imu_d  = imu  if imu  is not None else np.zeros(10)
            ee_l_d = ee_l if ee_l is not None else np.zeros(7)
            ee_r_d = ee_r if ee_r is not None else np.zeros(7)

            for i, v in enumerate(arm_l_pos):  obs[f"left_arm.pos.{i}"]  = float(v)
            for i, v in enumerate(arm_l_vel):  obs[f"left_arm.vel.{i}"]  = float(v)
            for i, v in enumerate(arm_r_pos):  obs[f"right_arm.pos.{i}"] = float(v)
            for i, v in enumerate(arm_r_vel):  obs[f"right_arm.vel.{i}"] = float(v)
            for i, v in enumerate(imu_d):      obs[f"imu.{i}"]           = float(v)
            for i, v in enumerate(ch_pos):     obs[f"chassis.pos.{i}"]   = float(v)
            for i, v in enumerate(ch_vel):     obs[f"chassis.vel.{i}"]   = float(v)
            for i, v in enumerate(to_pos):     obs[f"torso.pos.{i}"]     = float(v)
            for i, v in enumerate(to_vel):     obs[f"torso.vel.{i}"]     = float(v)
            obs["left_gripper.pos"]  = float(grip_l)
            obs["right_gripper.pos"] = float(grip_r)
            for i, v in enumerate(ee_l_d):    obs[f"left_ee.{i}"]  = float(v)
            for i, v in enumerate(ee_r_d):    obs[f"right_ee.{i}"] = float(v)
            obs["head_rgb"]        = head_img
            obs["head_right_rgb"]  = head_r
            obs["left_wrist_rgb"]  = wrist_l
            obs["right_wrist_rgb"] = wrist_r

        return obs

    # ── action 발행 ───────────────────────────
    def _pub_joint_state(self, pub, positions, vel: float = 0.6) -> None:
        msg = JointState()
        msg.position = [float(p) for p in np.asarray(positions).reshape(-1)]
        msg.velocity = [vel] * len(msg.position)
        pub.publish(msg)

    def _pub_twist(self, pub, v) -> None:
        v = np.asarray(v, np.float32).reshape(-1)
        tw = Twist()
        if len(v) >= 1: tw.linear.x  = float(v[0])
        if len(v) >= 2: tw.linear.y  = float(v[1])
        if len(v) >= 3: tw.linear.z  = float(v[2])
        if len(v) >= 4: tw.angular.x = float(v[3])
        if len(v) >= 5: tw.angular.y = float(v[4])
        if len(v) >= 6: tw.angular.z = float(v[5])
        pub.publish(tw)

    def send_action(self, action: RobotAction) -> RobotAction:
        if not self._connected:
            raise RuntimeError("Robot is not connected.")

        if self.config.mode == "partial":
            # diffusion용: 개별 서브키 → _PARTIAL_ACT_SLICES
            # 순서: left_gripper(0) + right_gripper(1) + left_arm(2:8) + right_arm(8:14)
            keys = list(self.action_features.keys())
            vals = np.array([action[k] for k in keys], dtype=np.float32)
            s = _PARTIAL_ACT_SLICES
            self._pub_joint_state(self._pub_grip_l, vals[s["left_gripper"][0]:s["left_gripper"][1]])
            self._pub_joint_state(self._pub_grip_r, vals[s["right_gripper"][0]:s["right_gripper"][1]])
            self._pub_joint_state(self._pub_arm_l,  vals[s["left_arm"][0]:s["left_arm"][1]])
            self._pub_joint_state(self._pub_arm_r,  vals[s["right_arm"][0]:s["right_arm"][1]])

        elif self.config.mode == "flat":
            # multi_task_dit/v3.0용: 개별 서브키 → _FLAT_ACT_SLICES
            # 순서: left_arm(0:6) + right_arm(6:12) + left_gripper(12:13) + right_gripper(13:14)
            keys = list(self.action_features.keys())
            vals = np.array([action[k] for k in keys], dtype=np.float32)
            s = _FLAT_ACT_SLICES
            self._pub_joint_state(self._pub_arm_l,  vals[s["left_arm"][0]:s["left_arm"][1]])
            self._pub_joint_state(self._pub_arm_r,  vals[s["right_arm"][0]:s["right_arm"][1]])
            self._pub_joint_state(self._pub_grip_l, vals[s["left_gripper"][0]:s["left_gripper"][1]])
            self._pub_joint_state(self._pub_grip_r, vals[s["right_gripper"][0]:s["right_gripper"][1]])

        else:  # full
            keys = list(self.action_features.keys())
            vals = np.array([action[k] for k in keys], dtype=np.float32)
            s = _FULL_ACT_SLICES
            self._pub_joint_state(self._pub_grip_l,   vals[s["left_gripper"][0]:s["left_gripper"][1]])
            self._pub_joint_state(self._pub_grip_r,   vals[s["right_gripper"][0]:s["right_gripper"][1]])
            self._pub_joint_state(self._pub_arm_l,    vals[s["left_arm"][0]:s["left_arm"][1]])
            self._pub_joint_state(self._pub_arm_r,    vals[s["right_arm"][0]:s["right_arm"][1]])
            self._pub_twist(self._pub_chassis, vals[s["chassis_vel"][0]:s["chassis_vel"][1]])
            self._pub_twist(self._pub_torso,   vals[s["torso_vel"][0]:s["torso_vel"][1]])

        return action
