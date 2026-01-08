#!/usr/bin/env python3
import math
import os
import threading
from typing import Optional, Tuple

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from geometry_msgs.msg import PoseStamped, TransformStamped
from sensor_msgs.msg import CameraInfo
from sensor_msgs.msg import Image as RosImage
from cv_bridge import CvBridge
import tf2_ros

import torch
import cv2

from huggingface_hub import snapshot_download
from depth_anything_3.api import DepthAnything3


MODEL_ID_MAP = {
    # user-facing option -> HF repo_id
    "DA3-SMALL": "depth-anything/da3-small",
    "DA3-BASE": "depth-anything/da3-base",
    "DA3METRIC-LARGE": "depth-anything/da3metric-large",
}
POSE_CAPABLE_MODELS = {"DA3-SMALL", "DA3-BASE"}


def _safe_model_dir(cache_dir: str, hf_repo_id: str) -> str:
    # e.g. depth-anything/da3-small -> depth-anything__da3-small
    safe = hf_repo_id.replace("/", "__")
    return os.path.join(cache_dir, safe)


def _ensure_local_snapshot(hf_repo_id: str, local_dir: str, force_download: bool = False) -> str:
    """
    Ensures the model repo is available at local_dir.
    If already present (and not force_download), reuse.
    """
    os.makedirs(local_dir, exist_ok=True)

    # Heuristic: if typical files exist, assume it's already downloaded
    # (DA3 uses config.json + model.safetensors)
    config_path = os.path.join(local_dir, "config.json")
    weights_path = os.path.join(local_dir, "model.safetensors")
    if (not force_download) and os.path.isfile(config_path) and os.path.isfile(weights_path):
        return local_dir

    # Download snapshot into local_dir
    # huggingface_hub versions differ slightly; keep it compatible.
    try:
        snapshot_download(
            repo_id=hf_repo_id,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            resume_download=not force_download,
        )
    except TypeError:
        snapshot_download(
            repo_id=hf_repo_id,
            local_dir=local_dir,
        )

    return local_dir


class DA3DepthNode(Node):
    def __init__(self):
        super().__init__("da3_depth_node")

        # Parameters
        self.declare_parameter("input_topic", "/camera/color/image_raw")
        self.declare_parameter("output_topic", "/depth_estimation/depth")
        self.declare_parameter("fps", 2.0)

        # Model selection
        self.declare_parameter("model_variant", "DA3-SMALL")  # DA3-SMALL | DA3-BASE | DA3METRIC-LARGE
        self.declare_parameter("device", "auto")  # auto | cuda | cpu
        self.declare_parameter("model_cache_dir", os.path.expanduser("~/.cache/da3_models"))
        self.declare_parameter("force_download", False)

        # Inference knobs
        self.declare_parameter("process_res", 504)  # DA3 inference default is 504
        self.declare_parameter("process_res_method", "upper_bound_resize")  # DA3 inference default
        self.declare_parameter("publish_encoding", "32FC1")  # 32FC1 recommended for float32 depth
        self.declare_parameter("use_ray_pose", True)
        self.declare_parameter("pose_topic", "/depth_estimation/pose")
        self.declare_parameter("camera_info_topic", "/depth_estimation/camera_info")
        self.declare_parameter("pose_frame_id", "world")
        self.declare_parameter("tf_publish", True)
        self.declare_parameter("tf_child_frame_id", "")

        self.input_topic = self.get_parameter("input_topic").get_parameter_value().string_value
        self.output_topic = self.get_parameter("output_topic").get_parameter_value().string_value
        self.fps = float(self.get_parameter("fps").get_parameter_value().double_value)

        self.model_variant = self.get_parameter("model_variant").get_parameter_value().string_value
        self.device_opt = self.get_parameter("device").get_parameter_value().string_value
        self.model_cache_dir = self.get_parameter("model_cache_dir").get_parameter_value().string_value
        self.force_download = self.get_parameter("force_download").get_parameter_value().bool_value

        self.process_res = int(self.get_parameter("process_res").get_parameter_value().integer_value)
        self.process_res_method = self.get_parameter("process_res_method").get_parameter_value().string_value
        self.publish_encoding = self.get_parameter("publish_encoding").get_parameter_value().string_value
        self.use_ray_pose = self.get_parameter("use_ray_pose").get_parameter_value().bool_value
        self.pose_topic = self.get_parameter("pose_topic").get_parameter_value().string_value
        self.camera_info_topic = (
            self.get_parameter("camera_info_topic").get_parameter_value().string_value
        )
        self.pose_frame_id = self.get_parameter("pose_frame_id").get_parameter_value().string_value
        self.tf_publish = self.get_parameter("tf_publish").get_parameter_value().bool_value
        self.tf_child_frame_id = (
            self.get_parameter("tf_child_frame_id").get_parameter_value().string_value
        )

        if self.model_variant not in MODEL_ID_MAP:
            raise ValueError(
                f"Invalid model_variant={self.model_variant}. Choose one of {list(MODEL_ID_MAP.keys())}"
            )
        self.pose_capable = self.model_variant in POSE_CAPABLE_MODELS
        if not self.pose_capable and self.use_ray_pose:
            self.get_logger().warn(
                f"Model {self.model_variant} does not support ray pose; publishing depth only."
            )
            self.use_ray_pose = False

        # ROS I/O
        self.bridge = CvBridge()
        self.sub = self.create_subscription(
            RosImage,
            self.input_topic,
            self._on_image,
            qos_profile_sensor_data,
        )
        self.pub = self.create_publisher(RosImage, self.output_topic, qos_profile_sensor_data)
        self.pose_pub = None
        self.camera_info_pub = None
        self.tf_broadcaster = None
        if self.pose_capable:
            self.pose_pub = self.create_publisher(
                PoseStamped, self.pose_topic, qos_profile_sensor_data
            )
            self.camera_info_pub = self.create_publisher(
                CameraInfo, self.camera_info_topic, qos_profile_sensor_data
            )
            if self.tf_publish:
                self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        # State
        self._latest_msg: Optional[RosImage] = None
        self._lock = threading.Lock()
        self._busy = False
        self._warned_pose_unavailable = False
        self._warned_missing_tf_child = False

        # Load model
        self.device = self._resolve_device(self.device_opt)
        self.model = self._load_model(self.model_variant, self.device)

        # Timer for fixed-rate inference
        period = 1.0 / max(self.fps, 1e-6)
        self.timer = self.create_timer(period, self._on_timer)

        self.get_logger().info(
            f"DA3DepthNode started. input={self.input_topic}, output={self.output_topic}, "
            f"fps={self.fps}, model={self.model_variant}, device={self.device}, "
            f"pose_publish={'on' if self.pose_capable else 'off'}, use_ray_pose={self.use_ray_pose}, "
            f"tf_publish={'on' if self.tf_broadcaster is not None else 'off'}"
        )

    def _resolve_device(self, device_opt: str) -> torch.device:
        device_opt = (device_opt or "auto").lower()
        if device_opt == "cuda":
            return torch.device("cuda")
        if device_opt == "cpu":
            return torch.device("cpu")
        # auto
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _load_model(self, model_variant: str, device: torch.device) -> DepthAnything3:
        hf_repo_id = MODEL_ID_MAP[model_variant]
        local_dir = _safe_model_dir(self.model_cache_dir, hf_repo_id)
        local_dir = _ensure_local_snapshot(hf_repo_id, local_dir, force_download=self.force_download)

        self.get_logger().info(f"Loading DA3 model from local_dir={local_dir} (repo_id={hf_repo_id})")

        # Load from local snapshot directory
        model = DepthAnything3.from_pretrained(local_dir)
        model = model.to(device=device)
        model.eval()

        # Some implementations keep a separate device field; set it explicitly for safety.
        try:
            model.device = device
        except Exception:
            pass

        return model

    def _on_image(self, msg: RosImage) -> None:
        with self._lock:
            self._latest_msg = msg

    def _on_timer(self) -> None:
        if self._busy:
            return

        with self._lock:
            msg = self._latest_msg

        if msg is None:
            return

        self._busy = True
        threading.Thread(target=self._infer_and_publish, args=(msg,), daemon=True).start()

    def _infer_and_publish(self, msg: RosImage) -> None:
        try:
            # Convert ROS Image -> BGR (OpenCV)
            cv_bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

            # Convert to RGB numpy array (H,W,3) uint8
            cv_rgb = cv2.cvtColor(cv_bgr, cv2.COLOR_BGR2RGB)

            # Run DA3 inference (single image -> N=1)
            use_ray_pose = self.use_ray_pose if self.pose_capable else False
            prediction = self.model.inference(
                [cv_rgb],
                export_dir=None,                 # no filesystem export
                export_format="mini_npz",         # irrelevant when export_dir is None
                process_res=self.process_res,
                process_res_method=self.process_res_method,
                use_ray_pose=use_ray_pose,
            )

            depth = prediction.depth  # expected [N,H,W] float32
            if isinstance(depth, np.ndarray):
                depth_map = depth[0]
            else:
                # very defensive fallback
                depth_map = np.asarray(depth)[0]

            depth_map = depth_map.astype(np.float32)  # ensure float32 for 32FC1

            out = RosImage()
            out.header = msg.header
            out.height = int(depth_map.shape[0])
            out.width = int(depth_map.shape[1])
            out.encoding = self.publish_encoding  # "32FC1"
            out.is_bigendian = 0
            out.step = out.width * 4  # float32 -> 4 bytes
            out.data = depth_map.tobytes()

            self.pub.publish(out)

            if self.pose_pub is not None and self.camera_info_pub is not None:
                extrinsics = prediction.extrinsics
                intrinsics = prediction.intrinsics
                if extrinsics is not None and intrinsics is not None:
                    c2w = self._extract_c2w(extrinsics[0], use_ray_pose)
                    pose_msg = self._build_pose_msg(c2w, msg)
                    info_msg = self._build_camera_info(intrinsics[0], depth_map.shape, msg)
                    self.pose_pub.publish(pose_msg)
                    self.camera_info_pub.publish(info_msg)
                    self._maybe_broadcast_tf(c2w, msg)
                elif not self._warned_pose_unavailable:
                    self.get_logger().warn(
                        "Pose or intrinsics not available; publishing depth only."
                    )
                    self._warned_pose_unavailable = True

        except Exception as e:
            self.get_logger().error(f"Inference failed: {e}")
        finally:
            self._busy = False

    def _extract_c2w(self, extr: np.ndarray, use_ray_pose: bool) -> np.ndarray:
        if extr.shape == (4, 4):
            extr = extr[:3, :]
        if use_ray_pose:
            return extr
        # w2c -> c2w
        R = extr[:3, :3]
        t = extr[:3, 3]
        R_t = R.T
        t_c2w = -R_t @ t
        return np.concatenate([R_t, t_c2w[:, None]], axis=1)

    def _build_pose_msg(self, c2w: np.ndarray, msg: RosImage) -> PoseStamped:
        pose = PoseStamped()
        pose.header.stamp = msg.header.stamp
        pose.header.frame_id = self.pose_frame_id
        t, quat = self._pose_from_c2w(c2w)
        pose.pose.position.x = float(t[0])
        pose.pose.position.y = float(t[1])
        pose.pose.position.z = float(t[2])
        pose.pose.orientation.x = quat[0]
        pose.pose.orientation.y = quat[1]
        pose.pose.orientation.z = quat[2]
        pose.pose.orientation.w = quat[3]
        return pose

    def _build_camera_info(
        self, intr: np.ndarray, shape: Tuple[int, int], msg: RosImage
    ) -> CameraInfo:
        height, width = int(shape[0]), int(shape[1])
        info = CameraInfo()
        info.header.stamp = msg.header.stamp
        info.header.frame_id = msg.header.frame_id
        info.height = height
        info.width = width
        info.distortion_model = "plumb_bob"
        info.d = [0.0, 0.0, 0.0, 0.0, 0.0]
        info.k = intr.reshape(-1).astype(float).tolist()
        info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        info.p = [
            float(intr[0, 0]),
            0.0,
            float(intr[0, 2]),
            0.0,
            0.0,
            float(intr[1, 1]),
            float(intr[1, 2]),
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
        ]
        return info

    def _maybe_broadcast_tf(self, c2w: np.ndarray, msg: RosImage) -> None:
        if self.tf_broadcaster is None:
            return
        child_frame_id = self.tf_child_frame_id or msg.header.frame_id
        if not child_frame_id:
            if not self._warned_missing_tf_child:
                self.get_logger().warn(
                    "TF child_frame_id is empty; set tf_child_frame_id or provide frame_id in input."
                )
                self._warned_missing_tf_child = True
            return

        t, quat = self._pose_from_c2w(c2w)
        tf_msg = TransformStamped()
        tf_msg.header.stamp = msg.header.stamp
        tf_msg.header.frame_id = self.pose_frame_id
        tf_msg.child_frame_id = child_frame_id
        tf_msg.transform.translation.x = float(t[0])
        tf_msg.transform.translation.y = float(t[1])
        tf_msg.transform.translation.z = float(t[2])
        tf_msg.transform.rotation.x = quat[0]
        tf_msg.transform.rotation.y = quat[1]
        tf_msg.transform.rotation.z = quat[2]
        tf_msg.transform.rotation.w = quat[3]
        self.tf_broadcaster.sendTransform(tf_msg)

    def _pose_from_c2w(self, c2w: np.ndarray) -> Tuple[np.ndarray, Tuple[float, float, float, float]]:
        R = c2w[:3, :3]
        t = c2w[:3, 3]
        quat = self._rotmat_to_quat(R)
        return t, quat

    def _rotmat_to_quat(self, R: np.ndarray) -> Tuple[float, float, float, float]:
        trace = float(R[0, 0] + R[1, 1] + R[2, 2])
        if trace > 0.0:
            s = 0.5 / math.sqrt(trace + 1.0)
            qw = 0.25 / s
            qx = (R[2, 1] - R[1, 2]) * s
            qy = (R[0, 2] - R[2, 0]) * s
            qz = (R[1, 0] - R[0, 1]) * s
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2.0 * math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            qw = (R[2, 1] - R[1, 2]) / s
            qx = 0.25 * s
            qy = (R[0, 1] + R[1, 0]) / s
            qz = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            qw = (R[0, 2] - R[2, 0]) / s
            qx = (R[0, 1] + R[1, 0]) / s
            qy = 0.25 * s
            qz = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            qw = (R[1, 0] - R[0, 1]) / s
            qx = (R[0, 2] + R[2, 0]) / s
            qy = (R[1, 2] + R[2, 1]) / s
            qz = 0.25 * s
        return float(qx), float(qy), float(qz), float(qw)


def main():
    rclpy.init()
    node = DA3DepthNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
