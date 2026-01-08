#!/usr/bin/env python3
import os
import threading
from typing import Optional, Tuple

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import Image as RosImage
from cv_bridge import CvBridge

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

        if self.model_variant not in MODEL_ID_MAP:
            raise ValueError(
                f"Invalid model_variant={self.model_variant}. Choose one of {list(MODEL_ID_MAP.keys())}"
            )

        # ROS I/O
        self.bridge = CvBridge()
        self.sub = self.create_subscription(
            RosImage,
            self.input_topic,
            self._on_image,
            qos_profile_sensor_data,
        )
        self.pub = self.create_publisher(RosImage, self.output_topic, qos_profile_sensor_data)

        # State
        self._latest_msg: Optional[RosImage] = None
        self._lock = threading.Lock()
        self._busy = False

        # Load model
        self.device = self._resolve_device(self.device_opt)
        self.model = self._load_model(self.model_variant, self.device)

        # Timer for fixed-rate inference
        period = 1.0 / max(self.fps, 1e-6)
        self.timer = self.create_timer(period, self._on_timer)

        self.get_logger().info(
            f"DA3DepthNode started. input={self.input_topic}, output={self.output_topic}, "
            f"fps={self.fps}, model={self.model_variant}, device={self.device}"
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
            prediction = self.model.inference(
                [cv_rgb],
                export_dir=None,                 # no filesystem export
                export_format="mini_npz",         # irrelevant when export_dir is None
                process_res=self.process_res,
                process_res_method=self.process_res_method,
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

        except Exception as e:
            self.get_logger().error(f"Inference failed: {e}")
        finally:
            self._busy = False


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
