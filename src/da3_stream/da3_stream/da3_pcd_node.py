#!/usr/bin/env python3
import math
from typing import Optional, Tuple

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import CameraInfo
from sensor_msgs.msg import Image as RosImage
from sensor_msgs.msg import PointCloud2, PointField


class DA3PointCloudNode(Node):
    def __init__(self):
        super().__init__("da3_pcd_node")

        self.declare_parameter("depth_topic", "/depth_estimation/depth")
        self.declare_parameter("camera_info_topic", "/depth_estimation/camera_info")
        self.declare_parameter("pose_topic", "/depth_estimation/pose")
        self.declare_parameter("output_topic", "/depth_estimation/pointcloud")
        self.declare_parameter("output_frame_id", "world")
        self.declare_parameter("stride", 1)
        self.declare_parameter("min_depth", 0.0)
        self.declare_parameter("max_depth", 0.0)

        self.depth_topic = self.get_parameter("depth_topic").get_parameter_value().string_value
        self.camera_info_topic = (
            self.get_parameter("camera_info_topic").get_parameter_value().string_value
        )
        self.pose_topic = self.get_parameter("pose_topic").get_parameter_value().string_value
        self.output_topic = self.get_parameter("output_topic").get_parameter_value().string_value
        self.output_frame_id = (
            self.get_parameter("output_frame_id").get_parameter_value().string_value
        )
        self.stride = int(self.get_parameter("stride").get_parameter_value().integer_value)
        self.min_depth = float(self.get_parameter("min_depth").get_parameter_value().double_value)
        self.max_depth = float(self.get_parameter("max_depth").get_parameter_value().double_value)

        self._latest_pose: Optional[PoseStamped] = None
        self._latest_info: Optional[CameraInfo] = None
        self._warned_missing = False

        self.create_subscription(
            RosImage, self.depth_topic, self._on_depth, qos_profile_sensor_data
        )
        self.create_subscription(
            CameraInfo, self.camera_info_topic, self._on_camera_info, qos_profile_sensor_data
        )
        self.create_subscription(
            PoseStamped, self.pose_topic, self._on_pose, qos_profile_sensor_data
        )
        self.pub = self.create_publisher(PointCloud2, self.output_topic, qos_profile_sensor_data)

        self.get_logger().info(
            "DA3PointCloudNode started. "
            f"depth={self.depth_topic}, camera_info={self.camera_info_topic}, "
            f"pose={self.pose_topic}, output={self.output_topic}"
        )

    def _on_pose(self, msg: PoseStamped) -> None:
        self._latest_pose = msg

    def _on_camera_info(self, msg: CameraInfo) -> None:
        self._latest_info = msg

    def _on_depth(self, msg: RosImage) -> None:
        if self._latest_pose is None or self._latest_info is None:
            if not self._warned_missing:
                self.get_logger().warn(
                    "Waiting for pose and camera_info; pointcloud publish is paused."
                )
                self._warned_missing = True
            return

        depth = self._decode_depth(msg)
        if depth is None:
            return

        info = self._latest_info
        pose = self._latest_pose

        if info.width != depth.shape[1] or info.height != depth.shape[0]:
            self.get_logger().warn(
                "camera_info size does not match depth image; check resolution settings."
            )

        fx, fy, cx, cy = self._extract_intrinsics(info)
        if fx <= 0.0 or fy <= 0.0:
            self.get_logger().warn("Invalid intrinsics; skipping pointcloud publish.")
            return

        points_cam = self._depth_to_points(depth, fx, fy, cx, cy)
        points_world = self._transform_points(points_cam, pose)

        frame_id = self.output_frame_id or pose.header.frame_id or "world"
        cloud_msg = self._build_pointcloud(points_world, msg, frame_id)
        self.pub.publish(cloud_msg)

    def _decode_depth(self, msg: RosImage) -> Optional[np.ndarray]:
        if msg.encoding == "32FC1":
            depth = np.frombuffer(msg.data, dtype=np.float32)
            depth = depth.reshape((msg.height, msg.width))
            return depth
        if msg.encoding == "16UC1":
            depth = np.frombuffer(msg.data, dtype=np.uint16)
            depth = depth.reshape((msg.height, msg.width)).astype(np.float32) / 1000.0
            return depth
        self.get_logger().warn(f"Unsupported depth encoding: {msg.encoding}")
        return None

    def _extract_intrinsics(self, info: CameraInfo) -> Tuple[float, float, float, float]:
        fx = float(info.k[0])
        fy = float(info.k[4])
        cx = float(info.k[2])
        cy = float(info.k[5])
        return fx, fy, cx, cy

    def _depth_to_points(
        self, depth: np.ndarray, fx: float, fy: float, cx: float, cy: float
    ) -> np.ndarray:
        stride = max(self.stride, 1)
        v = np.arange(0, depth.shape[0], stride)
        u = np.arange(0, depth.shape[1], stride)
        uu, vv = np.meshgrid(u, v)

        z = depth[vv, uu]
        mask = np.isfinite(z) & (z > self.min_depth)
        if self.max_depth > 0.0:
            mask &= z < self.max_depth

        z = z[mask]
        uu = uu[mask].astype(np.float32)
        vv = vv[mask].astype(np.float32)

        x = (uu - cx) / fx * z
        y = (vv - cy) / fy * z
        return np.stack([x, y, z], axis=1).astype(np.float32)

    def _transform_points(self, points: np.ndarray, pose: PoseStamped) -> np.ndarray:
        if points.size == 0:
            return points
        q = pose.pose.orientation
        t = pose.pose.position
        R = self._quat_to_rot(q.x, q.y, q.z, q.w)
        trans = np.array([t.x, t.y, t.z], dtype=np.float32)
        return (R @ points.T).T + trans[None, :]

    def _quat_to_rot(self, x: float, y: float, z: float, w: float) -> np.ndarray:
        n = math.sqrt(x * x + y * y + z * z + w * w)
        if n < 1e-8:
            return np.eye(3, dtype=np.float32)
        x /= n
        y /= n
        z /= n
        w /= n
        xx = x * x
        yy = y * y
        zz = z * z
        xy = x * y
        xz = x * z
        yz = y * z
        wx = w * x
        wy = w * y
        wz = w * z
        return np.array(
            [
                [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
                [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
                [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
            ],
            dtype=np.float32,
        )

    def _build_pointcloud(self, points: np.ndarray, depth_msg: RosImage, frame_id: str) -> PointCloud2:
        cloud = PointCloud2()
        cloud.header.stamp = depth_msg.header.stamp
        cloud.header.frame_id = frame_id
        cloud.height = 1
        cloud.width = int(points.shape[0])
        cloud.fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        cloud.is_bigendian = False
        cloud.point_step = 12
        cloud.row_step = cloud.point_step * cloud.width
        cloud.is_dense = True
        cloud.data = points.astype(np.float32).tobytes()
        return cloud


def main():
    rclpy.init()
    node = DA3PointCloudNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
