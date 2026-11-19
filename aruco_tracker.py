import argparse
import csv
from datetime import datetime
import json
import math
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np


CSV_HEADER = [
    "time_s",
    "status",
    "obs_x_m",
    "obs_z_m",
    "obs_yaw_rad",
    "ekf_x_m",
    "ekf_z_m",
    "ekf_yaw_rad",
    "model",
    "cmd_vx_m_s",
    "cmd_vy_m_s",
    "cmd_omega_rad_s",
    "wheel_fl_rad_s",
    "wheel_fr_rad_s",
    "wheel_rl_rad_s",
    "wheel_rr_rad_s",
]


MODEL_NAMES = {
    "hold": "1: No prediction",
    "linear": "2: Linear",
    "ekf": "3: EKF",
}
MODEL_COLORS = {
    "hold": (80, 160, 255),
    "linear": (100, 220, 100),
    "ekf": (255, 220, 90),
}


def _require_object(config, key):
    value = config.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"設定 '{key}' はオブジェクトで指定してください。")
    return value


def _require_number(config, key, *, minimum=None, strict_minimum=False):
    value = config.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"設定 '{key}' は数値で指定してください。")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"設定 '{key}' は有限の数値で指定してください。")
    if minimum is not None and (
        value <= minimum if strict_minimum else value < minimum
    ):
        comparison = "より大きい" if strict_minimum else "以上"
        raise ValueError(f"設定 '{key}' は {minimum}{comparison}にしてください。")
    return value


def _require_integer(config, key, *, minimum):
    value = config.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"設定 '{key}' は {minimum} 以上の整数で指定してください。")
    return value


def load_config(path):
    with path.open("r", encoding="utf-8") as config_file:
        raw_config = json.load(config_file)
    if not isinstance(raw_config, dict):
        raise ValueError("設定ファイルの最上位はJSONオブジェクトにしてください。")

    camera = _require_object(raw_config, "camera")
    marker = _require_object(raw_config, "marker")
    target = _require_object(raw_config, "target")
    prediction = _require_object(raw_config, "prediction")
    display = _require_object(raw_config, "display")
    pid_config = _require_object(raw_config, "pid")
    mecanum = _require_object(raw_config, "mecanum")

    distortion = camera.get("distortion_coefficients")
    if (
        not isinstance(distortion, list)
        or len(distortion) not in (4, 5, 8, 12, 14)
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            for value in distortion
        )
    ):
        raise ValueError(
            "設定 'camera.distortion_coefficients' は有限数値4/5/8/12/14個の配列にしてください。"
        )

    pid = {}
    for axis in ("x", "z", "yaw"):
        axis_config = _require_object(pid_config, axis)
        pid[axis] = {
            "kp": _require_number(axis_config, "kp", minimum=0.0),
            "ki": _require_number(axis_config, "ki", minimum=0.0),
            "kd": _require_number(axis_config, "kd", minimum=0.0),
            "output_limit": _require_number(
                axis_config, "output_limit", minimum=0.0, strict_minimum=True
            ),
        }

    marker_target_id = (
        _require_integer(marker, "target_id", minimum=0)
        if "target_id" in marker
        else 0
    )

    return {
        "camera": {
            "index": _require_integer(camera, "index", minimum=0),
            "focal_length_px": _require_number(
                camera, "focal_length_px", minimum=0.0, strict_minimum=True
            ),
            "distortion_coefficients": [float(value) for value in distortion],
        },
        "marker_real_width": _require_number(
            marker, "real_width_m", minimum=0.0, strict_minimum=True
        ),
        "marker_target_id": marker_target_id,
        "target": {
            "x": _require_number(target, "x_m"),
            "z": _require_number(target, "z_m", minimum=0.0, strict_minimum=True),
            "yaw": math.radians(_require_number(target, "yaw_deg")),
        },
        "max_prediction_seconds": _require_number(
            prediction, "max_seconds", minimum=0.0, strict_minimum=True
        ),
        "display": {
            "panel_width": _require_integer(display, "panel_width", minimum=500),
            "panel_height": _require_integer(display, "panel_height", minimum=720),
            "history_length": _require_integer(display, "history_length", minimum=1),
        },
        "pid": pid,
        "mecanum": {
            "wheel_radius": _require_number(
                mecanum, "wheel_radius_m", minimum=0.0, strict_minimum=True
            ),
            "half_length": _require_number(
                mecanum, "half_length_m", minimum=0.0
            ),
            "half_width": _require_number(
                mecanum, "half_width_m", minimum=0.0
            ),
            "max_wheel_speed": _require_number(
                mecanum, "max_wheel_speed_rad_s",
                minimum=0.0, strict_minimum=True,
            ),
        },
    }


def wrap_angle(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


class PIDController:
    def __init__(self, kp, ki, kd, output_limit):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_limit = output_limit
        self.integral = 0.0
        self.previous_error = None

    def reset(self):
        self.integral = 0.0
        self.previous_error = None

    def update(self, error, dt):
        if dt <= 0:
            return 0.0

        self.integral = float(np.clip(
            self.integral + error * dt,
            -self.output_limit / max(self.ki, 1e-9),
            self.output_limit / max(self.ki, 1e-9),
        ))
        derivative = 0.0
        if self.previous_error is not None:
            derivative = (error - self.previous_error) / dt
        self.previous_error = error

        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        return float(np.clip(output, -self.output_limit, self.output_limit))


class ExtendedKalmanFilter:
    """EKF for marker pose and body-frame velocity in the X-Z plane."""

    def __init__(self, max_prediction_seconds):
        self.max_prediction_seconds = max_prediction_seconds
        self.state = None
        self.covariance = None
        self.last_observation_time = None
        self.measurement_matrix = np.zeros((3, 6), dtype=np.float64)
        self.measurement_matrix[0, 0] = 1.0
        self.measurement_matrix[1, 1] = 1.0
        self.measurement_matrix[2, 2] = 1.0
        self.measurement_noise = np.diag([
            0.02**2,
            0.02**2,
            math.radians(4.0)**2,
        ])

    def _predict(self, dt):
        x, z, yaw, forward_speed, lateral_speed, yaw_rate = self.state
        sin_yaw = math.sin(yaw)
        cos_yaw = math.cos(yaw)

        transition = np.eye(6, dtype=np.float64)
        transition[0, 2] = (
            forward_speed * cos_yaw - lateral_speed * sin_yaw
        ) * dt
        transition[0, 3] = sin_yaw * dt
        transition[0, 4] = cos_yaw * dt
        transition[1, 2] = (
            -forward_speed * sin_yaw - lateral_speed * cos_yaw
        ) * dt
        transition[1, 3] = cos_yaw * dt
        transition[1, 4] = -sin_yaw * dt
        transition[2, 5] = dt

        self.state = np.array([
            x + (forward_speed * sin_yaw + lateral_speed * cos_yaw) * dt,
            z + (forward_speed * cos_yaw - lateral_speed * sin_yaw) * dt,
            wrap_angle(yaw + yaw_rate * dt),
            forward_speed,
            lateral_speed,
            yaw_rate,
        ], dtype=np.float64)
        self.state[3:5] = np.clip(self.state[3:5], -1.5, 1.5)
        self.state[5] = float(np.clip(self.state[5], -math.pi, math.pi))

        process_noise = np.diag([
            0.005**2 * dt,
            0.005**2 * dt,
            math.radians(1.0)**2 * dt,
            0.25**2 * dt,
            0.25**2 * dt,
            math.radians(20.0)**2 * dt,
        ])
        self.covariance = (
            transition @ self.covariance @ transition.T + process_noise
        )

    def step(self, measurement, timestamp, dt):
        if self.state is None:
            if measurement is None:
                return None
            self.state = np.array([
                measurement[0], measurement[1], measurement[2], 0.0, 0.0, 0.0
            ], dtype=np.float64)
            self.covariance = np.diag([
                0.02**2,
                0.02**2,
                math.radians(4.0)**2,
                0.5**2,
                0.5**2,
                math.radians(45.0)**2,
            ])
            self.last_observation_time = timestamp
            return self.state[:3].copy()

        if self.last_observation_time is None:
            prediction_dt = dt
        else:
            elapsed_before_frame = max(
                0.0, timestamp - dt - self.last_observation_time
            )
            prediction_dt = min(
                dt,
                max(0.0, self.max_prediction_seconds - elapsed_before_frame),
            )
        if prediction_dt > 0:
            self._predict(prediction_dt)

        if measurement is not None:
            residual = np.array([
                measurement[0] - self.state[0],
                measurement[1] - self.state[1],
                wrap_angle(measurement[2] - self.state[2]),
            ], dtype=np.float64)
            innovation_covariance = (
                self.measurement_matrix
                @ self.covariance
                @ self.measurement_matrix.T
                + self.measurement_noise
            )
            gain = np.linalg.solve(
                innovation_covariance,
                self.measurement_matrix @ self.covariance,
            ).T
            self.state += gain @ residual
            self.state[2] = wrap_angle(self.state[2])
            self.state[3:5] = np.clip(self.state[3:5], -1.5, 1.5)
            self.state[5] = float(np.clip(self.state[5], -math.pi, math.pi))
            identity = np.eye(6, dtype=np.float64)
            correction = identity - gain @ self.measurement_matrix
            self.covariance = (
                correction @ self.covariance @ correction.T
                + gain @ self.measurement_noise @ gain.T
            )
            self.covariance = (self.covariance + self.covariance.T) / 2.0
            self.last_observation_time = timestamp

        return self.state[:3].copy()


class PosePredictors:
    def __init__(self, max_prediction_seconds):
        self.max_prediction_seconds = max_prediction_seconds
        self.last_observation = None
        self.last_observation_time = None
        self.linear_velocity = np.zeros(3, dtype=np.float64)
        self.ekf = ExtendedKalmanFilter(max_prediction_seconds)

    def update(self, measurement, timestamp, dt):
        if measurement is not None:
            if self.last_observation is not None:
                elapsed = timestamp - self.last_observation_time
                if elapsed > 1e-3:
                    measured_velocity = np.array([
                        (measurement[0] - self.last_observation[0]) / elapsed,
                        (measurement[1] - self.last_observation[1]) / elapsed,
                        wrap_angle(
                            measurement[2] - self.last_observation[2]
                        ) / elapsed,
                    ], dtype=np.float64)
                    measured_velocity = np.clip(
                        measured_velocity,
                        [-1.5, -1.5, -math.pi],
                        [1.5, 1.5, math.pi],
                    )
                    self.linear_velocity = (
                        0.65 * self.linear_velocity + 0.35 * measured_velocity
                    )
            self.last_observation = measurement.copy()
            self.last_observation_time = timestamp
            hold_estimate = measurement.copy()
            linear_estimate = measurement.copy()
        elif self.last_observation is not None:
            elapsed = min(
                max(0.0, timestamp - self.last_observation_time),
                self.max_prediction_seconds,
            )
            hold_estimate = self.last_observation.copy()
            linear_estimate = self.last_observation + self.linear_velocity * elapsed
            linear_estimate[2] = wrap_angle(linear_estimate[2])
        else:
            hold_estimate = None
            linear_estimate = None

        ekf_estimate = self.ekf.step(measurement, timestamp, dt)
        return {
            "hold": hold_estimate,
            "linear": linear_estimate,
            "ekf": ekf_estimate,
        }


def estimate_marker_pose(corners, camera_matrix, distortion, marker_real_width):
    half_width = marker_real_width / 2.0
    object_points = np.array([
        [-half_width, half_width, 0.0],
        [half_width, half_width, 0.0],
        [half_width, -half_width, 0.0],
        [-half_width, -half_width, 0.0],
    ], dtype=np.float32)
    image_points = np.asarray(corners, dtype=np.float32).reshape(4, 2)

    solved, rotation_vector, translation_vector = cv2.solvePnP(
        object_points,
        image_points,
        camera_matrix,
        distortion,
        flags=cv2.SOLVEPNP_IPPE_SQUARE,
    )
    if not solved:
        return None

    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    yaw = math.atan2(rotation_matrix[0, 2], rotation_matrix[2, 2])
    position = translation_vector.reshape(3)
    if not np.all(np.isfinite(position)) or position[2] <= 0:
        return None
    return position, yaw, rotation_vector, translation_vector


def calculate_control(estimate, controllers, dt, target, mecanum):
    x_error = estimate[0] - target["x"]
    z_error = estimate[1] - target["z"]
    yaw_error = wrap_angle(estimate[2] - target["yaw"])

    vx = controllers[0].update(x_error, dt)
    vy = controllers[1].update(z_error, dt)
    omega = controllers[2].update(yaw_error, dt)

    wheel_radius = mecanum["wheel_radius"]
    arm = mecanum["half_length"] + mecanum["half_width"]

    def wheel_speeds(command):
        lateral, forward, rotation = command
        return np.array([
            (forward - lateral - arm * rotation) / wheel_radius,
            (forward + lateral + arm * rotation) / wheel_radius,
            (forward + lateral - arm * rotation) / wheel_radius,
            (forward - lateral + arm * rotation) / wheel_radius,
        ], dtype=np.float64)

    command = np.array([vx, vy, omega], dtype=np.float64)
    wheels = wheel_speeds(command)
    max_wheel_speed = mecanum["max_wheel_speed"]
    scale = max(1.0, float(np.max(np.abs(wheels))) / max_wheel_speed)
    command /= scale
    wheels = wheel_speeds(command)
    return command, wheels


def draw_text(image, text, position, color=(235, 235, 235), scale=0.55):
    cv2.putText(
        image, text, position, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1,
        cv2.LINE_AA,
    )


def draw_graph(canvas, offset_x, histories, panel_width, target):
    chart_left = offset_x + 42
    chart_top = 92
    chart_right = offset_x + panel_width - 18
    chart_bottom = 342
    graph_width = chart_right - chart_left
    graph_height = chart_bottom - chart_top

    all_positions = [[target["x"], target["z"]]]
    for history in histories.values():
        all_positions.extend((state[0], state[1]) for state in history)
    positions = np.asarray(all_positions, dtype=np.float64)
    min_x, min_z = np.min(positions, axis=0)
    max_x, max_z = np.max(positions, axis=0)
    x_span = max(max_x - min_x, 0.6)
    z_span = max(max_z - min_z, 0.6)
    x_margin = max(x_span * 0.15, 0.1)
    z_margin = max(z_span * 0.15, 0.1)
    min_x -= x_margin
    max_x += x_margin
    min_z -= z_margin
    max_z += z_margin

    def to_pixel(x, z):
        px = chart_left + int((x - min_x) / (max_x - min_x) * graph_width)
        py = chart_bottom - int((z - min_z) / (max_z - min_z) * graph_height)
        return px, py

    cv2.rectangle(
        canvas, (chart_left, chart_top), (chart_right, chart_bottom),
        (90, 90, 90), 1,
    )
    for index in range(1, 5):
        x = chart_left + index * graph_width // 5
        z = chart_top + index * graph_height // 5
        cv2.line(canvas, (x, chart_top), (x, chart_bottom), (55, 55, 55), 1)
        cv2.line(canvas, (chart_left, z), (chart_right, z), (55, 55, 55), 1)
    draw_text(canvas, "X (m)", (chart_left + graph_width // 2 - 15, chart_bottom + 20),
              (180, 180, 180), 0.42)
    draw_text(canvas, "Z (m)", (offset_x + 4, chart_top + graph_height // 2),
              (180, 180, 180), 0.42)

    target_pixel = to_pixel(target["x"], target["z"])
    cv2.drawMarker(
        canvas, target_pixel, (255, 255, 255), cv2.MARKER_STAR, 14, 2
    )
    draw_text(canvas, "target", (target_pixel[0] + 8, target_pixel[1] - 5),
              (220, 220, 220), 0.4)

    for name, history in histories.items():
        color = MODEL_COLORS[name]
        points = [to_pixel(state[0], state[1]) for state in history]
        if len(points) > 1:
            cv2.polylines(
                canvas, [np.asarray(points, dtype=np.int32)], False, color, 2,
                cv2.LINE_AA,
            )
        if points:
            cv2.circle(canvas, points[-1], 5, color, -1, cv2.LINE_AA)


def draw_gauge(canvas, x, y, label, value, limit, unit, color):
    bar_left = x + 120
    bar_width = 255
    bar_height = 18
    cv2.rectangle(
        canvas, (bar_left, y), (bar_left + bar_width, y + bar_height),
        (75, 75, 75), 1,
    )
    center_x = bar_left + bar_width // 2
    cv2.line(canvas, (center_x, y), (center_x, y + bar_height), (150, 150, 150), 1)
    fill_width = int(min(abs(value) / limit, 1.0) * (bar_width // 2))
    if value >= 0:
        start, end = center_x, center_x + fill_width
    else:
        start, end = center_x - fill_width, center_x
    cv2.rectangle(canvas, (start, y + 2), (end, y + bar_height - 2), color, -1)
    draw_text(canvas, label, (x, y + 14), scale=0.48)
    draw_text(
        canvas, f"{value:+.3f} {unit}", (bar_left + bar_width + 10, y + 14),
        (235, 235, 235), 0.45,
    )


def find_target_marker(corners, ids, target_id):
    if ids is None or len(corners) == 0:
        return None
    flat_ids = np.asarray(ids).reshape(-1)
    matching_indices = [
        index for index, marker_id in enumerate(flat_ids)
        if int(marker_id) == target_id
    ]
    if not matching_indices:
        return None
    if len(matching_indices) == 1:
        return matching_indices[0]
    areas = [
        abs(cv2.contourArea(corners[index].reshape(4, 2).astype(np.float32)))
        for index in matching_indices
    ]
    return matching_indices[int(np.argmax(areas))]


def is_safety_stop_active(measurement, last_observation_time, now, max_prediction_seconds):
    if measurement is not None or last_observation_time is None:
        return False
    return (now - last_observation_time) >= max_prediction_seconds - 1e-9


def format_csv_row(
    elapsed_time, status, measurement, ekf_estimate, selected_model, command, wheel_speeds
):
    obs_values = (
        [f"{measurement[0]:.6f}", f"{measurement[1]:.6f}", f"{measurement[2]:.6f}"]
        if measurement is not None
        else ["", "", ""]
    )
    ekf_values = (
        [f"{ekf_estimate[0]:.6f}", f"{ekf_estimate[1]:.6f}", f"{ekf_estimate[2]:.6f}"]
        if ekf_estimate is not None
        else ["", "", ""]
    )
    return [
        f"{elapsed_time:.4f}",
        status,
        *obs_values,
        *ekf_values,
        selected_model,
        f"{command[0]:.6f}",
        f"{command[1]:.6f}",
        f"{command[2]:.6f}",
        f"{wheel_speeds[0]:.6f}",
        f"{wheel_speeds[1]:.6f}",
        f"{wheel_speeds[2]:.6f}",
        f"{wheel_speeds[3]:.6f}",
    ]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="ArUcoマーカー追跡・PID制御シミュレーション"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("config.json"),
        help="設定JSONファイルのパス (既定: config.json)",
    )
    parser.add_argument(
        "--video",
        type=Path,
        default=None,
        help="入力動画ファイルのパス (指定時はカメラの代わりに動画を使用)",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="画面描画・ウィンドウ表示を無効化する (ヘッドレス・ラズパイ負荷軽減用)",
    )
    parser.add_argument(
        "--log",
        type=Path,
        default=None,
        help="CSVログの出力先パス (未指定時は aruco_log_YYYYMMDD_HHMMSS.csv に自動出力)",
    )
    parser.add_argument(
        "--no-log",
        action="store_true",
        help="CSVログ出力を無効化する",
    )
    parser.add_argument(
        "--model",
        choices=tuple(MODEL_NAMES.keys()),
        default="ekf",
        help="初期制御モデル (既定: ekf)",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    config_path = args.config
    try:
        config = load_config(config_path)
    except FileNotFoundError:
        print(f"エラー: 設定ファイルが見つかりません: {config_path}")
        return
    except json.JSONDecodeError as error:
        print(
            f"エラー: 設定ファイルのJSON形式が不正です "
            f"({error.lineno}行 {error.colno}列): {error.msg}"
        )
        return
    except (OSError, UnicodeError, ValueError) as error:
        print(f"エラー: 設定ファイルを読み込めません: {error}")
        return

    camera_config = config["camera"]
    display_config = config["display"]
    target = config["target"]
    mecanum = config["mecanum"]
    target_marker_id = config["marker_target_id"]
    max_prediction_seconds = config["max_prediction_seconds"]

    is_video_mode = args.video is not None
    if is_video_mode:
        if not args.video.exists():
            print(f"エラー: 動画ファイルが見つかりません: {args.video}")
            return
        cap = cv2.VideoCapture(str(args.video))
        if not cap.isOpened():
            print(f"エラー: 動画ファイルをオープンできませんでした: {args.video}")
            return
        raw_fps = cap.get(cv2.CAP_PROP_FPS)
        video_fps = (
            raw_fps if raw_fps and math.isfinite(raw_fps) and raw_fps > 0 else 30.0
        )
        video_dt = 1.0 / video_fps
    else:
        cap = cv2.VideoCapture(camera_config["index"])
        if not cap.isOpened():
            print("エラー: カメラをオープンできませんでした。インデックスを確認してください。")
            return
        video_dt = None

    log_file = None
    log_writer = None
    log_path = None
    if not args.no_log:
        if args.log is not None:
            log_path = args.log
        else:
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_path = Path(f"aruco_log_{timestamp_str}.csv")
        try:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_file = log_path.open("w", newline="", encoding="utf-8")
            log_writer = csv.writer(log_file)
            log_writer.writerow(CSV_HEADER)
        except OSError as error:
            print(f"エラー: CSVログファイルを作成できません: {error}")
            cap.release()
            return

    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)
    predictors = PosePredictors(max_prediction_seconds)
    histories = (
        {
            name: deque(maxlen=display_config["history_length"])
            for name in MODEL_NAMES
        }
        if not args.no_display
        else None
    )
    controllers = [
        PIDController(**config["pid"]["x"]),
        PIDController(**config["pid"]["z"]),
        PIDController(**config["pid"]["yaw"]),
    ]
    selected_model = args.model
    start_time = time.monotonic()
    previous_frame_time = start_time
    frame_index = 0
    was_safety_stopped = False

    mode_desc = f"動画入力 ({args.video})" if is_video_mode else f"カメラ {camera_config['index']}"
    display_desc = "画面描画OFF" if args.no_display else "1/2/3で推定方式を選択、qで終了"
    print(
        f"ArUco制御シミュレーション開始 [{mode_desc} / 追跡対象: ID {target_marker_id}]: "
        f"{display_desc}"
    )
    if log_path is not None:
        print(f"CSVログ出力先: {log_path}")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                if is_video_mode:
                    print(f"動画ファイルの処理が完了しました ({frame_index} フレーム)。")
                else:
                    print("エラー: フレームを取得できませんでした。")
                break

            if is_video_mode:
                now = frame_index * video_dt
                dt = video_dt
                elapsed_time = now
            else:
                now = time.monotonic()
                dt = float(np.clip(now - previous_frame_time, 1e-3, 0.2))
                previous_frame_time = now
                elapsed_time = now - start_time
            frame_index += 1

            height, width = frame.shape[:2]
            focal_length = camera_config["focal_length_px"]
            camera_matrix = np.array([
                [focal_length, 0.0, width / 2.0],
                [0.0, focal_length, height / 2.0],
                [0.0, 0.0, 1.0],
            ], dtype=np.float64)
            distortion = np.asarray(
                camera_config["distortion_coefficients"], dtype=np.float64
            ).reshape(-1, 1)

            corners, ids, _ = detector.detectMarkers(frame)
            pose = None
            pose_status = f"TARGET ID {target_marker_id} NOT DETECTED"
            if ids is not None and len(corners) > 0:
                if not args.no_display:
                    cv2.aruco.drawDetectedMarkers(frame, corners, ids)
                marker_index = find_target_marker(corners, ids, target_marker_id)
                if marker_index is not None:
                    pose = estimate_marker_pose(
                        corners[marker_index],
                        camera_matrix,
                        distortion,
                        config["marker_real_width"],
                    )
                    if pose is not None:
                        position, yaw, rotation_vector, translation_vector = pose
                        if not args.no_display:
                            marker_center = tuple(
                                np.round(np.mean(corners[marker_index][0], axis=0)).astype(int)
                            )
                            cv2.circle(frame, marker_center, 5, (0, 0, 255), -1)
                            cv2.drawFrameAxes(
                                frame, camera_matrix, distortion, rotation_vector,
                                translation_vector, config["marker_real_width"] * 0.5,
                            )
                        pose_status = (
                            f"ID {target_marker_id} X:{position[0]:+.2f} "
                            f"Y:{position[1]:+.2f} Z:{position[2]:.2f}m "
                            f"Yaw:{math.degrees(yaw):+.1f}deg"
                        )
                    else:
                        pose_status = f"ID {target_marker_id} POSE ESTIMATION FAILED"

            measurement = None
            if pose is not None:
                position, yaw, _, _ = pose
                measurement = np.array(
                    [position[0], position[2], yaw], dtype=np.float64
                )
            estimates = predictors.update(measurement, now, dt)
            if histories is not None:
                for name, estimate in estimates.items():
                    if estimate is not None:
                        histories[name].append(estimate.copy())

            safety_stop = is_safety_stop_active(
                measurement,
                predictors.last_observation_time,
                now,
                max_prediction_seconds,
            )

            estimate = estimates[selected_model]
            if safety_stop:
                if not was_safety_stopped:
                    for controller in controllers:
                        controller.reset()
                command = np.zeros(3, dtype=np.float64)
                wheel_speeds = np.zeros(4, dtype=np.float64)
                status_label = "SAFETY_STOP"
            elif estimate is not None:
                command, wheel_speeds = calculate_control(
                    estimate, controllers, dt, target, mecanum
                )
                status_label = "MEASUREMENT" if measurement is not None else "PREDICTING"
            else:
                command = np.zeros(3, dtype=np.float64)
                wheel_speeds = np.zeros(4, dtype=np.float64)
                status_label = "WAITING"
            was_safety_stopped = safety_stop

            if log_writer is not None:
                log_writer.writerow(
                    format_csv_row(
                        elapsed_time,
                        status_label,
                        measurement,
                        estimates["ekf"],
                        selected_model,
                        command,
                        wheel_speeds,
                    )
                )

            if args.no_display:
                continue

            if safety_stop:
                lost_seconds = now - predictors.last_observation_time
                status_color = (0, 0, 255)
                camera_status = (
                    f"SAFETY STOP (LOST {lost_seconds:.1f}s >= "
                    f"{max_prediction_seconds:.1f}s)"
                )
            elif measurement is None:
                status_color = (0, 150, 255)
                camera_status = (
                    "OCCLUDED / PREDICTING"
                    if estimates["ekf"] is not None
                    else "WAITING FOR MARKER"
                )
            else:
                status_color = (80, 220, 80)
                camera_status = "MEASUREMENT"
            draw_text(frame, camera_status, (12, 24), status_color, 0.6)
            draw_text(frame, pose_status, (12, 48), (235, 235, 235), 0.48)
            draw_text(
                frame,
                f"Control model: {MODEL_NAMES[selected_model]} (display only)",
                (12, 72), (255, 220, 80), 0.48,
            )

            panel_width = display_config["panel_width"]
            display_height = max(height, display_config["panel_height"])
            canvas = np.full(
                (display_height, width + panel_width, 3), (28, 28, 28),
                dtype=np.uint8,
            )
            canvas[:height, :width] = frame
            panel_x = width
            cv2.line(
                canvas, (panel_x, 0), (panel_x, display_height - 1),
                (90, 90, 90), 1,
            )
            draw_text(canvas, "Pose prediction comparison (X-Z)", (panel_x + 14, 28),
                      (240, 240, 240), 0.58)
            legend_x = panel_x + 14
            for name in MODEL_NAMES:
                cv2.circle(canvas, (legend_x + 5, 61), 5, MODEL_COLORS[name], -1)
                draw_text(canvas, MODEL_NAMES[name], (legend_x + 16, 66),
                          MODEL_COLORS[name], 0.43)
                legend_x += 155
            draw_graph(canvas, panel_x, histories, panel_width, target)

            draw_text(
                canvas, f"CONTROL INPUT: {MODEL_NAMES[selected_model]}",
                (panel_x + 14, 378), (255, 220, 80), 0.55,
            )
            if safety_stop:
                draw_text(
                    canvas, "SAFETY STOP: Wheel speed forced to 0",
                    (panel_x + 14, 405), (0, 0, 255), 0.48,
                )
            elif estimate is None:
                draw_text(canvas, "Waiting for the first valid pose", (panel_x + 14, 405),
                          (0, 150, 255), 0.48)
            else:
                draw_text(
                    canvas,
                    f"Estimate X:{estimate[0]:+.3f}m  Z:{estimate[1]:.3f}m  "
                    f"Yaw:{math.degrees(estimate[2]):+.1f}deg",
                    (panel_x + 14, 405), (210, 210, 210), 0.45,
                )
            draw_text(canvas, "Virtual PID output (no hardware command)", (panel_x + 14, 438),
                      (200, 200, 200), 0.45)
            draw_gauge(canvas, panel_x + 14, 454, "Vx lateral", command[0],
                       0.4, "m/s", (80, 160, 255))
            draw_gauge(canvas, panel_x + 14, 483, "Vy forward", command[1],
                       0.4, "m/s", (100, 220, 100))
            draw_gauge(canvas, panel_x + 14, 512, "Omega", command[2],
                       1.2, "rad/s", (255, 220, 80))
            draw_text(
                canvas,
                f"Targets: X={target['x']:.2f}m  Z={target['z']:.2f}m  "
                f"Yaw={math.degrees(target['yaw']):.0f}deg",
                (panel_x + 14, 554), (195, 195, 195), 0.45,
            )
            draw_text(canvas, "Mecanum wheel angular velocity (rad/s)", (panel_x + 14, 584),
                      (200, 200, 200), 0.45)
            draw_text(
                canvas,
                f"FL {wheel_speeds[0]:+5.2f}  FR {wheel_speeds[1]:+5.2f}  "
                f"RL {wheel_speeds[2]:+5.2f}  RR {wheel_speeds[3]:+5.2f}",
                (panel_x + 14, 610), (220, 220, 220), 0.48,
            )
            draw_text(
                canvas, "Keys: 1 No prediction | 2 Linear | 3 EKF | q Quit",
                (panel_x + 14, 672), (180, 180, 180), 0.43,
            )

            cv2.imshow("ArUco PID Simulation", canvas)
            key = cv2.waitKey(1) & 0xFF
            key_models = {ord("1"): "hold", ord("2"): "linear", ord("3"): "ekf"}
            if key in key_models:
                selected_model = key_models[key]
                for controller in controllers:
                    controller.reset()
            elif key == ord("q"):
                break
    except KeyboardInterrupt:
        print("\n中断されました。")
    finally:
        cap.release()
        if not args.no_display:
            cv2.destroyAllWindows()
        if log_file is not None:
            log_file.close()
            print(f"CSVログを保存しました: {log_path}")


if __name__ == "__main__":
    main()
