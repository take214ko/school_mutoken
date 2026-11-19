# =============================================================================
# ArUcoマーカー追跡 ＆ メカナムホイールPID制御シミュレーション
#
# このプログラムの全体の流れ：
#   1. config.json（設定ファイル）を読み込む
#   2. カメラ映像（または動画ファイル）から ArUco マーカー（ID 0）を検出する
#   3. マーカーの3次元位置 (X, Z) と向き (Yaw) を計算する
#   4. 見失ったときのために 3つの方式（予測なし / 線形予測 / EKF）で位置を推定する
#   5. 目標位置とのズレから PID制御でロボットの速度と4輪の回転速度を計算する
#   6. 結果をCSVファイルに記録し、必要なら画面に表示する
# =============================================================================

# --- 標準ライブラリ（Pythonに最初から入っている便利な道具） ---
import argparse           # コマンドライン引数（--video や --no-display など）を扱う
import csv                # CSVファイル（Excel等で開ける表形式データ）を書き出す
from datetime import datetime  # 現在日時を取得してログファイル名に使う
import json               # config.json（設定ファイル）を読み込む
import math               # sin, cos, atan2, radians などの数学関数
import platform           # 実行中のOS（Windows か Linux/ラズパイ か）を判定する
import time               # 経過時間（フレーム間の時間 dt）を測る
from collections import deque  # 最大長を決められるリスト（古い履歴を自動で消す）
from pathlib import Path  # ファイルのパス（場所）を安全に扱う

# --- 外部ライブラリ（pip でインストールしたもの） ---
import cv2                # OpenCV：カメラ映像の取得、ArUco検出、画面描画
import numpy as np        # NumPy：配列や行列の高速な数値計算


# CSVログの1行目（列名＝ヘッダー）を定義するリスト
CSV_HEADER = [
    "time_s",           # 開始からの経過時間 [秒]
    "status",           # 状態 (MEASUREMENT / PREDICTING / SAFETY_STOP / WAITING)
    "obs_x_m",          # カメラの観測値：横方向 X [m]
    "obs_z_m",          # カメラの観測値：奥行き Z [m]
    "obs_yaw_rad",      # カメラの観測値：向き Yaw [ラジアン]
    "ekf_x_m",          # EKFの推定値：横方向 X [m]
    "ekf_z_m",          # EKFの推定値：奥行き Z [m]
    "ekf_yaw_rad",      # EKFの推定値：向き Yaw [ラジアン]
    "model",            # 制御に使っているモデル名 (hold / linear / ekf)
    "cmd_vx_m_s",       # PID出力：横移動速度指令 Vx [m/s]
    "cmd_vy_m_s",       # PID出力：前後移動速度指令 Vy [m/s]
    "cmd_omega_rad_s",  # PID出力：旋回角速度指令 Omega [rad/s]
    "wheel_fl_rad_s",   # 左前 (Front-Left)  車輪の角速度 [rad/s]
    "wheel_fr_rad_s",   # 右前 (Front-Right) 車輪の角速度 [rad/s]
    "wheel_rl_rad_s",   # 左後 (Rear-Left)   車輪の角速度 [rad/s]
    "wheel_rr_rad_s",   # 右後 (Rear-Right)  車輪の角速度 [rad/s]
]


# 辞書（dict）：キー（内部名）と値（画面表示名・色）のペアを管理する
MODEL_NAMES = {
    "hold": "1: No prediction",  # 1キー：予測なし（最後に見えた位置をそのまま保持）
    "linear": "2: Linear",       # 2キー：線形予測（等速直線運動とみなして外挿）
    "ekf": "3: EKF",             # 3キー：拡張カルマンフィルタ（ノイズ除去＋運動モデル予測）
}
# OpenCVの色指定は RGB ではなく BGR（青, 緑, 赤）の順番なので注意！
MODEL_COLORS = {
    "hold": (80, 160, 255),      # オレンジ系
    "linear": (100, 220, 100),   # 緑系
    "ekf": (255, 220, 90),       # 水色系
}


# -----------------------------------------------------------------------------
# 設定ファイル (config.json) のチェック＆読み込み関数群
# ※ 先頭に「_」が付く関数は「このファイルの中だけで使う補助関数」というPythonの慣習
# -----------------------------------------------------------------------------
def _require_object(config, key):
    """辞書 config から key を取り出し、中身が辞書（JSONオブジェクト）か確認する。"""
    value = config.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"設定 '{key}' はオブジェクトで指定してください。")
    return value


def _require_number(config, key, *, minimum=None, strict_minimum=False):
    """設定値が正しい数値（小数または整数）かチェックして float で返す。
    ※ 引数の「*」より後ろは、呼び出すときに minimum=0.0 のように名前付きで指定する必要がある。
    """
    value = config.get(key)
    # Pythonでは bool (True/False) も int の仲間なので、先に除外する
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"設定 '{key}' は数値で指定してください。")
    value = float(value)
    # NaN（非数）や無限大 (inf) を弾く
    if not math.isfinite(value):
        raise ValueError(f"設定 '{key}' は有限の数値で指定してください。")
    # 最小値の条件チェック（strict_minimum=True なら「より大きい」、False なら「以上」）
    if minimum is not None and (
        value <= minimum if strict_minimum else value < minimum
    ):
        comparison = "より大きい" if strict_minimum else "以上"
        raise ValueError(f"設定 '{key}' は {minimum}{comparison}にしてください。")
    return value


def _require_integer(config, key, *, minimum):
    """設定値が minimum 以上の整数 (int) かチェックして返す。"""
    value = config.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"設定 '{key}' は {minimum} 以上の整数で指定してください。")
    return value


def load_config(path):
    """config.json を読み込み、値の型や範囲をチェックした上で辞書として返す。"""
    # with文を使うと、処理が終わったときに自動でファイルを閉じてくれる
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

    # カメラのレンズ歪み係数（OpenCVの仕様上、要素数は 4, 5, 8, 12, 14 のいずれか）
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

    # x, z, yaw の3軸それぞれの PID ゲインを読み込む
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

    # target_id が書かれていればその値を使い、省略されていれば既定の 0 を使う
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
            # 角度は設定ファイルでは「度(deg)」で書き、プログラム内部では「ラジアン(rad)」に変換する
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
    """角度[rad]を -π 〜 +π (-180度 〜 +180度) の範囲に収める。
    例：+179度 と -179度 の差が 358度 にならず、正しく 2度 と計算できるようにするため。
    """
    return (angle + math.pi) % (2 * math.pi) - math.pi


# -----------------------------------------------------------------------------
# PID制御器クラス
# 「目標とのズレ (error)」から「どれくらい動かすか (速度指令)」を計算する仕組み
#   P (比例): 今のズレが大きいほど強く動かす
#   I (積分): 過去のズレの積み重ねを打ち消す（定常偏差をなくす）
#   D (微分): ズレの変化スピードを見てブレーキをかける（行き過ぎを防ぐ）
# -----------------------------------------------------------------------------
class PIDController:
    def __init__(self, kp, ki, kd, output_limit):
        """クラスを作るときに最初に1回だけ呼ばれる初期化メソッド。"""
        self.kp = kp                      # Pゲイン（比例の強さ）
        self.ki = ki                      # Iゲイン（積分の強さ）
        self.kd = kd                      # Dゲイン（微分の強さ）
        self.output_limit = output_limit  # 出力の最大値（これ以上速くしない）
        self.integral = 0.0               # 誤差の累積値（I項用）
        self.previous_error = None        # 1フレーム前の誤差（D項用）

    def reset(self):
        """溜まった積分値や前回の誤差を0に戻す（モード切替時や安全停止時に使用）。"""
        self.integral = 0.0
        self.previous_error = None

    def update(self, error, dt):
        """現在の誤差 error と経過時間 dt [秒] から制御出力を計算して返す。"""
        if dt <= 0:
            return 0.0

        # I項（積分）を更新。大きくなりすぎないよう np.clip で上下限を制限（アンチワインドアップ）
        self.integral = float(np.clip(
            self.integral + error * dt,
            -self.output_limit / max(self.ki, 1e-9),
            self.output_limit / max(self.ki, 1e-9),
        ))
        # D項（微分）＝ (今回の誤差 - 前回の誤差) ÷ 経過時間
        derivative = 0.0
        if self.previous_error is not None:
            derivative = (error - self.previous_error) / dt
        self.previous_error = error

        # P + I + D を足し合わせ、出力上限 [-output_limit, +output_limit] の範囲に収める
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        return float(np.clip(output, -self.output_limit, self.output_limit))


# -----------------------------------------------------------------------------
# 拡張カルマンフィルタ (EKF: Extended Kalman Filter) クラス
# カメラのノイズを滑らかに除去しつつ、マーカーが見えない間も「現在の速度」から
# 次の位置を予測し続けるためのアルゴリズム。
#
# 内部で管理する状態 (self.state) は 6つの数値の配列：
#   [x位置, z位置, 向きyaw, 前進速度, 横移動速度, 旋回角速度]
# -----------------------------------------------------------------------------
class ExtendedKalmanFilter:
    """EKF for marker pose and body-frame velocity in the X-Z plane."""

    def __init__(self, max_prediction_seconds):
        self.max_prediction_seconds = max_prediction_seconds  # 予測を続ける最大時間[秒]
        self.state = None                  # 現在の推定状態 (長さ6の配列)
        self.covariance = None             # 推定の不確かさ（共分散行列 6x6）
        self.last_observation_time = None  # 最後にカメラで観測できた時刻

        # 観測行列 H (3x6)：6つの状態のうち、カメラで直接測れるのは [x, z, yaw] の先頭3つだけ
        self.measurement_matrix = np.zeros((3, 6), dtype=np.float64)
        self.measurement_matrix[0, 0] = 1.0
        self.measurement_matrix[1, 1] = 1.0
        self.measurement_matrix[2, 2] = 1.0

        # 観測ノイズ R (3x3)：カメラ測定にどれくらい誤差が含まれるかの目安（標準偏差の2乗）
        self.measurement_noise = np.diag([
            0.02**2,               # X方向の誤差目安 2cm
            0.02**2,               # Z方向の誤差目安 2cm
            math.radians(4.0)**2,  # Yaw角の誤差目安 4度
        ])

    def _predict(self, dt):
        """【予測ステップ】dt秒後の位置と不確かさ（共分散）を運動モデルから予測する。"""
        x, z, yaw, forward_speed, lateral_speed, yaw_rate = self.state
        sin_yaw = math.sin(yaw)
        cos_yaw = math.cos(yaw)

        # ヤコビ行列 F (6x6)：非線形な三角関数 (sin/cos) を現在の角度付近で直線近似したもの
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

        # 現在の速度で dt 秒動いた先の位置・角度を計算する
        self.state = np.array([
            x + (forward_speed * sin_yaw + lateral_speed * cos_yaw) * dt,
            z + (forward_speed * cos_yaw - lateral_speed * sin_yaw) * dt,
            wrap_angle(yaw + yaw_rate * dt),
            forward_speed,
            lateral_speed,
            yaw_rate,
        ], dtype=np.float64)
        # 推定速度が異常に大きな値にならないよう範囲制限
        self.state[3:5] = np.clip(self.state[3:5], -1.5, 1.5)
        self.state[5] = float(np.clip(self.state[5], -math.pi, math.pi))

        # プロセスノイズ Q：モデル通りに動かない分の不確かさを足す
        process_noise = np.diag([
            0.005**2 * dt,
            0.005**2 * dt,
            math.radians(1.0)**2 * dt,
            0.25**2 * dt,
            0.25**2 * dt,
            math.radians(20.0)**2 * dt,
        ])
        # 「@」は NumPy の行列の掛け算、.T は転置行列
        self.covariance = (
            transition @ self.covariance @ transition.T + process_noise
        )

    def step(self, measurement, timestamp, dt):
        """1フレーム分の処理（予測ステップ → 観測があれば更新ステップ）を行い、[x, z, yaw] を返す。"""
        # まだ1回もマーカーを見つけていない場合
        if self.state is None:
            if measurement is None:
                return None
            # 最初の観測値で位置を初期化（速度はまだ分からないので 0 で開始）
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

        # 見失っている間は max_prediction_seconds を超えて予測し続けないように時間を制限
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

        # 【更新ステップ】カメラでマーカーが観測できたときだけ、予測値を観測値で補正する
        if measurement is not None:
            # 観測値と予測値の差（イノベーション / 残差）
            residual = np.array([
                measurement[0] - self.state[0],
                measurement[1] - self.state[1],
                wrap_angle(measurement[2] - self.state[2]),
            ], dtype=np.float64)
            # カルマンゲイン K を計算（予測と観測のどちらをどれくらい信用するかの重み）
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
            # 状態と不確かさ（共分散）を補正
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
            # 計算誤差で対称行列が崩れないよう平均をとって整形
            self.covariance = (self.covariance + self.covariance.T) / 2.0
            self.last_observation_time = timestamp

        # 状態6要素のうち先頭3つ [x, z, yaw] のコピーを返す
        return self.state[:3].copy()


# -----------------------------------------------------------------------------
# 3つの予測モデル（hold / linear / ekf）をまとめて管理・同時実行するクラス
# -----------------------------------------------------------------------------
class PosePredictors:
    def __init__(self, max_prediction_seconds):
        self.max_prediction_seconds = max_prediction_seconds
        self.last_observation = None       # 最後に観測できた位置 [x, z, yaw]
        self.last_observation_time = None  # 最後に観測できた時刻
        self.linear_velocity = np.zeros(3, dtype=np.float64)  # 線形予測用の移動速度
        self.ekf = ExtendedKalmanFilter(max_prediction_seconds)

    def update(self, measurement, timestamp, dt):
        """今回の観測値 measurement（見失い時は None）から3モデルすべての推定値を辞書で返す。"""
        if measurement is not None:
            # マーカーが見えているとき：前回の位置との差から速度を計算して滑らかに更新
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
                    # 指数移動平均 (前回の65% + 今回の35%) で速度のノイズを抑える
                    self.linear_velocity = (
                        0.65 * self.linear_velocity + 0.35 * measured_velocity
                    )
            self.last_observation = measurement.copy()
            self.last_observation_time = timestamp
            # 見えている間は hold も linear も観測値をそのまま使う
            hold_estimate = measurement.copy()
            linear_estimate = measurement.copy()
        elif self.last_observation is not None:
            # マーカーを見失っているとき：経過時間（最大 max_prediction_seconds まで）を計算
            elapsed = min(
                max(0.0, timestamp - self.last_observation_time),
                self.max_prediction_seconds,
            )
            hold_estimate = self.last_observation.copy()  # 予測なし：最後の位置で停止
            linear_estimate = self.last_observation + self.linear_velocity * elapsed  # 等速外挿
            linear_estimate[2] = wrap_angle(linear_estimate[2])
        else:
            # 起動してからまだ1度も見えていないとき
            hold_estimate = None
            linear_estimate = None

        ekf_estimate = self.ekf.step(measurement, timestamp, dt)
        return {
            "hold": hold_estimate,
            "linear": linear_estimate,
            "ekf": ekf_estimate,
        }


# -----------------------------------------------------------------------------
# マーカーの3次元位置・姿勢推定 (solvePnP)
# -----------------------------------------------------------------------------
def estimate_marker_pose(corners, camera_matrix, distortion, marker_real_width):
    """画像上のマーカー4隅の座標 (corners) から、カメラ基準の3次元位置とヨー角を求める。"""
    # 現実世界でのマーカーの4隅の3D座標 [m]（中心を原点 (0,0,0) とする正方形）
    half_width = marker_real_width / 2.0
    object_points = np.array([
        [-half_width, half_width, 0.0],
        [half_width, half_width, 0.0],
        [half_width, -half_width, 0.0],
        [-half_width, -half_width, 0.0],
    ], dtype=np.float32)
    # 画像上で検出された4隅の2Dピクセル座標を (4行, 2列) の形に整える
    image_points = np.asarray(corners, dtype=np.float32).reshape(4, 2)

    # PnP問題（3D点と2D点の対応からカメラに対する位置・向きを求める計算）を解く
    solved, rotation_vector, translation_vector = cv2.solvePnP(
        object_points,
        image_points,
        camera_matrix,
        distortion,
        flags=cv2.SOLVEPNP_IPPE_SQUARE,  # 正方形マーカー専用の高速・高精度な解法
    )
    if not solved:
        return None

    # 回転ベクトルを 3x3 の回転行列に変換し、水平面の向き（Yaw角）を取り出す
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    yaw = math.atan2(rotation_matrix[0, 2], -rotation_matrix[2, 2])
    position = translation_vector.reshape(3)  # [X(横), Y(縦), Z(奥行き)] [m]
    if not np.all(np.isfinite(position)) or position[2] <= 0:
        return None
    return position, yaw, rotation_vector, translation_vector


# -----------------------------------------------------------------------------
# PID制御 ＆ メカナムホイール逆運動学（車体速度 → 4輪の回転速度への変換）
# -----------------------------------------------------------------------------
def calculate_control(estimate, controllers, dt, target, mecanum):
    """推定位置 estimate [x, z, yaw] から車体速度指令 [vx, vy, omega] と4輪角速度を計算する。"""
    # 現在の推定位置と目標位置の差（誤差）を計算
    x_error = estimate[0] - target["x"]
    z_error = estimate[1] - target["z"]
    yaw_error = wrap_angle(estimate[2] - target["yaw"])

    # 3軸それぞれを PID 制御器に通して目標速度を計算
    vx = controllers[0].update(x_error, dt)       # 横移動速度 [m/s]
    vy = controllers[1].update(z_error, dt)       # 前後速度 [m/s]
    omega = controllers[2].update(yaw_error, dt)  # 旋回角速度 [rad/s]

    wheel_radius = mecanum["wheel_radius"]
    arm = mecanum["half_length"] + mecanum["half_width"]  # 旋回時の腕の長さ L + W

    def wheel_speeds(command):
        """メカナムホイールの逆運動学の公式：車体速度 → [左前FL, 右前FR, 左後RL, 右後RR] の角速度"""
        lateral, forward, rotation = command
        return np.array([
            (forward - lateral - arm * rotation) / wheel_radius,
            (forward + lateral + arm * rotation) / wheel_radius,
            (forward + lateral - arm * rotation) / wheel_radius,
            (forward - lateral + arm * rotation) / wheel_radius,
        ], dtype=np.float64)

    command = np.array([vx, vy, omega], dtype=np.float64)
    wheels = wheel_speeds(command)
    # もし4輪のどれかが上限角速度を超えたら、進行方向が狂わないように全体を同じ比率で縮小する
    max_wheel_speed = mecanum["max_wheel_speed"]
    scale = max(1.0, float(np.max(np.abs(wheels))) / max_wheel_speed)
    command /= scale
    wheels = wheel_speeds(command)
    return command, wheels


# -----------------------------------------------------------------------------
# 画面描画（GUI）用の補助関数群（--no-display 時は呼ばれない）
# -----------------------------------------------------------------------------
def draw_text(image, text, position, color=(235, 235, 235), scale=0.55):
    """画像上の指定座標 position=(x, y) に文字を描く。"""
    cv2.putText(
        image, text, position, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1,
        cv2.LINE_AA,
    )


def draw_graph(canvas, offset_x, histories, panel_width, target):
    """右パネル上部に、3モデルの軌跡を X-Z 平面グラフとして描画する。"""
    chart_left = offset_x + 42
    chart_top = 92
    chart_right = offset_x + panel_width - 18
    chart_bottom = 342
    graph_width = chart_right - chart_left
    graph_height = chart_bottom - chart_top

    # グラフの軸範囲が自動で収まるように、目標位置と全履歴の最小値・最大値を求める
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
        """メートル座標 (x, z) を画像上のピクセル座標 (px, py) に変換する。"""
        px = chart_left + int((x - min_x) / (max_x - min_x) * graph_width)
        py = chart_bottom - int((z - min_z) / (max_z - min_z) * graph_height)
        return px, py

    # 枠線とグリッド（補助線）を描画
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

    # 目標位置を白い星マークで描画
    target_pixel = to_pixel(target["x"], target["z"])
    cv2.drawMarker(
        canvas, target_pixel, (255, 255, 255), cv2.MARKER_STAR, 14, 2
    )
    draw_text(canvas, "target", (target_pixel[0] + 8, target_pixel[1] - 5),
              (220, 220, 220), 0.4)

    # 各モデルの軌跡を折れ線と最新点の丸で描画
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
    """速度指令などを左右に伸びるバーゲージとして描画する（中央が 0）。"""
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


# -----------------------------------------------------------------------------
# マーカー選択・安全停止・CSV整形・コマンドライン引数解析の関数群
# -----------------------------------------------------------------------------
def find_target_marker(corners, ids, target_id):
    """検出されたマーカーの中から、指定ID (既定: 0) のインデックスを探す。
    他のIDがより大きく映っていても無視し、見つからなければ None を返す。
    """
    if ids is None or len(corners) == 0:
        return None
    flat_ids = np.asarray(ids).reshape(-1)  # 2次元配列を1次元リスト状に平坦化
    # リスト内包表記：target_id と一致する番号 (index) だけをすべて集める
    matching_indices = [
        index for index, marker_id in enumerate(flat_ids)
        if int(marker_id) == target_id
    ]
    if not matching_indices:
        return None
    if len(matching_indices) == 1:
        return matching_indices[0]
    # 万が一同じIDが複数誤検出された場合は、その中で面積が一番大きいものを選ぶ
    areas = [
        abs(cv2.contourArea(corners[index].reshape(4, 2).astype(np.float32)))
        for index in matching_indices
    ]
    return matching_indices[int(np.argmax(areas))]


def is_safety_stop_active(measurement, last_observation_time, now, max_prediction_seconds):
    """安全停止を発動すべきか判定する。
    マーカーが見えず、最後に見えた時刻から max_prediction_seconds (2.0秒) 以上経過したら True。
    """
    if measurement is not None or last_observation_time is None:
        return False
    # 1e-9 (10億分の1秒) は小数の丸め誤差でギリギリ判定漏れするのを防ぐための補正
    return (now - last_observation_time) >= max_prediction_seconds - 1e-9


def format_csv_row(
    elapsed_time, status, measurement, ekf_estimate, selected_model, command, wheel_speeds
):
    """1フレーム分のデータを CSV の1行分（文字列のリスト）に整形して返す。"""
    # 未観測 (None) のときは空文字 "" にしてCSVの列ズレを防ぐ
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
    # 「*obs_values」はリストの中身を展開して並べる書き方（アンパック）
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
    """実行時のコマンドラインオプション（--video や --no-display 等）を解析する。"""
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
        action="store_true",  # オプションが書かれたら True、書かれなければ False になる
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


# -----------------------------------------------------------------------------
# メイン処理：プログラム起動時にここから実行される
# -----------------------------------------------------------------------------
def main(argv=None):
    # 1. コマンドライン引数と設定ファイル (config.json) を読み込む
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

    # 2. 映像入力を開く（--video があれば動画ファイル、なければ実カメラ）
    is_video_mode = args.video is not None
    if is_video_mode:
        if not args.video.exists():
            print(f"エラー: 動画ファイルが見つかりません: {args.video}")
            return
        cap = cv2.VideoCapture(str(args.video))
        if not cap.isOpened():
            print(f"エラー: 動画ファイルをオープンできませんでした: {args.video}")
            return
        # 動画モードでは実際のPC時計ではなく、動画のFPSから1フレームあたりの時間 dt を決める
        raw_fps = cap.get(cv2.CAP_PROP_FPS)
        video_fps = (
            raw_fps if raw_fps and math.isfinite(raw_fps) and raw_fps > 0 else 30.0
        )
        video_dt = 1.0 / video_fps
    else:
        # Windowsでは DirectShow (CAP_DSHOW) を使うと外付けカメラ(index=1等)が安定して開ける。
        # ラズパイ(Linux)へ移植したときにエラーにならないよう OS で分岐する。
        if platform.system() == "Windows":
            cap = cv2.VideoCapture(camera_config["index"], cv2.CAP_DSHOW)
        else:
            cap = cv2.VideoCapture(camera_config["index"])
        if not cap.isOpened():
            print("エラー: カメラをオープンできませんでした。インデックスを確認してください。")
            return
        video_dt = None

    # 3. CSVログファイルを準備する（--no-log が付いていなければ作成）
    log_file = None
    log_writer = None
    log_path = None
    if not args.no_log:
        if args.log is not None:
            log_path = args.log
        else:
            # ファイル名が重複しないように現在時刻（例: 20261119_134055）を付ける
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_path = Path(f"aruco_log_{timestamp_str}.csv")
        try:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_file = log_path.open("w", newline="", encoding="utf-8")
            log_writer = csv.writer(log_file)
            log_writer.writerow(CSV_HEADER)  # 1行目に列名を書き込む
        except OSError as error:
            print(f"エラー: CSVログファイルを作成できません: {error}")
            cap.release()
            return

    # 4. ArUco検出器・予測器・PID制御器を初期化する
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)
    predictors = PosePredictors(max_prediction_seconds)
    # 画面表示ONのときだけ軌跡履歴 (deque) をメモリに確保する
    histories = (
        {
            name: deque(maxlen=display_config["history_length"])
            for name in MODEL_NAMES
        }
        if not args.no_display
        else None
    )
    # 「**config["pid"]["x"]」は辞書の中身を kp=..., ki=... と引数に展開して渡す書き方
    controllers = [
        PIDController(**config["pid"]["x"]),
        PIDController(**config["pid"]["z"]),
        PIDController(**config["pid"]["yaw"]),
    ]
    selected_model = args.model
    start_time = time.monotonic()  # 時刻の巻き戻りが起きない単調増加タイマー
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

    # 5. メインループ（1フレームごとに「検出 → 推定 → 制御 → 記録 → 描画」を繰り返す）
    # try ... finally を使うと、途中でエラーや Ctrl+C が起きても必ず後片付けが実行される
    try:
        while True:
            ret, frame = cap.read()  # カメラまたは動画から1枚画像 (frame) を読み込む
            if not ret:
                if is_video_mode:
                    print(f"動画ファイルの処理が完了しました ({frame_index} フレーム)。")
                else:
                    print("エラー: フレームを取得できませんでした。")
                break

            # 経過時間とフレーム間の時間幅 dt [秒] を計算
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

            # カメラ内部行列（焦点距離と画像中心）を作成
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

            # 画像から ArUco マーカーを検出し、指定ID (0) の位置・向きを推定
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

            # 観測値 [X, Z, Yaw] を作り、3つの予測モデルすべてを更新する
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

            # 安全停止の判定（見失いが max_prediction_seconds 以上続いたら停止）
            safety_stop = is_safety_stop_active(
                measurement,
                predictors.last_observation_time,
                now,
                max_prediction_seconds,
            )

            # 選択中モデルの推定値を使って PID 制御と車輪角速度を計算
            estimate = estimates[selected_model]
            if safety_stop:
                # 安全停止に入った瞬間だけ PID の積分履歴をリセットする
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

            # CSVログに1フレーム分のデータを書き込む
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

            # --no-display が指定されていれば、以降の画面描画をすべて飛ばして次のフレームへ
            if args.no_display:
                continue

            # --- ここから画面描画処理 ---
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

            # 左側にカメラ映像、右側に情報パネルを並べた1枚の大きな画像 (canvas) を作る
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

            # ウィンドウに画像を表示し、キーボード入力を1ミリ秒待つ
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
        # 終了時の後片付け：カメラ解放、ウィンドウを閉じる、CSVファイルを安全に閉じる
        cap.release()
        if not args.no_display:
            cv2.destroyAllWindows()
        if log_file is not None:
            log_file.close()
            print(f"CSVログを保存しました: {log_path}")


# このファイルを直接「python aruco_tracker.py」で実行したときだけ main() を動かす。
# 別のファイル（test_PID.py や simulation_ekf.py）から import されたときは main() を動かさない。
if __name__ == "__main__":
    main()
