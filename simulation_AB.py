
import csv
import math
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from aruco_tracker import (
    load_config,
    PIDController,
    calculate_control,
)

config = load_config(Path(__file__).with_name("config.json"))

DT = 0.05       # 計算周期 50 ms
DURATION = 12.0 # 実験時間 12秒
SPEED = 0.1     # 対象速度 0.1 m/s
DISTANCE = config["target"]["z"]  # 現在は0.5 m


def target_x(t):
    """対象が左右30cmを移動する軌道"""
    if t <= 3.0:
        return -SPEED * t
    if t <= 9.0:
        return -0.3 + SPEED * (t - 3.0)
    return 0.3 - SPEED * (t - 9.0)


def simulate(mode):
    controllers = [
        PIDController(**config["pid"][axis])
        for axis in ("x", "z", "yaw")
    ]

    # 仮想ロボットの初期位置と向き
    robot_x = 0.0
    robot_z = 0.0
    yaw = 0.0

    results = []

    steps = round(DURATION / DT)

    for i in range(steps + 1):
        t = i * DT

        # 対象のワールド座標
        tx = target_x(t)
        tz = DISTANCE

        # 対象とロボットの位置差
        dx = tx - robot_x
        dz = tz - robot_z

        # ロボットから見た対象位置
        relative_x = math.cos(yaw) * dx - math.sin(yaw) * dz
        relative_z = math.sin(yaw) * dx + math.cos(yaw) * dz

        if mode == "A":
            # 前進・旋回方式
            distance = math.hypot(relative_x, relative_z)
            bearing = math.atan2(relative_x, relative_z)

            estimate = np.array([0.0, distance, bearing])
        else:
            # 全方向移動方式
            estimate = np.array([relative_x, relative_z, 0.0])

        # 既存のPIDと車輪速度制限を使用
        command, wheels = calculate_control(
            estimate,
            controllers,
            DT,
            config["target"],
            config["mecanum"],
        )

        vx, vz, omega = command

        # 実験データを保存
        results.append([
            t,
            tx,
            robot_x,
            robot_z,
            yaw,
            tx - robot_x,
            math.hypot(dx, dz) - DISTANCE,
        ])

        # 最終時刻では位置更新しない
        if i == steps:
            break

        # 仮想ロボットの位置を更新
        robot_x += (
            vx * math.cos(yaw) + vz * math.sin(yaw)
        ) * DT

        robot_z += (
            -vx * math.sin(yaw) + vz * math.cos(yaw)
        ) * DT

        yaw += omega * DT

    return np.asarray(results)


# 方式A・Bをシミュレーション
data_A = simulate("A")
data_B = simulate("B")

# 横方向追従誤差のRMSE
for name, data in [("A", data_A), ("B", data_B)]:
    rmse = np.sqrt(np.mean(data[:, 5] ** 2))
    print(f"方式{name} 横方向RMSE: {rmse:.4f} m")

# CSV保存
with open("simulation_AB.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow([
        "mode", "time_s", "target_x_m", "robot_x_m",
        "robot_z_m", "yaw_rad", "x_error_m",
        "distance_error_m"
    ])
    for name, data in [("A", data_A), ("B", data_B)]:
        for row in data:
            writer.writerow([name, *row])

# グラフ1：対象とロボットの横位置
plt.figure(figsize=(9, 5))
plt.plot(data_A[:, 0], data_A[:, 1], "k--", label="Target")
plt.plot(data_A[:, 0], data_A[:, 2], label="Method A")
plt.plot(data_B[:, 0], data_B[:, 2], label="Method B")
plt.xlabel("Time [s]")
plt.ylabel("Horizontal position X [m]")
plt.title("Target Tracking Simulation")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("tracking_AB.png", dpi=150)

# グラフ2：横方向追従誤差
plt.figure(figsize=(9, 5))
plt.plot(data_A[:, 0], np.abs(data_A[:, 5]), label="Method A")
plt.plot(data_B[:, 0], np.abs(data_B[:, 5]), label="Method B")
plt.xlabel("Time [s]")
plt.ylabel("Absolute horizontal error [m]")
plt.title("Tracking Error Comparison")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("error_AB.png", dpi=150)

plt.show()
