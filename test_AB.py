
import math
from pathlib import Path
import numpy as np

from aruco_tracker import (
    load_config,
    PIDController,
    calculate_control,
)

config = load_config(Path(__file__).with_name("config.json"))

# 仮想的な対象の位置
# 正面50cm、右20cm
x = 0.20
z = 0.50
dt = 0.1

target = config["target"]

# 現在の設定では X目標=0、Yaw目標=0
assert abs(target["x"]) < 1e-9
assert abs(target["yaw"]) < 1e-9

# 方式A：横移動を使わず、距離と方向角で制御
distance = math.hypot(x, z)
bearing = math.atan2(x, z)
estimate_A = np.array([0.0, distance, bearing])

# 方式B：前後・左右の位置誤差で制御
estimate_B = np.array([x, z, 0.0])

for name, estimate in [
    ("方式A：前進・旋回", estimate_A),
    ("方式B：全方向移動", estimate_B),
]:
    controllers = [
        PIDController(**config["pid"][axis])
        for axis in ("x", "z", "yaw")
    ]

    command, wheels = calculate_control(
        estimate,
        controllers,
        dt,
        target,
        config["mecanum"],
    )

    print(f"\n【{name}】")
    print(f"横方向速度: {command[0]:+.4f} m/s")
    print(f"前後方向速度: {command[1]:+.4f} m/s")
    print(f"旋回速度: {command[2]:+.4f} rad/s")
    print(f"車輪角速度: {np.round(wheels, 3)}")
