
from pathlib import Path
import numpy as np

from aruco_tracker import (
    load_config,
    PIDController,
    calculate_control,
)

# 既存の設定を読み込む
config = load_config(
    Path(__file__).with_name("config.json")
)

# 仮想的なマーカー位置
# (横方向X, 前方距離Z, マーカー角度[度])
test_cases = [
    ("目標位置", 0.0, 0.5, 0),
    ("目標より30cm遠い", 0.0, 0.8, 0),
    ("目標より20cm右", 0.2, 0.5, 0),
    ("マーカー角度+10度", 0.0, 0.5, 10),
]

for name, x, z, yaw_deg in test_cases:
    # 各テストでPIDの履歴を初期化
    controllers = [
        PIDController(**config["pid"][axis])
        for axis in ("x", "z", "yaw")
    ]

    estimate = np.array([
        x, z, np.deg2rad(yaw_deg)
    ])

    command, wheels = calculate_control(
        estimate,
        controllers,
        0.1,  # 計算周期0.1秒
        config["target"],
        config["mecanum"],
    )

    print(f"\n【{name}】")
    print(f"横移動速度: {command[0]:+.4f} m/s")
    print(f"前後速度:   {command[1]:+.4f} m/s")
    print(f"旋回速度:   {command[2]:+.4f} rad/s")
    print(f"4輪の角速度: {np.round(wheels, 3)}")
