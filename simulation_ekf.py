import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from aruco_tracker import (
    MODEL_NAMES,
    PIDController,
    PosePredictors,
    calculate_control,
    load_config,
    wrap_angle,
)

DT = 0.05  # 計算周期 50 ms (20 Hz)
DURATION = 12.0  # 実験時間 12 秒
OCCLUSION_WINDOWS = [
    (2.0, 2.5),  # 0.5秒遮蔽（移動中）
    (5.5, 6.0),  # 0.5秒遮蔽（折り返し後・旋回含む）
    (9.0, 9.5),  # 0.5秒遮蔽（復帰移動中）
]

# 観測ノイズ（カメラ・solvePnP由来の誤差を模擬）
NOISE_STD_X = 0.015  # m
NOISE_STD_Z = 0.015  # m
NOISE_STD_YAW = math.radians(2.0)  # rad
RANDOM_SEED = 42


def target_world_pose(t, base_distance):
    """対象の真値軌道 (x, z, yaw)。左右移動＋わずかな前後・角度変化を含む。"""
    x = 0.30 * math.sin(2.0 * math.pi * t / 6.0)
    z = base_distance + 0.08 * math.sin(2.0 * math.pi * t / 12.0)
    vx = 0.30 * (2.0 * math.pi / 6.0) * math.cos(2.0 * math.pi * t / 6.0)
    vz = 0.08 * (2.0 * math.pi / 12.0) * math.cos(2.0 * math.pi * t / 12.0)
    # 進行方向に応じてわずかに向きが変わる（最大約±15度）
    yaw = math.radians(15.0) * math.sin(2.0 * math.pi * t / 6.0)
    return x, z, yaw, vx, vz


def is_occluded(t, windows=OCCLUSION_WINDOWS):
    return any(start <= t < end for start, end in windows)


def estimate_lag_seconds(reference, signal, dt, max_lag_s=1.0):
    """相互相関により信号の追従遅れ時間[s]を推定する（正の値が遅れ）。"""
    ref = np.asarray(reference, dtype=np.float64) - np.mean(reference)
    sig = np.asarray(signal, dtype=np.float64) - np.mean(signal)
    if np.std(ref) < 1e-9 or np.std(sig) < 1e-9:
        return 0.0
    max_shift = min(int(round(max_lag_s / dt)), len(ref) - 1)
    best_shift = 0
    best_corr = -np.inf
    for shift in range(-max_shift, max_shift + 1):
        if shift >= 0:
            r = ref[: len(ref) - shift]
            s = sig[shift:]
        else:
            r = ref[-shift:]
            s = sig[: len(sig) + shift]
        corr = float(np.dot(r, s)) / max(len(r), 1)
        if corr > best_corr:
            best_corr = corr
            best_shift = shift
    return best_shift * dt


def compute_recovery_time(times, errors, windows, threshold=0.03):
    """各遮蔽終了後、誤差が threshold 以下に戻るまでの平均復帰時間[s]。"""
    recovery_times = []
    for _, end_t in windows:
        after_indices = np.where(times >= end_t)[0]
        recovered = None
        for idx in after_indices:
            if errors[idx] <= threshold:
                recovered = times[idx] - end_t
                break
        if recovered is not None:
            recovery_times.append(recovered)
    return float(np.mean(recovery_times)) if recovery_times else float("nan")


def simulate_model(model_name, config, noise_sequence):
    """指定モデル（hold / linear / ekf）で0.5秒遮蔽ありの閉ループ追従シミュレーションを実行。"""
    target = config["target"]
    mecanum = config["mecanum"]
    base_distance = target["z"]

    predictors = PosePredictors(config["max_prediction_seconds"])
    controllers = [
        PIDController(**config["pid"][axis]) for axis in ("x", "z", "yaw")
    ]

    # 仮想ロボット（メカナム全方向移動）の初期状態
    robot_x = 0.0
    robot_z = 0.0
    robot_yaw = 0.0

    steps = round(DURATION / DT)
    records = []

    for i in range(steps + 1):
        t = i * DT
        tx, tz, tyaw, _, _ = target_world_pose(t, base_distance)

        # ロボット座標系から見た対象の真の相対位置・姿勢
        dx = tx - robot_x
        dz = tz - robot_z
        true_rel_x = math.cos(robot_yaw) * dx - math.sin(robot_yaw) * dz
        true_rel_z = math.sin(robot_yaw) * dx + math.cos(robot_yaw) * dz
        true_rel_yaw = wrap_angle(tyaw - robot_yaw)

        occluded = is_occluded(t)
        if occluded:
            measurement = None
        else:
            noise = noise_sequence[i]
            measurement = np.array(
                [
                    true_rel_x + noise[0],
                    true_rel_z + noise[1],
                    wrap_angle(true_rel_yaw + noise[2]),
                ],
                dtype=np.float64,
            )

        estimates = predictors.update(measurement, t, DT)
        estimate = estimates[model_name]

        if estimate is not None:
            # 方式B（全方向移動）：相対X, Z誤差で追従（Yawは正面維持）
            control_input = np.array([estimate[0], estimate[1], 0.0], dtype=np.float64)
            command, wheels = calculate_control(
                control_input, controllers, DT, target, mecanum
            )
            est_err_x = estimate[0] - true_rel_x
            est_err_z = estimate[1] - true_rel_z
            est_err_yaw = wrap_angle(estimate[2] - true_rel_yaw)
            est_err_xz = math.hypot(est_err_x, est_err_z)
        else:
            command = np.zeros(3, dtype=np.float64)
            wheels = np.zeros(4, dtype=np.float64)
            est_err_x = est_err_z = est_err_yaw = est_err_xz = float("nan")

        track_err_x = tx - robot_x
        track_err_z = (tz - base_distance) - robot_z
        track_err_xz = math.hypot(track_err_x, track_err_z)

        records.append(
            {
                "model": model_name,
                "time_s": t,
                "occluded": int(occluded),
                "target_x_m": tx,
                "target_z_m": tz,
                "target_yaw_rad": tyaw,
                "true_rel_x_m": true_rel_x,
                "true_rel_z_m": true_rel_z,
                "true_rel_yaw_rad": true_rel_yaw,
                "obs_x_m": "" if measurement is None else measurement[0],
                "obs_z_m": "" if measurement is None else measurement[1],
                "obs_yaw_rad": "" if measurement is None else measurement[2],
                "est_x_m": "" if estimate is None else estimate[0],
                "est_z_m": "" if estimate is None else estimate[1],
                "est_yaw_rad": "" if estimate is None else estimate[2],
                "est_err_xz_m": est_err_xz,
                "est_err_yaw_deg": math.degrees(est_err_yaw),
                "robot_x_m": robot_x,
                "robot_z_m": robot_z,
                "track_err_x_m": track_err_x,
                "track_err_z_m": track_err_z,
                "track_err_xz_m": track_err_xz,
                "cmd_vx_m_s": command[0],
                "cmd_vy_m_s": command[1],
                "cmd_omega_rad_s": command[2],
            }
        )

        if i == steps:
            break

        vx, vz, omega = command
        robot_x += (vx * math.cos(robot_yaw) + vz * math.sin(robot_yaw)) * DT
        robot_z += (-vx * math.sin(robot_yaw) + vz * math.cos(robot_yaw)) * DT
        robot_yaw = wrap_angle(robot_yaw + omega * DT)

    return records


def summarize_metrics(records):
    times = np.array([r["time_s"] for r in records])
    occluded = np.array([bool(r["occluded"]) for r in records])
    est_err_xz = np.array([r["est_err_xz_m"] for r in records], dtype=np.float64)
    est_err_yaw = np.array([r["est_err_yaw_deg"] for r in records], dtype=np.float64)
    track_err_xz = np.array([r["track_err_xz_m"] for r in records], dtype=np.float64)
    target_x = np.array([r["target_x_m"] for r in records], dtype=np.float64)
    robot_x = np.array([r["robot_x_m"] for r in records], dtype=np.float64)

    occ_mask = occluded & np.isfinite(est_err_xz)
    all_mask = np.isfinite(est_err_xz)

    return {
        "lost_rate_pct": 100.0 * float(np.mean(occluded)),
        "est_rmse_all_m": float(np.sqrt(np.mean(est_err_xz[all_mask] ** 2))),
        "est_rmse_occ_m": float(np.sqrt(np.mean(est_err_xz[occ_mask] ** 2))),
        "est_max_occ_m": float(np.max(est_err_xz[occ_mask])),
        "est_yaw_rmse_occ_deg": float(np.sqrt(np.mean(est_err_yaw[occ_mask] ** 2))),
        "track_rmse_all_m": float(np.sqrt(np.mean(track_err_xz ** 2))),
        "track_rmse_occ_m": float(np.sqrt(np.mean(track_err_xz[occluded] ** 2))),
        "track_max_m": float(np.max(track_err_xz)),
        "lag_s": estimate_lag_seconds(target_x, robot_x, DT),
        "recovery_s": compute_recovery_time(times, est_err_xz, OCCLUSION_WINDOWS, threshold=0.03),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="0.5秒遮蔽（見失い）条件における hold / linear / EKF の比較シミュレーション"
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="グラフウィンドウを表示せずにPNG保存のみ行う",
    )
    args = parser.parse_args(argv)

    config = load_config(Path(__file__).with_name("config.json"))
    steps = round(DURATION / DT)
    rng = np.random.default_rng(RANDOM_SEED)
    noise_sequence = np.column_stack(
        [
            rng.normal(0.0, NOISE_STD_X, size=steps + 1),
            rng.normal(0.0, NOISE_STD_Z, size=steps + 1),
            rng.normal(0.0, NOISE_STD_YAW, size=steps + 1),
        ]
    )

    models = ("hold", "linear", "ekf")
    all_records = {}
    summaries = {}
    for model_name in models:
        records = simulate_model(model_name, config, noise_sequence)
        all_records[model_name] = records
        summaries[model_name] = summarize_metrics(records)

    print("=== 0.5秒遮蔽（見失い）条件シミュレーション結果 ===")
    print(f"遮蔽区間: {OCCLUSION_WINDOWS} (各0.5秒, ロスト率 {summaries['ekf']['lost_rate_pct']:.1f}%)")
    print(
        f"{'Model':<18} | {'EstRMSE(All)':>12} | {'EstRMSE(Occ)':>12} | "
        f"{'EstMax(Occ)':>11} | {'TrackRMSE(All)':>14} | {'TrackRMSE(Occ)':>14} | {'Lag[s]':>6}"
    )
    print("-" * 102)
    for model_name in models:
        s = summaries[model_name]
        print(
            f"{MODEL_NAMES[model_name]:<18} | "
            f"{s['est_rmse_all_m']:>10.4f} m | "
            f"{s['est_rmse_occ_m']:>10.4f} m | "
            f"{s['est_max_occ_m']:>9.4f} m | "
            f"{s['track_rmse_all_m']:>12.4f} m | "
            f"{s['track_rmse_occ_m']:>12.4f} m | "
            f"{s['lag_s']:>6.2f}"
        )

    csv_path = Path(__file__).with_name("simulation_ekf.csv")
    fieldnames = list(all_records["ekf"][0].keys())
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for model_name in models:
            writer.writerows(all_records[model_name])
    print(f"\nCSV保存先: {csv_path}")

    # グラフ1: ターゲット軌道と各モデルのロボット追従軌道（横方向X）
    times = [r["time_s"] for r in all_records["ekf"]]
    target_x = [r["target_x_m"] for r in all_records["ekf"]]
    colors = {"hold": "#e67e22", "linear": "#2ecc71", "ekf": "#2980b9"}

    plt.figure(figsize=(10, 5))
    for start_t, end_t in OCCLUSION_WINDOWS:
        plt.axvspan(start_t, end_t, color="gray", alpha=0.22, label="Occlusion (0.5s)" if start_t == OCCLUSION_WINDOWS[0][0] else None)
    plt.plot(times, target_x, "k--", linewidth=1.6, label="Target (Ground Truth)")
    for model_name in models:
        rx = [r["robot_x_m"] for r in all_records[model_name]]
        plt.plot(times, rx, color=colors[model_name], label=MODEL_NAMES[model_name])
    plt.xlabel("Time [s]")
    plt.ylabel("Horizontal position X [m]")
    plt.title("Tracking Performance under 0.5s Occlusions")
    plt.grid(True, alpha=0.4)
    plt.legend()
    plt.tight_layout()
    tracking_png = Path(__file__).with_name("ekf_occlusion_tracking.png")
    plt.savefig(tracking_png, dpi=150)

    # グラフ2: 推定誤差と追従誤差の比較（2段）
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for ax in axes:
        for start_t, end_t in OCCLUSION_WINDOWS:
            ax.axvspan(start_t, end_t, color="gray", alpha=0.22, label="Occlusion (0.5s)" if start_t == OCCLUSION_WINDOWS[0][0] else None)
        ax.grid(True, alpha=0.4)

    for model_name in models:
        est_err = [r["est_err_xz_m"] for r in all_records[model_name]]
        trk_err = [r["track_err_xz_m"] for r in all_records[model_name]]
        axes[0].plot(times, est_err, color=colors[model_name], label=MODEL_NAMES[model_name])
        axes[1].plot(times, trk_err, color=colors[model_name], label=MODEL_NAMES[model_name])

    axes[0].set_ylabel("Pose estimation error [m]")
    axes[0].set_title("Pose Estimation Error (X-Z Euclidean) under 0.5s Occlusions")
    axes[0].legend()

    axes[1].set_xlabel("Time [s]")
    axes[1].set_ylabel("Tracking error [m]")
    axes[1].set_title("Closed-Loop Tracking Error (X-Z Euclidean)")
    axes[1].legend()

    plt.tight_layout()
    error_png = Path(__file__).with_name("ekf_occlusion_error.png")
    plt.savefig(error_png, dpi=150)
    print(f"グラフ保存先: {tracking_png}, {error_png}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
