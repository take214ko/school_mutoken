# 評価指標および正解値（Ground Truth）取得方法の設計書

本ドキュメントは、ArUcoマーカーを用いた視覚追従・EKF予測・メカナムホイールPID制御システム（`aruco_tracker.py` / `simulation_AB.py` / `simulation_ekf.py`）の実験評価（論文第5節相当の評価実験）に向けて、**定量評価指標の定義**と**正解値（Ground Truth）の取得方法**を整理したものです。

---

## 1. 評価の3段階構成

本研究では、再現性の高いオフライン検証から実機評価へ段階的に移行します。

| 段階 | 実験環境 | 対象スクリプト | 主な検証内容 |
| :--- | :--- | :--- | :--- |
| **Stage 1: シミュレーション** | 仮想軌道＋模擬ノイズ | `simulation_AB.py`, `simulation_ekf.py` | 方式A/B比較、0.5秒遮蔽時の `hold` / `linear` / `ekf` 精度比較 |
| **Stage 2: 録画動画ベンチマーク** | 同一録画動画 (`--video --no-display`) | `aruco_tracker.py` | 照明変動・手ブレ・実遮蔽下でのモデル間公平比較、処理負荷測定 |
| **Stage 3: 実機追従実験** | 実ロボット＋実カメラ | `aruco_tracker.py` | 実床面での追従精度・遅れ・安全停止動作の総合評価 |

---

## 2. 評価指標の定義

ロボットの視覚追従・状態推定に関する実験評価（参考論文第5節の標準的な構成）に基づき、以下の4カテゴリで定量評価を行います。

### 2.1 位置・姿勢精度（Accuracy）
1. **横方向・距離方向・平面位置 RMSE（Root Mean Square Error）**
   $$\mathrm{RMSE}_X = \sqrt{\frac{1}{N}\sum_{k=1}^{N} (x_k - x_k^{\mathrm{gt}})^2}, \quad \mathrm{RMSE}_{XZ} = \sqrt{\frac{1}{N}\sum_{k=1}^{N} \left\{(x_k - x_k^{\mathrm{gt}})^2 + (z_k - z_k^{\mathrm{gt}})^2\right\}}$$
   - **推定誤差 RMSE（EstRMSE）**: カメラ座標系での推定値 $(\hat{x}, \hat{z})$ と真の相対位置 $(x^{\mathrm{gt}}, z^{\mathrm{gt}})$ の差。全区間（All）と遮蔽区間（Occlusion）を分けて算出する。
   - **追従誤差 RMSE（TrackRMSE）**: ワールド座標系でのロボット位置と目標追従位置（対象の正面距離 $Z_{\text{target}}=0.5\text{ m}$）との差。
2. **姿勢角 RMSE（$\mathrm{RMSE}_{\mathrm{yaw}}$）**
   - 角度の折り返し（$\pm\pi$）を考慮した `wrap_angle` 後の誤差で算出（単位: `deg`）。
3. **最大誤差（Max Error）**
   - 遮蔽中や急加減速時の最悪ケース（衝突・見失いリスク）を把握するために記録する。

### 2.2 時間応答性・遅れ（Latency / Lag）
1. **追従時間遅れ（Tracking Lag, $\tau_{\mathrm{lag}}$）**
   - 対象軌道 $x_{\mathrm{target}}(t)$ とロボット軌道 $x_{\mathrm{robot}}(t)$ の相互相関（Cross-Correlation）が最大となるシフト時間 $\tau$ を算出（単位: `s`）。
   - `simulation_ekf.py` の `estimate_lag_seconds()` で実装済み。
2. **遮蔽復帰時間（Recovery Time）**
   - マーカー再検出後、推定誤差または追従誤差が許容しきい値（例: $0.03\text{ m}$）以下に収束するまでの経過時間（単位: `s`）。
3. **システム処理遅延・フレームレート（FPS / Processing Time）**
   - ラズパイ等の実機上で `--no-display` 有無による1フレーム処理時間（ms）と実効FPSを比較する。

### 2.3 ロバスト性・検出安定性（Robustness）
1. **マーカーロスト率（Lost Rate）**
   $$\mathrm{Lost\ Rate\ [\%]} = \frac{\text{未検出フレーム数（`obs_x_m` が空欄）}}{\text{全フレーム数}} \times 100$$
2. **最大連続見失い時間（Max Continuous Occlusion）**
   - 連続して未検出となった最長時間（秒）。予測上限 `max_seconds = 2.0 s` との関係を確認する。
3. **安全停止発動回数・停止時間（Safety Stop Count / Duration）**
   - CSVログの `status == "SAFETY_STOP"` の発生回数および該当区間で車輪角速度指令が確実に `0.0 rad/s` となっていることの確認。
4. **誤検出・対象切替率（ID Switch Rate）**
   - 複数マーカー（ID 0以外）を同時配置した条件で、ID 0以外へ切り替わった回数（ID 0固定化により `0回` であることを検証）。

### 2.4 制御の滑らかさ・安全性（Control Smoothness）
1. **制御入力変動量（Control Variation / Jerk 指標）**
   - 速度指令 $(V_x, V_y, \omega)$ のフレーム間差分のRMS。遮蔽復帰時の急激なスパイク有無を評価する。
2. **車輪速度飽和率（Wheel Saturation Rate）**
   - 4輪のいずれかが `max_wheel_speed_rad_s`（`12.0 rad/s`）に達したフレームの割合。

---

## 3. 正解値（Ground Truth）の取得方法の設計

実験環境とコスト制約に応じて、以下の3段階の取得方法を定義します。本研究の学校・研究室環境では **方式B（俯瞰固定カメラ＋基準マーカー）** を主軸とし、簡易検証に **方式A（床面グリッド＋定速ガイド）** を併用する構成を推奨します。

### 方式比較表

| 方式 | 概要 | 想定精度 | 導入コスト | 適用フェーズ |
| :--- | :--- | :--- | :--- | :--- |
| **シミュレーション真値** | 数式軌道から直接取得 | 誤差0（厳密値） | なし | Stage 1（アルゴリズム比較） |
| **方式A: 床面ガイド＋定速移動** | 床面テープ・目盛り＋一定速度（メトロノーム/電動レール）で移動 | $\pm 1\text{〜}3\text{ cm}$ | 極低 | Stage 2〜3（簡易定性・定量検証） |
| **方式B: 俯瞰固定カメラ＋ArUco（推奨）** | 天井・三脚の固定カメラで床面基準マーカー4点＋ロボット上面マーカー＋対象マーカーを同時撮影しホモグラフィ変換 | $\pm 0.5\text{〜}1.5\text{ cm}$ | 低（既存カメラ＋印刷マーカー） | Stage 2〜3（本実験の主Ground Truth） |
| **方式C: モーションキャプチャ** | OptiTrack / Vicon 等による赤外線反射マーカ追跡 | $< 1\text{ mm}$ | 高（設備依存） | 設備利用可能な場合 |

### 推奨手法（方式B: 俯瞰固定カメラ＋ホモグラフィ変換）の具体設計

1. **マーカー配置（ID割り当てルール）**
   - **ID 0**: 追従対象（ターゲット）前面に貼付（ロボット搭載カメラが追跡）
   - **ID 10**: 追従対象の上面に貼付（俯瞰カメラ用 Ground Truth）
   - **ID 11**: ロボット上面に貼付（俯瞰カメラ用 Ground Truth）
   - **ID 20〜23**: 実験エリア四隅（例: $2.0\text{ m} \times 2.0\text{ m}$ の正方形）の床面に固定（ホモグラフィ変換の基準点）
2. **座標変換手順**
   - 俯瞰カメラ映像から床面基準マーカー（ID 20〜23）の中心ピクセル座標を検出し、既知のワールド座標 $(0,0), (2,0), (2,2), (0,2)$ への射影変換行列 $H$（`cv2.getPerspectiveTransform` / `cv2.findHomography`）を算出する。
   - 各フレームで検出した ID 10（対象）と ID 11（ロボット）のピクセル座標に $H$ を適用し、ワールド座標系での正解位置 $(x_{\mathrm{target}}^{\mathrm{gt}}, z_{\mathrm{target}}^{\mathrm{gt}})$, $(x_{\mathrm{robot}}^{\mathrm{gt}}, z_{\mathrm{robot}}^{\mathrm{gt}})$ を取得する。
3. **時刻同期（Synchronization）**
   - 実験開始時に「LED点滅」または「基準マーカーの瞬間遮蔽（合図）」をロボットカメラ・俯瞰カメラの双方に映し込み、そのフレームを $t = 0.0\text{ s}$ としてタイムスタンプを揃える（同一PCで2台のカメラを録画する場合は `time.monotonic()` を共有）。

---

## 4. `aruco_tracker.py` CSVログと評価指標の対応

`aruco_tracker.py` が出力するCSVログ列から、各指標を以下のように直接集計できます。

| CSV列名 | 算出に用いる評価指標 |
| :--- | :--- |
| `time_s` | 時系列軸、追従遅れ $\tau_{\mathrm{lag}}$、遮蔽復帰時間、実効FPS |
| `status` | 安全停止発動回数 (`SAFETY_STOP`)、予測区間 (`PREDICTING`) の抽出 |
| `obs_x_m`, `obs_z_m`, `obs_yaw_rad` | ロスト率（空欄割合）、観測ノイズ標準偏差、最大連続見失い時間 |
| `ekf_x_m`, `ekf_z_m`, `ekf_yaw_rad` | 推定誤差 RMSE / Max Error（Ground Truthとの差分） |
| `cmd_vx_m_s`, `cmd_vy_m_s`, `cmd_omega_rad_s` | 制御入力変動量、安全停止時のゼロ出力検証 |
| `wheel_fl_rad_s` 〜 `wheel_rr_rad_s` | 車輪速度飽和率、安全停止時の全輪 `0.0 rad/s` 検証 |

---

## 5. 参考：0.5秒遮蔽シミュレーション（`simulation_ekf.py`）のベースライン結果

`simulation_ekf.py`（$DT=0.05\text{ s}$, 12秒間, 0.5秒遮蔽×3回 = ロスト率 $12.4\%$, 観測ノイズ $\sigma_{x,z}=0.015\text{ m}, \sigma_{\mathrm{yaw}}=2^\circ$）での実行結果は以下の通りです。

| モデル | 推定RMSE (全区間) | 推定RMSE (0.5秒遮蔽中) | 推定最大誤差 (遮蔽中) | 追従RMSE (全区間) | 追従遅れ (Lag) |
| :--- | ---: | ---: | ---: | ---: | ---: |
| `1: No prediction` | `0.0400 m` | `0.0992 m` | `0.1893 m` | `0.1638 m` | `0.90 s` |
| `2: Linear` | `0.0341 m` | `0.0793 m` | `0.1645 m` | `0.1597 m` | `0.85 s` |
| `3: EKF` | **`0.0152 m`** | **`0.0205 m`** | **`0.0351 m`** | **`0.1540 m`** | **`0.80 s`** |

- **知見**: 0.5秒遮蔽中の推定RMSEは、EKFが予測なし（`hold`）比で **約79%低減**、線形外挿（`linear`）比でも **約74%低減**しており、実機実験における有効性のベースラインとなります。
