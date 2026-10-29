import cv2
import numpy as np

def main():
    # 1. カメラオープン (内蔵Webカメラ: 0)
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("エラー: カメラをオープンできませんでした。インデックスを確認してください。")
        return

    # 2. ArUco辞書および検出器の設定 (OpenCV 5.0.0対応)
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)

    # 3. パラメータ設定
    MARKER_REAL_WIDTH = 0.08  # マーカー一辺の実測サイズ [m] (8cm)
    FOCAL_LENGTH_PIXEL = 1000  # 実距離1mでの計測値に合わせて校正 [pixel]
    TARGET_DISTANCE = 0.50    # 目標維持距離 [m] (50cm)

    print("ArUcoトラッキングテスト開始 (終了するには画面選択後に 'q' を押してください)")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("エラー: フレームを取得できませんでした。")
            break

        height, width, _ = frame.shape
        img_center_x = width // 2
        img_center_y = height // 2

        # 4. マーカー検出の実行
        corners, ids, rejected = detector.detectMarkers(frame)

        if ids is not None:
            # 検出されたマーカーの枠描画
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)

            for corner in corners:
                pts = corner[0]
                # マーカー中心座標 [pixel] の算出
                marker_center_x = int(np.mean(pts[:, 0]))
                marker_center_y = int(np.mean(pts[:, 1]))

                # 横ズレ誤差 e_x(t) [pixel]
                e_x = marker_center_x - img_center_x

                # マーカー一辺の外形ピクセル幅の算定
                pixel_width = np.linalg.norm(pts[0] - pts[1])

                # 簡易距離 D [m] の推定
                distance = (MARKER_REAL_WIDTH * FOCAL_LENGTH_PIXEL) / pixel_width if pixel_width > 0 else 0.0

                # 距離誤差 e_y(t) [m] (目標距離との差)
                e_y = distance - TARGET_DISTANCE

                # 5. 画面描画処理
                # マーカー中心点の描画と中心線
                cv2.circle(frame, (marker_center_x, marker_center_y), 5, (0, 0, 255), -1)
                cv2.line(frame, (img_center_x, img_center_y), (marker_center_x, marker_center_y), (255, 0, 0), 2)

                # 算出した誤差データのテキストオーバーレイ
                info_ex = f"e_x: {e_x} px"
                info_dist = f"Dist: {distance:.2f} m (e_y: {e_y:.2f} m)"
                cv2.putText(frame, info_ex, (marker_center_x + 10, marker_center_y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.putText(frame, info_dist, (marker_center_x + 10, marker_center_y + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 画面中心のガイド十字描画
        cv2.line(frame, (img_center_x - 15, img_center_y), (img_center_x + 15, img_center_y), (0, 255, 255), 1)
        cv2.line(frame, (img_center_x, img_center_y - 15), (img_center_x, img_center_y + 15), (0, 255, 255), 1)

        cv2.imshow("ArUco Tracker Test", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()