import cv2

# 4x4_50 辞書の取得
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
# ID: 0 のマーカーを 300x300 ピクセルで生成
marker_img = cv2.aruco.generateImageMarker(dictionary, 0, 300)
# 画像の保存
cv2.imwrite("marker_4x4_50_id0.png", marker_img)
print("マーカー画像を保存しました: marker_4x4_50_id0.png")