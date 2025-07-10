# line_detection
![ex_screenshot](./img/screenshot.png)
python library
-----
```
pip install numpy
pip install opencv-python
``` 

source code
---

```
import cv2
import numpy as np

# 1. 이미지 불러오기
img = cv2.imread('/Users/wonseo/Documents/vscode_project/math/ex/road.webp')

# 2. HLS 색상공간 변환
hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)

# 3. 흰색, 노란색 차선 마스킹
# 흰색 범위 (밝기가 높고 채도가 낮음)
lower_white = np.array([0, 200, 0])
upper_white = np.array([180, 255, 255]) 
white_mask = cv2.inRange(hls, lower_white, upper_white)

# 노란색 범위 (색상 15~35, 밝기·채도 적당)
lower_yellow = np.array([15, 30, 115])
upper_yellow = np.array([35, 204, 255])
yellow_mask = cv2.inRange(hls, lower_yellow, upper_yellow)

# 두 마스크 합치기
mask = cv2.bitwise_or(white_mask, yellow_mask)
masked = cv2.bitwise_and(img, img, mask=mask)

# 4. 블러 처리(노이즈 제거)
blur = cv2.GaussianBlur(masked, (5, 5), 0)

# 5. 엣지 검출
edges = cv2.Canny(blur, 50, 150)

# 6. 허프 변환으로 직선(차선) 검출
lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=200, minLineLength=50, maxLineGap=60)

# 7. 좌우 차선 분류 및 대표 차선 추출
left_lines = []
right_lines = []
img_center = img.shape[1] // 2

if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 - x1 == 0:
            continue
        slope = (y2 - y1) / (x2 - x1)
        if abs(slope) < 0.3:  # 너무 수평한 선 제외
            continue
        if slope < 0 and x1 < img_center and x2 < img_center:
            left_lines.append(line[0])
        elif slope > 0 and x1 > img_center and x2 > img_center:
            right_lines.append(line[0])

def select_main_line(lines, img_center):
    if not lines:
        return None
    # 중앙에 가까운 x좌표를 기준으로 정렬
    lines = sorted(lines, key=lambda l: min(abs(l[0]-img_center), abs(l[2]-img_center)))
    return lines[0]

left_line = select_main_line(left_lines, img_center)
right_line = select_main_line(right_lines, img_center)

# 8. 결과 그리기

roi_result = img.copy()
edge_result = edges.copy()
if left_line is not None and right_line is not None:
    # 네 꼭짓점 좌표(시계방향)
    line_points = [
        [left_line[2], left_line[3]],
        [left_line[0], left_line[1]],
        [right_line[2], right_line[3]],
        [right_line[0], right_line[1]]
    ]
    # 직선 그리기
    cv2.line(roi_result, (left_line[0], left_line[1]), (left_line[2], left_line[3]), (200, 0, 0), 3)
    cv2.line(roi_result, (right_line[0], right_line[1]), (right_line[2], right_line[3]), (200, 0, 0), 3)
    
    # 반투명 폴리곤
    # 반투명 폴리곤(원본)
    pts = np.array(line_points, np.int32)
    overlay = roi_result.copy()
    cv2.fillPoly(overlay, [pts], (0, 255, 0))
    alpha = 0.4
    cv2.addWeighted(overlay, alpha, roi_result, 1 - alpha, 0, roi_result)

    # 반투명 폴리곤(masked)
    overlay_masked = edge_result.copy()
    cv2.fillPoly(overlay_masked, [pts], (0, 255, 0))
    cv2.addWeighted(overlay_masked, alpha, edge_result, 1 - alpha, 0, edge_result)

cv2.imshow('Lane Area', roi_result)
cv2.imshow('Lane Area (masked)', edge_result)
#cv2.imwrite('lane_result.jpg', roi_result)
#cv2.imwrite('ree.jpg', edge_result)
cv2.waitKey(0)
cv2.destroyAllWindows()

```
