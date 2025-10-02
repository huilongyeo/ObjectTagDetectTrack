import cv2
import numpy as np
import os
import random
import glob

img_dir = 'input' 
label_dir = 'input'
out_img_dir = 'train'
out_label_dir = 'train'
os.makedirs(out_img_dir, exist_ok=True)

# 角度變換設定
angles = [-15, -10, -5, 0, 5, 10, 15]

def load_yolo_label(label_path, img_w, img_h):
    labels = []
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts: continue
            cls, xc, yc, w, h = map(float, parts)
            x1 = int((xc - w/2) * img_w)
            y1 = int((yc - h/2) * img_h)
            x2 = int((xc + w/2) * img_w)
            y2 = int((yc + h/2) * img_h)
            labels.append([cls, x1, y1, x2, y2])
    return labels

def save_yolo_label(label_path, labels, img_w, img_h):
    with open(label_path, 'w') as f:
        for cls, x1, y1, x2, y2 in labels:
            xc = ((x1 + x2) / 2) / img_w
            yc = ((y1 + y2) / 2) / img_h
            w = (x2 - x1) / img_w
            h = (y2 - y1) / img_h
            f.write(f"{int(cls)} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")

def rotate_img_and_boxes(img, labels, angle):
    (h, w) = img.shape[:2]
    center = (w//2, h//2)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_img = cv2.warpAffine(img, rot_mat, (w, h), borderValue=(127,127,127))
    new_labels = []
    for cls, x1, y1, x2, y2 in labels:
        pts = np.array([[x1, y1], [x2, y2]])
        ones = np.ones(shape=(len(pts), 1))
        pts_ones = np.hstack([pts, ones])
        rotated_pts = rot_mat.dot(pts_ones.T).T
        x1r, y1r = rotated_pts[0]
        x2r, y2r = rotated_pts[1]
        x1n, x2n = min(x1r, x2r), max(x1r, x2r)
        y1n, y2n = min(y1r, y2r), max(y1r, y2r)
        new_labels.append([cls, x1n, y1n, x2n, y2n])
    return rotated_img, new_labels

def flip_img_and_boxes(img, labels, flipCode):
    h, w = img.shape[:2]
    flipped_img = cv2.flip(img, flipCode)
    new_labels = []
    for cls, x1, y1, x2, y2 in labels:
        if flipCode == 1: 
            x1f, x2f = w - x2, w - x1
            new_labels.append([cls, x1f, y1, x2f, y2])
        elif flipCode == 0: 
            y1f, y2f = h - y2, h - y1
            new_labels.append([cls, x1, y1f, x2, y2f])
        elif flipCode == -1: 
            x1f, x2f = w - x2, w - x1
            y1f, y2f = h - y2, h - y1
            new_labels.append([cls, x1f, y1f, x2f, y2f])
    return flipped_img, new_labels

def trapezoid_transform(img, labels, horiz=True, max_ratio=0.25):
    h, w = img.shape[:2]
    offset = int((w if horiz else h) * max_ratio)
    if horiz:
        pts1 = np.float32([[0,0],[w,0],[w,h],[0,h]])
        pts2 = np.float32([[random.randint(0,offset),0],
                           [w-random.randint(0,offset),0],
                           [w-random.randint(0,offset),h],
                           [random.randint(0,offset),h]])
    else:
        pts1 = np.float32([[0,0],[w,0],[w,h],[0,h]])
        pts2 = np.float32([[0,random.randint(0,offset)],
                           [w,random.randint(0,offset)],
                           [w,h-random.randint(0,offset)],
                           [0,h-random.randint(0,offset)]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    img_warp = cv2.warpPerspective(img, M, (w,h), borderValue=(127,127,127))
    new_labels = []
    for cls, x1, y1, x2, y2 in labels:
        box_pts = np.array([[x1, y1], [x2, y2]], dtype=np.float32)
        box_pts = cv2.perspectiveTransform(box_pts[np.newaxis, :, :], M)[0]
        x1n, y1n = box_pts[0]
        x2n, y2n = box_pts[1]
        x1n, x2n = min(x1n, x2n), max(x1n, x2n)
        y1n, y2n = min(y1n, y2n), max(y1n, y2n)
        new_labels.append([cls, x1n, y1n, x2n, y2n])
    return img_warp, new_labels

def color_jitter(img):
    alpha = random.uniform(0.7, 1.3)
    beta = random.randint(-30, 30)
    img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    s = hsv[:,:,1].astype(np.float32)
    sat_factor = random.uniform(0.7, 1.3)
    s = np.clip(s * sat_factor, 0, 255)
    hsv[:,:,1] = s.astype(np.uint8)
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return img

img_files = glob.glob(os.path.join(img_dir, '*.jpg'))
label_dir = 'input'
out_label_dir = 'train'
os.makedirs(out_label_dir, exist_ok=True)

augmentations = [
    ('rotate', angles),
    ('flip', [1, 0, -1]),
    ('trapezoid', ['horiz', 'vert'])
]

for img_path in img_files:
    img_name = os.path.splitext(os.path.basename(img_path))[0]
    label_path = os.path.join(label_dir, f"{img_name}.txt")
    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    labels = load_yolo_label(label_path, w, h)

    for angle in angles:
        rot_img, rot_labels = rotate_img_and_boxes(img, labels, angle)
        for code, desc in zip([1, 0, -1], ['hflip', 'vflip', 'hvflip']):
            flip_img, flip_labels = flip_img_and_boxes(rot_img, rot_labels, code)
            for horiz, tdesc in zip([True, False], ['trap_h', 'trap_v']):
                trap_img, trap_labels = trapezoid_transform(flip_img, flip_labels, horiz=horiz)
                for jitter in [False, True]:
                    if jitter:
                        out_img = os.path.join(out_img_dir, f"{img_name}_rot{angle}_{desc}_{tdesc}_jitter.jpg")
                        out_label = os.path.join(out_label_dir, f"{img_name}_rot{angle}_{desc}_{tdesc}_jitter.txt")
                        out_img_data = color_jitter(trap_img)
                    else:
                        out_img = os.path.join(out_img_dir, f"{img_name}_rot{angle}_{desc}_{tdesc}.jpg")
                        out_label = os.path.join(out_label_dir, f"{img_name}_rot{angle}_{desc}_{tdesc}.txt")
                        out_img_data = trap_img
                    save_yolo_label(out_label, trap_labels, w, h)
                    cv2.imwrite(out_img, out_img_data)

print("Completed")