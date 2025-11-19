import numpy as np
import pandas as pd 
from PIL import Image
from tqdm import tqdm
import os
import cv2
from mtcnn import MTCNN

output_size = (224, 224)
detector = MTCNN()


def mtcnn_align(img):  # img: BGR ndarray
    faces = detector.detect_faces(img)
    if len(faces) == 0:
        return None
    face = max(faces, key=lambda f: f['confidence'])
    x, y, w, h = face['box']
    keypoints = face['keypoints']
    left_eye, right_eye = keypoints['left_eye'], keypoints['right_eye']
    dx, dy = right_eye[0] - left_eye[0], right_eye[1] - left_eye[1]
    angle = np.degrees(np.arctan2(dy, dx))
    eye_center = ((left_eye[0] + right_eye[0]) / 2, (left_eye[1] + right_eye[1]) / 2)
    M = cv2.getRotationMatrix2D(eye_center, angle, 1)
    aligned = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
    face_img = aligned[y:y+h, x:x+w]
    if face_img.size == 0:
        return None
    face_img = cv2.resize(face_img, output_size)
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    return gray


# making folders
outer_names = ['test','train']
inner_names = ['angry', 'disgusted', 'fearful', 'happy', 'sad', 'surprised', 'neutral']
os.makedirs('data', exist_ok=True)
for outer_name in outer_names:
    os.makedirs(os.path.join('data',outer_name), exist_ok=True)
    for inner_name in inner_names:
        os.makedirs(os.path.join('data',outer_name,inner_name), exist_ok=True)


# to keep count of each category
angry = 0
disgusted = 0
fearful = 0
happy = 0
sad = 0
surprised = 0
neutral = 0
angry_test = 0
disgusted_test = 0
fearful_test = 0
happy_test = 0
sad_test = 0
surprised_test = 0
neutral_test = 0


df = pd.read_csv('./fer2013.csv')
mat = np.zeros((48,48),dtype=np.uint8)
print("Saving images...")


# read the csv file line by line
for i in tqdm(range(len(df))):
    pixels = np.array(df['pixels'][i].split(), dtype=np.uint8).reshape((48, 48))
    img = cv2.cvtColor(pixels, cv2.COLOR_GRAY2BGR)
    aligned_gray = mtcnn_align(img)
    if aligned_gray is None:
        continue


    # train
    if i < 28709:
        base_dir = 'data'
        if df['emotion'][i] == 0:
            Image.fromarray(aligned_gray).save(os.path.join(base_dir, 'train', 'angry', f"im{angry}.png"))
            angry += 1
        elif df['emotion'][i] == 1:
            Image.fromarray(aligned_gray).save(os.path.join(base_dir, 'train', 'disgusted', f"im{disgusted}.png"))
            disgusted += 1
        elif df['emotion'][i] == 2:
            Image.fromarray(aligned_gray).save(os.path.join(base_dir, 'train', 'fearful', f"im{fearful}.png"))
            fearful += 1
        elif df['emotion'][i] == 3:
            Image.fromarray(aligned_gray).save(os.path.join(base_dir, 'train', 'happy', f"im{happy}.png"))
            happy += 1
        elif df['emotion'][i] == 4:
            Image.fromarray(aligned_gray).save(os.path.join(base_dir, 'train', 'sad', f"im{sad}.png"))
            sad += 1
        elif df['emotion'][i] == 5:
            Image.fromarray(aligned_gray).save(os.path.join(base_dir, 'train', 'surprised', f"im{surprised}.png"))
            surprised += 1
        elif df['emotion'][i] == 6:
            Image.fromarray(aligned_gray).save(os.path.join(base_dir, 'train', 'neutral', f"im{neutral}.png"))
            neutral += 1


    # test
    else:
        base_dir = 'data'
        if df['emotion'][i] == 0:
            Image.fromarray(aligned_gray).save(os.path.join(base_dir, 'test', 'angry', f"im{angry_test}.png"))
            angry_test += 1
        elif df['emotion'][i] == 1:
            Image.fromarray(aligned_gray).save(os.path.join(base_dir, 'test', 'disgusted', f"im{disgusted_test}.png"))
            disgusted_test += 1
        elif df['emotion'][i] == 2:
            Image.fromarray(aligned_gray).save(os.path.join(base_dir, 'test', 'fearful', f"im{fearful_test}.png"))
            fearful_test += 1
        elif df['emotion'][i] == 3:
            Image.fromarray(aligned_gray).save(os.path.join(base_dir, 'test', 'happy', f"im{happy_test}.png"))
            happy_test += 1
        elif df['emotion'][i] == 4:
            Image.fromarray(aligned_gray).save(os.path.join(base_dir, 'test', 'sad', f"im{sad_test}.png"))
            sad_test += 1
        elif df['emotion'][i] == 5:
            Image.fromarray(aligned_gray).save(os.path.join(base_dir, 'test', 'surprised', f"im{surprised_test}.png"))
            surprised_test += 1
        elif df['emotion'][i] == 6:
            Image.fromarray(aligned_gray).save(os.path.join(base_dir, 'test', 'neutral', f"im{neutral_test}.png"))
            neutral_test += 1


print("Done!")