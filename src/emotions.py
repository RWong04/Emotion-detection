import numpy as np
import argparse
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import os

# 關閉多餘Log
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 參數設置
ap = argparse.ArgumentParser()
ap.add_argument("--mode", help="train/display")
mode = ap.parse_args().mode

train_dir = 'data/train'
val_dir = 'data/test'
batch_size = 64
num_epoch = 50
fine_tune_epoch = 30

# def add_gaussian_noise(img):
#     # gray to [224,224,1] float
#     noise = tf.random.normal(shape=tf.shape(img), mean=0.0, stddev=0.05)
#     img = img + noise
#     return tf.clip_by_value(img, 0., 1.)
def add_augmentation_noise(img):
    # tf Tensor: 資料進來時已[0,1] float
    # 1. 加高斯雜訊
    gauss = tf.random.normal(tf.shape(img), mean=0.0, stddev=0.05)
    noisy_img = img + gauss
    noisy_img = tf.clip_by_value(noisy_img, 0., 1.)

    # 2. 調對比度（隨機90%~120%）
    contrast_factor = tf.random.uniform([], 0.9, 1.2)
    noisy_img = tf.image.adjust_contrast(noisy_img, contrast_factor)

    return tf.clip_by_value(noisy_img, 0., 1.)


# 加強版資料增強（保守設定，適用MTCNN後的224x224臉部照片）
# train_datagen = ImageDataGenerator(
#     rescale=1./255,
#     rotation_range=10,
#     width_shift_range=0.05,
#     height_shift_range=0.05,
#     zoom_range=0.05,
#     horizontal_flip=True,
#     # preprocessing_function=add_gaussian_noise
# )
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=10,
    zoom_range=0.1,
    horizontal_flip=True,
    brightness_range=[0.7, 1.3],
    preprocessing_function=add_augmentation_noise
)


val_datagen = ImageDataGenerator(rescale=1./255)

# 灰階轉三通道的Generator（MobileNetV2需要RGB）
def gray_to_rgb_generator(generator):
    for x, y in generator:
        # [N,224,224,1] -> [N,224,224,3]
        yield np.repeat(x, 3, axis=-1), y

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=batch_size,
    color_mode="grayscale",
    class_mode='categorical'
)

validation_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(224, 224),
    batch_size=batch_size,
    color_mode="grayscale",
    class_mode='categorical'
)

steps_per_epoch = len(train_generator)
validation_steps = len(validation_generator)

# Model 建構
inputs = Input(shape=(224, 224, 3))
base_model = MobileNetV2(include_top=False, weights='imagenet', input_tensor=inputs)
base_model.trainable = False  # 先凍結 backbone

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
outputs = Dense(7, activation='softmax')(x)

model = Model(inputs=inputs, outputs=outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Callbacks
callbacks = [
    EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)
]

if mode == "train":
    model_info = model.fit(
        gray_to_rgb_generator(train_generator),
        steps_per_epoch=len(train_generator),
        epochs=num_epoch,
        validation_data=gray_to_rgb_generator(validation_generator),
        validation_steps=len(validation_generator),
        callbacks=callbacks
    )

    # 描繪訓練過程
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    # Accuracy
    axs[0].plot(model_info.history['accuracy'])
    axs[0].plot(model_info.history['val_accuracy'])
    axs[0].set_title('Model Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Accuracy')
    axs[0].legend(['Train', 'Validation'], loc='best')
    # Loss
    axs[1].plot(model_info.history['loss'])
    axs[1].plot(model_info.history['val_loss'])
    axs[1].set_title('Model Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Loss')
    axs[1].legend(['Train', 'Validation'], loc='best')
    plt.savefig('plot_transfer.png')
    plt.show()
    model.save_weights('mobilenetv2_fer2013.weights.h5')

elif mode == "finetune":
    # 先載入 head 訓練完的最佳權重
    model.load_weights('mobilenetv2_fer2013.weights.h5')
    
    # 階段二：解凍 MobileNetV2 最後幾個 block
    base_model.trainable = True
    for layer in base_model.layers[:-30]:  # 只解凍最後30層
        layer.trainable = False
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    history_finetune = model.fit(
        gray_to_rgb_generator(train_generator),
        steps_per_epoch=steps_per_epoch,
        epochs=fine_tune_epoch,
        validation_data=gray_to_rgb_generator(validation_generator),
        validation_steps=validation_steps,
        callbacks=callbacks
    )
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    axs[0].plot(history_finetune.history['accuracy'])
    axs[0].plot(history_finetune.history['val_accuracy'])
    axs[0].set_title('Model Accuracy (finetune)')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Accuracy')
    axs[0].legend(['Train', 'Validation'], loc='best')
    axs[1].plot(history_finetune.history['loss'])
    axs[1].plot(history_finetune.history['val_loss'])
    axs[1].set_title('Model Loss (finetune)')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Loss')
    axs[1].legend(['Train', 'Validation'], loc='best')
    plt.savefig('plot_transfer_finetune.png')
    plt.show()
    model.save_weights('mobilenetv2_fer2013_finetuned.weights.h5')

elif mode == "display":
    model.load_weights('mobilenetv2_fer2013_finetuned.weights.h5')
    cv2.ocl.setUseOpenCL(False)
    emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        facecasc = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = facecasc.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
            roi_gray = gray[y:y + h, x:x + w]
            roi_gray_resized = cv2.resize(roi_gray, (224, 224))
            roi_input = roi_gray_resized.astype('float32') / 255.0
            roi_input = np.expand_dims(np.expand_dims(roi_input, -1), 0)
            roi_input_rgb = np.repeat(roi_input, 3, axis=-1)  # (1,224,224,3)
            prediction = model.predict(roi_input_rgb)
            maxindex = int(np.argmax(prediction))
            cv2.putText(frame, emotion_dict[maxindex], (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
        cv2.imshow('Video', cv2.resize(frame,(1600,960),interpolation=cv2.INTER_CUBIC))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

print("模型構建與訓練完成！")
