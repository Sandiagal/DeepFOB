# -*- coding: utf-8 -*-
# Author: Sandiagal <sandiagal2525@gmail.com>,
# License: GPL-3.0

# USAGE
# python3 train.py

# import the necessary packages
import os
from pickle import dump
from time import localtime
from time import strftime

#from imblearn.over_sampling import RandomOverSampler
from imgaug import augmenters as ia
from keras import backend as K
import keras.backend.tensorflow_backend as KTF
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.callbacks import ReduceLROnPlateau
from keras.models import Model
import keras.optimizers as op
from keras.preprocessing.image import ImageDataGenerator
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
#from sklearn.utils import shuffle
import tensorflow as tf

import idiplab_cv.dataset_io as io
from idiplab_cv import models
from idiplab_cv.preprocess import Imgaug

# %% 自适应显存

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
KTF.set_session(session)

# %% 记录系统信息

nowTime = strftime("%Y%m%d", localtime())
np.random.seed(seed=0)
tf.set_random_seed(seed=0)

# %% 全局变量

DATASET_PATH = "训练集"
INPUT_SHAPE = (224, 224, 3)
BATCH_SIZE = 32
EPOCHS = [100, 20, 25, 20]
#EPOCHS = [1, 1, 1, 1]s
DROPOUT = 0.5
INIT_LR = [1e-4, 1e-4]
AUGMENT_AMOUNT = 10
FEATURE_LAYER = "conv_pw_13_relu"
WEIGHT_LAYER = "predictions"

# %% 读入数据

dataset = io.Dataset(augment)
class_to_index, sample_per_class = dataset.load_data(
    path=DATASET_PATH,
    shape=INPUT_SHAPE[:2])
imgs_train, labels_train, imgs_valid, labels_valid = dataset.train_test_split(
    test_shape=0.1)
labels_train = np.array(dataset.labels_origin)
imgs_train = np.array(dataset.imgs_origin)

labels_train_splite = []
for labels in labels_train:
    label = labels.split("_")
    labels_train_splite.append(label)

labels_valid_splite = []
for labels in labels_valid:
    label = labels.split("_")
    labels_valid_splite.append(label)

mlb = MultiLabelBinarizer()
labels_train = mlb.fit_transform(labels_train_splite)
labels_valid = mlb.fit_transform(labels_valid_splite)
labels_train = io.label_smooth(labels_train, [0, 1, 4, 5])

for (i, label) in enumerate(mlb.classes_):
    print("{}. {}".format(i + 1, label))

# %% 数据预处理

imgs_train = np.array(imgs_train, dtype="float32")
imgs_valid = np.array(imgs_valid, dtype="float32")

normalization_datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True
)

imgs_train_origin = imgs_train
normalization_datagen.fit(imgs_train_origin)

train_generator = normalization_datagen.flow(
    imgs_train, labels_train,
    batch_size=BATCH_SIZE,
    shuffle=True)

valid_generator = normalization_datagen.flow(
    imgs_valid, labels_valid,
    batch_size=BATCH_SIZE,
    shuffle=False)

mean = normalization_datagen.mean,
std = normalization_datagen.std

# %% 提取数据特征层

print("[INFO] Loading mobilenetGAP...")
feature_model = models.mobilenetGAP(
    input_shape=INPUT_SHAPE,
    classes=len(mlb.classes_),
    include_top=False)
# feature_model.summary()

features_train = feature_model.predict_generator(
    generator=train_generator,
    steps=len(labels_train) / BATCH_SIZE,
    verbose=1)
features_valid = feature_model.predict_generator(
    generator=valid_generator,
    steps=len(labels_valid) / BATCH_SIZE,
    verbose=1)

# %% 训练反馈

checkpoint = ModelCheckpoint(
    filepath="record_epoch.{epoch:02d}_loss.{val_loss:.2f}_acc.{val_acc:.2f}.h5",
    monitor='val_acc',
    verbose=1,
    save_best_only=True)

reduceLR = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=10,
    verbose=1,
    epsilon=0,
    cooldown=0,
    min_lr=0)

earlyStop = EarlyStopping(
    monitor='val_loss',
    min_delta=0,
    patience=21,
    verbose=1)

callbacks_list = [checkpoint, reduceLR]

# %% 1阶段训练

top_model = models.top(
    input_shape=features_train.shape[1:],
    classes=len(mlb.classes_),
    dropout=DROPOUT,
    finalAct="sigmoid")
top_model.summary()

top_model.compile(
    optimizer=op.adam(lr=INIT_LR[0], decay=INIT_LR[0]/EPOCHS[0]),
    loss='binary_crossentropy',
    metrics=['accuracy'])

print("[INFO] stage 1 training...")
History = top_model.fit(
    x=features_train,
    y=labels_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS[0],
    callbacks=callbacks_list,
    validation_data=(features_valid, labels_valid))

EPOCHS[0] = len(History.history["loss"])-1

# %% 2阶段训练

print("[INFO] Loading mobilenetGAP...")
model = models.mobilenetGAP(
    input_shape=INPUT_SHAPE,
    classes=len(mlb.classes_),
    dropout=DROPOUT,
    finalAct="sigmoid")
for new_layer, layer in zip(model.layers[-2:], top_model.layers[-2:]):
    new_layer.set_weights(layer.get_weights())
model.summary()

model.compile(
    optimizer=op.adam(lr=INIT_LR[1], decay=INIT_LR[1]/EPOCHS[1]),
    loss='binary_crossentropy',
    metrics=['accuracy'])

reduceLR = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.9,
    patience=10,
    verbose=1,
    epsilon=0,
    cooldown=0,
    min_lr=0)

callbacks_list = [checkpoint, reduceLR]

print("[INFO] stage 2 training...")
History2 = model.fit_generator(
    generator=train_generator,
    steps_per_epoch=len(labels_train) / BATCH_SIZE,
    epochs=EPOCHS[1],
    validation_data=valid_generator,
    validation_steps=len(labels_valid) / BATCH_SIZE,
    callbacks=callbacks_list,
    shuffle=True)

for key in History2.history.keys():
    for value in History2.history[key]:
        History.history[key].append(value)

EPOCHS[1] = len(History.history["loss"])-1

# %% 数据增强方法

seq = ia.Sometimes(
    AUGMENT_AMOUNT/(AUGMENT_AMOUNT+1),
    ia.Sequential([

        ia.CropAndPad(
            percent=(0, 0.1),
            pad_mode=["constant", "edge"],
            pad_cval=(0)
        ),

        ia.OneOf([
            ia.AdditiveGaussianNoise(scale=(0.0, 0.1*255)),
            ia.CoarseDropout(0.1, size_percent=0.2)
        ]),

        ia.OneOf([
            ia.Add((-20, 0)),
            ia.Multiply((1, 1.2))
        ]),

        ia.Affine(
            scale={"x": (0.95, 1.05), "y": (1.025, 1.025)},
            translate_percent={"x": (-0.2, 0.2), "y": (-0.1, 0.1)},
            rotate=(-1, 1),
            shear=(-1, 1),
            mode=["constant", "edge"],
            cval=(0)
        )

    ], random_order=True)  # apply augmenters in random order
)

preprocessor = Imgaug(seq)

augment_datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    preprocessing_function=preprocessor)

augment_datagen.fit(imgs_train_origin)
train_augment_generator = augment_datagen.flow(
    imgs_train, labels_train,
    batch_size=BATCH_SIZE,
    shuffle=True)

valid_augment_generator = augment_datagen.flow(
    imgs_valid, labels_valid,
    batch_size=BATCH_SIZE,
    shuffle=False)

# %% 3阶段训练

print("[INFO] stage 3 training...")

model.compile(
    optimizer=op.adam(lr=INIT_LR[1], decay=INIT_LR[1]/EPOCHS[2]),
    loss='binary_crossentropy',
    metrics=['accuracy'])

History2 = model.fit_generator(
    generator=train_augment_generator,
    steps_per_epoch=(AUGMENT_AMOUNT+1)*len(labels_train) / BATCH_SIZE,
    epochs=EPOCHS[2],
    validation_data=valid_augment_generator,
    validation_steps=(AUGMENT_AMOUNT+1)*len(labels_valid) / BATCH_SIZE,
    callbacks=callbacks_list,
    shuffle=True)

for key in History2.history.keys():
    for value in History2.history[key]:
        History.history[key].append(value)

EPOCHS[2] = len(History.history["loss"])-1

# %% 4阶段训练

print("[INFO] stage 4 compile...")
model.compile(
    optimizer=op.adam(lr=INIT_LR[1], decay=INIT_LR[1]/EPOCHS[3]),
    loss='binary_crossentropy',
    metrics=['accuracy'])

print("[INFO] stage 4 training...")
History2 = model.fit_generator(
    generator=train_generator,
    steps_per_epoch=len(labels_train) / BATCH_SIZE,
    epochs=EPOCHS[3],
    validation_data=valid_generator,
    validation_steps=len(labels_valid) / BATCH_SIZE,
    callbacks=callbacks_list,
    shuffle=True)

for key in History2.history.keys():
    for value in History2.history[key]:
        History.history[key].append(value)

EPOCHS[3] = len(History.history["loss"])-1

# %% 记录文件

print("[INFO] Tail end...")
model.save(nowTime+"_model.h5")

# %% 记录特征层与最后输出

getFeatureMaps = Model(
    inputs=model.input,
    outputs=model.get_layer(FEATURE_LAYER).output)
feature_maps = getFeatureMaps.predict_generator(
    generator=valid_generator,
    steps=len(labels_valid) / BATCH_SIZE,
    verbose=1)

weights = model.get_layer(WEIGHT_LAYER).get_weights()[0]

getScoresPredict = K.function([model.get_layer(index=model.layers.index(
    model.get_layer(FEATURE_LAYER))+1).input], [model.output])
[scores_predict] = getScoresPredict([feature_maps])

result = {
    "epochs": EPOCHS,
    "history": History.history,
    "labels_valid": np.argmax(labels_valid, axis=1),
    "mean": mean,
    #    "names_valid": names_valid,
    "scores_predict": scores_predict,
    "std": std, }
f = open(nowTime+"_result.h5", "wb")
dump(result, f, True)
f.close()

result = {
    "feature_maps": feature_maps,
    "weights": weights}
f = open(nowTime+"_feature_maps.h5", "wb")
dump(result, f, True)
f.close()

# %% 打印训练过程

plt.style.use("ggplot")
plt.figure()
plt.plot(History.history["loss"], "o-",
         label="train_loss (%.4f)" % (History.history["loss"][-1]))
plt.plot(History.history["val_loss"], "o-",
         label="val_loss (%.4f)" % (History.history["val_loss"][-1]))
plt.plot(History.history["acc"], "o-",
         label="train_acc (%.2f%%)" % (History.history["acc"][-1]*100))
plt.plot(History.history["val_acc"], "o-",
         label="val_acc (%.2f%%)" % (History.history["val_acc"][-1]*100))
plt.vlines(EPOCHS, 0, History.history["loss"][0], colors="c",
           linestyles="dashed", label="steps gap")

plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="best", shadow=1)
plt.savefig("Training Loss and Accuracy.png")
