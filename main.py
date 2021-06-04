import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from keras import Sequential
from keras.layers import BatchNormalization, GlobalAveragePooling2D, Dense, Dropout
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image
from sklearn.model_selection import train_test_split
import tensorflow as tf




PATH="./dataset"
IMG_SIZE = 250
NUM_OF_LABEL=102


def get_df():
    train=[]
    for folder in os.listdir(PATH):
        if not folder=="test":
          for label in os.listdir(PATH+"/"+folder):
              for img in  os.listdir(PATH+"/"+folder+"/"+label):
                  path=PATH+"/"+folder+"/"+label+"/"+img
                  train.append([path,int(label)-1])
    df_train=pd.DataFrame(train,columns=["id","label"])

    return df_train

def get_image_data(df) :
    arr=[]
    for record in df.iterrows():
        img_path=record[1]["id"]
        img = Image.open( img_path )
        img.load()
        img = img.resize((IMG_SIZE, IMG_SIZE), Image.ANTIALIAS)
        data = np.asarray( img, dtype="int32" )
        arr.append([data,record[1]["label"]])
    return pd.DataFrame(arr,columns=["features","label"])

def convert_label_to_vector(y, num_of_label):
    vectors=[]
    y = np.array(y)
    y = y.reshape(-1, 1)
    labels = np.array(y).reshape(-1)
    vector_labels=np.eye(num_of_label)
    for l in labels:
        vectors.append(vector_labels[int(l)])
    return vectors


def data_generator(data):
    generator = ImageDataGenerator(
        rotation_range=50,
        shear_range=0.2,
        zoom_range=[0.75, 1.25],
        brightness_range=[0.5, 1.5],
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True)
    generator.fit(np.array([x for x in data]))
    return generator



def split_data(data,test_size,valid_size):
    X_train, X_test, y_train, y_test=train_test_split(data["features"],data["label"],test_size=test_size,stratify=df['label'])
    X_train, X_valid, y_train, y_valid = train_test_split(X_train,y_train, test_size=valid_size)
    y_train=convert_label_to_vector(y_train,NUM_OF_LABEL)
    y_valid=convert_label_to_vector(y_valid,NUM_OF_LABEL)
    y_test=convert_label_to_vector(y_test,NUM_OF_LABEL)
    X_train=np.array(X_train)
    y_train=np.array(y_train)
    X_valid=np.array(X_valid)
    y_valid=np.array(y_valid)
    X_test=np.array(X_test)
    y_test=np.array(y_test)
    generator=data_generator(X_train)

    return  X_train,X_valid, X_test, y_train, y_valid,y_test,generator

def get_model(name):
    model=Sequential()
    if name == "vgg16":
        app=tf.keras.applications.VGG16(weights='imagenet', include_top=False, input_shape=(IMG_SIZE,IMG_SIZE,3))
    elif name== "densenet121":
        app=tf.keras.applications.DenseNet121(weights='imagenet', include_top=False, input_shape=(IMG_SIZE,IMG_SIZE,3))
    for layer in app.layers:
        if isinstance(layer, BatchNormalization):
            layer.trainable = True
        else:
            layer.trainable = False
    model.add(app)
    return model

def add_layers_to_model(model):
    model.add(GlobalAveragePooling2D())
    model.add(Dense(500, activation='selu'))
    model.add(Dropout(.4))
    model.add(BatchNormalization())
    model.add(Dense(256, activation='selu'))
    model.add(Dropout(.25))
    model.add(BatchNormalization())
    model.add(Dense(NUM_OF_LABEL, activation='softmax'))
    model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=["accuracy"])


def run_model(model,X_train,X_valid, X_test, y_train, y_valid,y_test,generator,batch=200):
    es = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)
    cb_list = [es]
    fit_model_log = model.fit(generator.flow(
        np.array([x for x in X_train]), y_train,
        batch_size=batch),
        validation_data=(np.array([x for x in X_valid]), y_valid),
        epochs=50,
        shuffle=True,
        callbacks=[cb_list]
    )
    test_score = model.evaluate(np.array([x for x in X_test]), y_test)

    plt.plot(fit_model_log.history['loss'])
    plt.plot(fit_model_log.history['val_loss'])
    plt.title('Loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()

    plt.plot(fit_model_log.history['accuracy'])
    plt.plot(fit_model_log.history['val_accuracy'])
    plt.title('Accuracy')
    plt.ylabel('acc')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()

    print(model.metrics_names[0] + ":"+ str(test_score[0]))
    print(model.metrics_names[1] + ":"+ str(test_score[1]))






if __name__ == '__main__':
    df=get_df()
    df=get_image_data(df)
    # print(df.head())
    X_train,X_valid, X_test, y_train, y_valid,y_test,generator=split_data(df,0.5,0.5)
    model=get_model("vgg16")
    add_layers_to_model(model)
    run_model(model,X_train,X_valid, X_test, y_train, y_valid,y_test,generator)


