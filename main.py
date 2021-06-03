import pandas as pd
import os
import keras.preprocessing.image
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image
from sklearn.model_selection import train_test_split




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
    generator.fit(data)
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




if __name__ == '__main__':
    df=get_df()
    df=get_image_data(df)
    print(df.head())
    X_train,X_valid, X_test, y_train, y_valid,y_test,generator=split_data(df,0.5,0.5)
    print(X_train.shape,y_train.shape)
    print(X_valid.shape,y_valid.shape)
    print(X_test.shape,y_test.shape)

