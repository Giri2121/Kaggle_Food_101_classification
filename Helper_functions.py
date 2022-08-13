import zipfile
import glob
import os
import shutil
import itertools
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score,precision_recall_fscore_support


def make_confusion_matrix(y_true,y_pred,classes = None,figsize = (10,10),text_size = 15,norm = False,savefig = False):
    """
    :param y_true: Array of truth labels [must be same shape as y_pred]
    :param y_pred: Array of predicted labels
    :param classes: Array of class labels; if None integer labels are used
    :param figsize: size of the output figure
    :param text_size: size of text size in output figure
    :param norm: normalize values or not (default is False)
    :param savefig: save confusion matrix to file (default = False)
    :return: A labelled confusion matrix to the file (default = False)
    """

    #create a confusion matrix
    cm = confusion_matrix(y_true,y_pred)
    cm_norm = cm.astype('float')/cm.sum(axis=1)[:,np.newaxis]
    n_classes = cm.shape[0]

    #plotting the figure
    fig,ax = plt.subplots(figsize = figsize)
    cax = ax.matshow(cm,cmap = 'Blue') #represents how correct a class is; darker is better
    fig.colorbar(cax)

    if classes:
        labels = classes
    else:
        labels = np.arange(cm.shape[0])

    #label the axes
    ax.set(title = 'Confusion Matrix',
           xlabel = 'Predicted_label',
           ylabel = 'True label',
           xticks = np.arange(n_classes),#creates enough axis slots for each class
           yticks = np.arange(n_classes),
           xticklabels = labels, #axis will be labelled with class names (if they exist) or intigers
           yticklables = labels
           )

    #make x-axis labels appear on bottom
    ax.xaxis.set_label_position('bottom')
    ax.xaxis.tick_bottom()

    #setting threshold for different colors
    threshold = (cm.max() + cm.min())/2.

    #plotting the text on each cell
    for i,j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if norm:
            plt.text(j, i, f"{cm[i, j]} ({cm_norm[i, j]*100:.1f}%)",
                     horizontalalignment = 'center',
                     color = 'white' if cm[i, j] > threshold else "black",
                     size = text_size)
        else:
            plt.text(j, i, f"{cm[i, j]}",
                     horizontalalignment="center",
                     color="white" if cm[i, j] > threshold else "black",
                     size=text_size)

    if savefig:
        fig.savefig('ConfusionMatrix.png')


def split_data(path_to_data,path_to_train,path_to_test,split_size=0.20):
    folders = os.listdir(path_to_data)

    for folder in folders:
        if folder == '.DS_Store':
            continue
        full_path = os.path.join(path_to_data,folder)
        images_path = glob.glob(os.path.join(full_path, '*.jpg'))
        x_train,x_test = train_test_split(images_path,test_size=split_size)

        for x in x_train:
            path_to_folder = os.path.join(path_to_train,folder)

            if not os.path.isdir(path_to_folder):
                os.makedirs(path_to_folder)

            shutil.copy(x,path_to_folder)

        for x in x_test:
            path_to_folder = os.path.join(path_to_test,folder)

            if not os.path.isdir(path_to_folder):
                os.makedirs(path_to_folder)

            shutil.copy(x,path_to_folder)

def create_generators(batch_size,path_to_train,path_to_test):
    train_preprocessor = ImageDataGenerator(
        rescale=1/255.,
        rotation_range = 1,
        zoom_range=0.2

    )

    test_preprocessor = ImageDataGenerator(
        rescale=1/255.
    )

    train_generator = train_preprocessor.flow_from_directory(
        path_to_train,
        class_mode = 'categorical',
        target_size = (512,512),
        color_mode = 'rgb',
        shuffle=True,
        batch_size = batch_size
    )

    test_generator = test_preprocessor.flow_from_directory(
        path_to_test,
        class_mode = 'categorical',
        target_size=(512,512),
        color_mode = 'rgb',
        shuffle = True,
        batch_size=batch_size
    )

    return train_generator,test_generator

def performance_plots(history):
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']

    epochs = range(len(history.history['loss'])) #can be any of the parameter

    #loss plots
    plt.plot(epochs,loss,label = 'training_loss')
    plt.plot(epochs,val_loss,label = 'validation_loss')
    plt.title('LOSS')
    plt.xlabel('Epochs')
    plt.legend()

    #accuracy plots
    plt.plot(epochs,accuracy,label = 'training_accuracy')
    plt.plot(epochs,val_accuracy,label = 'validation_accuracy')
    plt.title('ACCURACY')
    plt.xlabel('Epochs')
    plt.legend()


def load_and_prep_image(filename,img_shape = 60,scale = True):
    """
    Reads an image from filename and turns it into a tensor of the shape (227,227,3)
    :param filename: filename of the target image (str)
    :param img_shape: size to resize the target image
    :param scale: rescales the pixel values to 0 and 1; default is True
    :return: processed image
    """
    #reading the image
    img = tf.io.read_file(filename)
    #decoding the image to a tensor
    img = tf.image.decode_jpeg(img)
    #resizing the decoded image
    img = tf.image.resize(img,[img_shape,img_shape])
    if scale:
        return (img/255.)
    else:
        return img



def pred_and_plot(model,filename,class_names):
    """
    Imports a image located at the filename,makes a prediction with the trained
    model and then plots the image with the predicted class as title
    :param model: Trained or the saved model we are working with
    :param filename: path where the test images is located
    :param class_names: respective class names we are working on
    :return: predicted label and prediction probability
    """

    #importing the target images and then preprocessing it
    img = load_and_prep_image(filename)

    #making the prediction
    pred = model.predict(tf.expand_dims(img,axis=0)) #to make it a tensor of size 4

    #getting the prediction class
    if len(pred[0]) > 1:
        pred_class = class_names[pred.argmax()] #incase of multiclass classification
        pred_prob = pred.argmax()
    else:
        pred_class = class_names[int(tf.round(pred)[0][0])]

    #plotting the image and the predicted class
    plt.imshow(img)
    plt.title(f'Predicted to be : {pred_class}')
    plt.axis(False)
    return pred_class,pred_prob


def unzip_data(filename):
    """
    Unzips the input file into the current working directory
    :param filename: Zipped file name
    :return: None; unzips the files into the CWD
    """
    zip_ref = zipfile.ZipFile(filename,'r')
    zip_ref.extractall()
    zip_ref.close()

def calculate_results(y_true,y_pred):
    """
    calculates the classification metrics of the model
    :param y_true: true labels in form of 1D array
    :param y_pred: predicted labels in form of 1D array
    :return: a Dictionary of accuracy,precision,recall and f1 scores
    """

    #calculating the models accuracy
    model_accuracy = accuracy_score(y_true,y_pred)*100

    #calculating the models precision,accuracy,recall and f1 score
    models_precision,models_recall,models_f1,_ = precision_recall_fscore_support(y_true,y_pred,average="weighted")
    model_results = {"accuracy": model_accuracy,
                     "precision": models_precision,
                     "recall": models_recall,
                     "f1": models_f1}

    return model_results