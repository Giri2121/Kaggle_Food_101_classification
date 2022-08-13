import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping
from Helper_functions import split_data
from Helper_functions import create_generators
from Helper_functions import performance_plots
from Deep_Learning_Model import *
import os
import keras

if __name__ == '__main__':
    os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
    path_to_data = r"C:\Users\vikassaigiridhar\Music\food_101_test\Images"
    path_to_train = r"C:\Users\vikassaigiridhar\Music\food_101_test\Train_Images"
    path_to_test = r"C:\Users\vikassaigiridhar\Music\food_101_test\Test_Images"
    epochs = 20
    batch_size = 2
    lr = 0.001
    IMG_SIZE = (227, 227)

    train_data = tf.keras.preprocessing.image_dataset_from_directory(path_to_train,
                                                                     label_mode='categorical',
                                                                     image_size=IMG_SIZE,
                                                                     shuffle=True)

    test_data = tf.keras.preprocessing.image_dataset_from_directory(path_to_test,
                                                                    label_mode='categorical',
                                                                    image_size=IMG_SIZE,
                                                                    shuffle=False)

    num_classes = test_data.class_names
    no_of_classes = len(num_classes)

    path_to_save_model = './Models'
    check_point_saver = ModelCheckpoint(path_to_save_model,
                                        monitor='val_accuracy',
                                        mode='max',
                                        save_best_only=True,
                                        save_freq='epoch',
                                        verbose=1
                                        )

    early_stop = EarlyStopping(monitor='val_accuracy', patience=3)

    train_generator,test_generator = create_generators(batch_size,path_to_train,path_to_test)
    no_of_classes = train_generator.num_classes
    model = food_classification_model(no_of_classes)

    TRAIN = True
    TEST = False

    if TRAIN:
        path_to_save_model = './Models'
        url = 'https://tfhub.dev/tensorflow/efficientnet/b0/classification/1'
        check_point_saver = ModelCheckpoint(
            path_to_save_model,
            monitor = "val_accuracy",
            mode = "max",
            save_best_only = True,
            save_freq = 'epoch',
            verbose = 1
        )
        early_stop = EarlyStopping(monitor="val_accuracy",patience=3)
        #model = food_classification_model(no_of_classes)
        model = food_classification_model_pretrained(url,no_of_classes)
        optimizer = tf.keras.optimizers.Adam(learning_rate = lr,amsgrad = True)
        model.compile(
            optimizer=optimizer,
            loss = 'categorical_crossentropy',
            metrics=['accuracy']
        )

        history = model.fit(
            train_generator,
            epochs = epochs,
            steps_per_epoch = len(train_generator),
            batch_size=batch_size,
            validation_data = test_generator,
            validation_steps = len(test_generator),
            callbacks=[check_point_saver,early_stop]
        )
    
    if TEST:
        model = tf.keras.models.load_model('./Models')
        model.summary()

        print(f'Evaluating test set {model.evaluate(test_generator)}')
    
    #plotting the loss and accuracy curves
    performance_plots(history)