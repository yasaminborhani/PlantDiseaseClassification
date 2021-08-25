import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pickle as pkl
import os

def gen_maker(train_path, val_path, target_size=(100, 100), batch_size=16, mode='categorical'):
    """
    This function creates data generators for train and validation data.
    params:
        train_path: path to the training data folder, string.
        val_path: path to the validation data folder, string. 
        target_size: size of the inputs to the network, tuple.
        batch_size: the batch size for training and validation, integer. 
        mode: classification mode, it can be either "binary" or "categorical"
    returns:
        train_generator: data generator for training data.
        validation_generator: data generator for validation data.
    """

    train_datagen = ImageDataGenerator( rotation_range=10,
                                    width_shift_range=0.2,
                                    height_shift_range=0.2,
                                    shear_range=0.1,
                                    zoom_range=0.1,
                                    channel_shift_range=0.0,
                                    horizontal_flip=True,
                                    vertical_flip=True,
                                    rescale=1./255)

    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        train_path,  
        target_size=target_size,  
        batch_size=batch_size,
        class_mode=mode)  

    validation_generator = test_datagen.flow_from_directory(
        val_path,
        target_size=target_size,
        batch_size=batch_size,
        shuffle=False,
        class_mode=mode)
    return train_generator, validation_generator

class CustomCallback(tf.keras.callbacks.Callback):
    """
    This callback saves the model at the end of each epoch and calculates
    the confusion matrix and classification report on the validation data.
    """

    def __init__(self, val_gen, model_path, model_id):
        
        super(CustomCallback, self).__init__()
        self.val_gen = val_gen
        self.model_path = model_path
        self.model_id = model_id
        if not os.path.exists('temp.pkl'):
            with open('temp.pkl', 'wb') as f:
                dict = {}
                pkl.dump(dict, f)
    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1)%5 == 0 or (epoch+1)>=85:
            self.model.save(self.model_path + 'epoch{}-id{}'.format(epoch,self.model_id ))
            self.model.save_weights(self.model_path + 'modelWeights')
            #os.system('git add ' + self.model_path + 'epoch{}-id{}'.format(epoch,self.model_id ))
            #os.system('git rm -r ' + self.model_path + 'epoch{}-id{}'.format(epoch-5,self.model_id ))
            #os.system('git add ' + self.model_path)
            #os.system('cp temp.pkl ' + self.model_path + 'temp.pkl')
            #os.system('git add ' + self.model_path + 'temp.pkl')
            #os.system('git commit -m "model has been trained"')
            #os.system("git push origin HEAD:dev")
        y_pred = self.model.predict(self.val_gen)
        y_pred = np.squeeze(np.argmax(y_pred, axis = 1))
        y_true = self.val_gen.classes
        cnf = confusion_matrix(y_true, y_pred)
        cls_report = classification_report(y_true, y_pred, digits=4)
        print('\nclassification report:\n', cls_report)
        print('\nconfusion matrix:\n', cnf)
        with open('temp.pkl', 'rb') as f:
            dict = pkl.load(f)
        with open('temp.pkl', 'wb') as f:
            cls_report = classification_report(y_true, y_pred, digits=4, output_dict=True)
            dict[epoch] = {'cls_report':cls_report, 'logs':logs}
            pkl.dump(dict, f)
        
