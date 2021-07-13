import os
import argparse
import tensorflow as tf 
import numpy as np 
import matplotlib.pyplot as plt 
from utils import gen_maker, CustomCallback
from networks import model_maker
from tensorflow.keras.losses import categorical_crossentropy
import tensorflow_addons as tfa
import pickle as pkl
from networks import *
# gpus = tf.config.experimental.list_physical_devices('GPU') 
# tf.config.experimental.set_memory_growth(gpus[0], True)

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default = 100, help='number of total epochs')
parser.add_argument('--init_epoch', type=int, default = 0, help='initial epoch')
parser.add_argument('--train_dir', type=str, default = '/content/train/', help = 'training data folder path')
parser.add_argument('--val_dir', type=str, default = '/content/val/', help = 'validation data folder path')
parser.add_argument('--model_id', type = int, default = 1, help = 'model ID. it can bee 1, 2, 3 or 4')
parser.add_argument('--load_model', type = int, default = 0, help = 'if 1, you should specify the address of the previously trained model to load it')
parser.add_argument('--load_path', type = str, default= None, help = 'path to pre-trained models')
parser.add_argument('--backup_path', type = str, default = '/content/', help ='path to store the model')
parser.add_argument('--bach_size', type = int, default = 16, help = 'batch size for training')
parser.add_argument('--mode', type = str, default = 'categorical', help = 'classification mode')
parser.add_argument('--target_size', type = int, default = 100, help ='size of the input images')
parser.add_argument('--batch_size', type = int, default = 16, help ='batch size')
parser.add_argument('--num_classes', type = int, default = 3, help ='no. of classes')
parser.add_argument('--FC_units', type = int, default = 50, help ='FC units')

args = parser.parse_args()
    
def main():

    train_gen, val_gen = gen_maker(args.train_dir,
     args.val_dir,
      target_size=(args.target_size, args.target_size),
       batch_size=args.batch_size,
       mode=args.mode)
    
    clbk = CustomCallback(val_gen, args.backup_path, args.model_id)
    learning_rate = 0.001
    weight_decay = 0.0001
    model = model_maker((args.target_size, args.target_size),
                         args.model_id,
                         args.num_classes,
                         args.FC_units)
                         
    optimizer = tfa.optimizers.AdamW(
        learning_rate=learning_rate, weight_decay=weight_decay)

    model.compile(
        optimizer=optimizer,
        loss=categorical_crossentropy,
        metrics='acc')
        
    if args.load_model:
        model = tf.keras.models.load_model(args.load_path)
        #model.load_weights(args.load_path)
    results = model.fit(train_gen,
                    epochs = args.epochs,
                    validation_data = (val_gen),
                    callbacks = [clbk], initial_epoch=args.init_epoch)

    y_pred_valid = model.predict(val_gen)

    history = {'train loss' : results.history['loss'],
           'val loss' : results.history['val_loss'], 
           'train acc': results.history['acc'],
           'val acc' : results.history['val_acc'],
           'y_true_valid': val_gen.classes,
           'y_pred_valid': y_pred_valid, 
           'id': args.model_id}
    with open(args.backup_path + 'history-id-{}.pkl'.format(args.model_id), 'wb') as f:
        pkl.dump(history, f)
    
    plt.subplots(figsize = (15, 15))
    plt.subplot(2, 1, 1)
    plt.plot(results.history['loss'], '-', color = [0, 0, 1, 1])
    plt.plot(results.history['val_loss'], '-', color = [1, 0, 0, 1])
    plt.legend(['train loss', 'val loss'])
    plt.subplot(2, 1, 2)
    plt.plot([0, *results.history['acc']], '-', color = [0, 0, 1, 1])
    plt.plot([0, *results.history['val_acc']], '-', color = [1, 0, 0, 1])
    plt.legend(['train acc', 'val acc'])
    plt.savefig(args.backup_path + 'charts.png')


if __name__ == "__main__":
    main()
