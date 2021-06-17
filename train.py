import os
import argparse
import tensorflow as tf 
import numpy as np 
import matplotlib.pyplot as plt 
from utils import gen_maker, CustomCallback
from networks import model_maker
from tensorflow.keras.losses import categorical_crossentropy



def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, defualt = 100, help='number of total epochs')
    parser.add_argument('--init_epoch', type=int, defualt = 0, help='initial epoch')
    parser.add_argument('--train_dir', type=str, defualt = '/content/train/', help = 'training data folder path')
    parser.add_argument('--val_dir', type=str, defualt = '/content/val/', help = 'validation data folder path')
    parser.add_argument('--model_id', type = int, default = 1, help = 'model ID. it can bee 1, 2, 3 or 4')
    parser.add_argument('--load_model', type = int, defulat = 0, help = 'if 1, you should specify the address of the previously trained model to load it')
    parser.add_argument('--load_path', type = str, default= None, help = 'path to pre-trained models')
    parser.add_argument('--backup_path', type = str, default = '/content/', help ='path to store the model')
    parser.add_argument('--bach_size', type = int, default = 16, help = 'batch size for training')
    parser.add_argument('--mode', type = str, default = 'categorical', help = 'classification mode')
    parser.add_argument('--target_size', type = int, defualt = 100, help ='size of the input images')
    args = parser.parse_args()

    train_gen, val_gen = gen_maker(args.train_dir,
     args.val_dir,
      target_size=(args.target_size, args.target_size),
       batch_size=args.batch_size,
        mode=args.mode)
    
    clbk = CustomCallback(val_gen, args.backup_path, args.model_id)
    learning_rate = 0.001
    weight_decay = 0.0001
    model = model_maker((args.target_size, args.target_size), args.model_id)
    optimizer = tfa.optimizers.AdamW(
        learning_rate=learning_rate, weight_decay=weight_decay)

    model.compile(
        optimizer=optimizer,
        loss=categorical_crossentropy,
        metrics='acc')
        
    if args.load_model:
        model = tf.keras.load_model(args.load_path)

    results = model.fit(train_gen,
                    epochs = args.epochs,
                    validation_data = (val_gen),
                    callbacks = [clbk], init_epoch=args.init_epoch)

    with open(args.backup_path + 'history.pkl', 'wb') as f:
        pkl.dump(results, f)
    
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