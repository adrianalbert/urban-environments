import numpy as np
import pandas as pd

import h5py

from skimage.io import imread
from skimage.transform import resize, pyramid_reduce

from collections import Counter


def load_weights_into_model(model, weights_file, \
    layers_to_skip=None, transpose_conv=False):
    '''
    Loads pretrained weights from weights_file in to the structure defined by model. Has simple checks to ensure that weights from file are being loaded to the corresponding layers in the model.

    layers_to_skip is a list of either layer indices (ints) or names (strings).
    '''
    f = h5py.File(weights_file, "r")    
    if not layers_to_skip:
        layers_to_skip = []

    N_layers_file = f.attrs['nb_layers'] if 'nb_layers' in f.attrs.keys()\
                        else len(f.attrs['layer_names'])
    N_layers_model= len(model.layers)
    layers_to_load = [l.name for k,l in enumerate(model.layers) \
                            if len(l.get_weights())>0 and l.name not in layers_to_skip and k not in layers_to_skip]

    print "Loading %d layers from file into model..."%len(layers_to_load)

    l_file = 0
    for l_model in layers_to_load:
        # go to next layer in file that has weights to load
        while True:
            g = f['layer_{}'.format(l_file)]
            weights=[g['param_{}'.format(p)] \
                        for p in range(g.attrs['nb_params'])]
            l_file += 1
            if len(weights)>0:
                break
        print model.get_layer(l_model).name, 
        # transpose convolutional layers saved with a different backend   
        layer = model.get_layer(l_model)    
        k = [k for k,l in enumerate(model.layers) if l.name==l_model][0]
        if layer.__class__.__name__ in ['Conv2D', 'Convolution2D'] and transpose_conv:
            kernel, bias = weights
            if kernel.ndim > 2:
                kernel = np.transpose(kernel, (2, 3, 1, 0))
            else:
                print 'reshaping ...'
                kernel = np.reshape(kernel, layer.get_weights()[0].shape)
                print kernel.shape
            model.layers[k].set_weights([kernel, bias])
        else:
            model.layers[k].set_weights(weights)
    print "done."
    f.close()

    return model
        

def load_and_preprocess(filename, new_shape=None, channels="RGB", 
    downsample=None, crop=None):
    '''
    Load image and do basic preprocessing.
        - resize image to a specified shape;
        - subtract ImageNet mean;
        - make sure output image data is 0...255 uint8.
    '''
    img = imread(filename) # RGB image
    if downsample is not None:
        img = pyramid_reduce(img)
    if img.max()<=1.0:
        img = img * 255.0 / img.max()
    if crop is not None:
        i = np.random.randint(crop/2, img.shape[0]-crop/2)
        j = np.random.randint(crop/2, img.shape[1]-crop/2)
        img = img[(i-crop/2):(i+crop/2),(j-crop/2):(j+crop/2)]
    if new_shape is not None:
        img = resize(img, new_shape, preserve_range=True)
    # imagenet_mean_bgr = np.array([103.939, 116.779, 123.68])
    imagenet_mean_rgb = np.array([123.68, 116.779, 103.939])
    for i in range(3):
        img[:,:,i] = img[:,:,i] - imagenet_mean_rgb[i]
    # for VGG networks pre-trained on ImageNet, channels are BGR 
    # (ports from Caffe)
    if channels=="BGR":
        img = img[:, :, [2,1,0]] # swap channel from RGB to BGR
    return img.astype(np.uint8)


def balanced_df(df, nrows=None, k=1, class_column="class"):
    cnts = df[class_column].value_counts()
    min_cnt = cnts.min()
    ret = []
    for c in cnts.index:
        ret.append(df[df[class_column]==c].sample(min([cnts[c],k*min_cnt])))
    ret = pd.concat(ret)
    if nrows is not None:
        if len(ret) < nrows:
            weights = 1.0 / (df[class_column].value_counts() + 1)
            weights = {i:w for i,w in zip(weights.index, weights.values)}
            weights = df[class_column].apply(lambda x: weights[x])
            ret = pd.concat([ret, df.sample(nrows-len(ret), weights=weights)])
        else:
            ret = ret.sample(nrows)
    return ret


def get_class_weights(y):
    counter = Counter(y)
    majority = max(counter.values())
    return  {cls: float(majority)/count for cls, count in counter.items()}


def generator_from_df(df, image_generator=None, balance=None, \
                        class_column="class", filename_column="filename",
                        batch_size=32, seed=None, new_img_shape=None, \
                        class_dict=None, shuffle=True, channels="RGB",
                        downsample=None, crop=None):
    idx = 0
    if class_dict is None:
        myclasses = df[class_column].unique()
        myclasses.sort()
        class_dict = {c:i for i,c in enumerate(myclasses)}
    ok = True
    while ok:
        if shuffle:
            if balance is not None:
                df_bal = balanced_df(df, k=balance, nrows=batch_size,
                    class_column=class_column)
            else:
                df_bal = df
            df_batch = df_bal.sample(batch_size, random_state=seed)
        else:
            df_batch = df.iloc[idx:(idx+batch_size)]
            print "Reading ids %d -- %d"%(idx, idx+batch_size)
            if idx + batch_size >= len(df):
                print "should stop now!"
                ok = False
            idx += batch_size
        y = []
        X = []
        for i,r in df_batch.iterrows():
            img = load_and_preprocess(r[filename_column], 
                                      new_shape=new_img_shape, crop=crop,
                                      channels=channels, downsample=downsample)
            X.append(img)
            y.append(class_dict[r[class_column]])
        X = np.array(X)
        y = np.array(y)
        yoh = np.zeros((len(y), len(class_dict)))
        yoh[np.arange(len(y)), y] = 1
        if image_generator is not None:
            for X_batch, y_batch in image_generator.flow(X, yoh, batch_size=batch_size, shuffle=shuffle):
                break
        else:
            X_batch, y_batch = X, yoh
        yield (X_batch, y_batch)
        


def generator_from_file(filename, image_generator=None, balance=None, \
                        batch_size=32, seed=None, new_img_shape=None, \
                        class_dict=None, shuffle=True, channels="RGB",
                        downsample=False, crop=None):
    df = pd.read_csv(filename)
    return generator_from_df(df, 
                             image_generator=image_generator, 
                             balance=balance,
                             batch_size=batch_size, 
                             seed=seed, 
                             crop=crop,
                             new_img_shape=new_img_shape,
                             class_dict=class_dict, 
                             shuffle=shuffle,
                             channels=channels)
        