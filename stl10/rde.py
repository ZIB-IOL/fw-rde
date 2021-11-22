import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.keras.backend as K
from models import load_model, load_adfmodel
import instances


# GENERAL PARAMETERS
MODE = 'diag'       # 'diag', 'half', or 'full'
RANK = 200           # only affects 'half' mode
IMG_SHAPE = [224, 224, 3]

# LOAD MODEL
adfmodel = load_adfmodel(mode=MODE)
model = load_model()

# LOAD DATA STATISTICS AND SAMPLE GENERATOR
mean, covariance = instances.load_statistics(mode=MODE, rank=RANK)
mean = np.expand_dims(mean, 0)
covariance = np.expand_dims(covariance, 0)
generator = instances.load_generator()
if MODE == 'diag':
    mean = mean.mean() * np.ones_like(mean)
    covariance = covariance.mean() * np.ones_like(covariance)


def get_data_sample(index):
    return (
        generator[index],
        os.path.splitext(os.path.split(generator.filenames[index])[1])[0],
    )


def store_single_result(mapping, index, fname, rate):
    savedir = os.path.join('results', fname)
    os.makedirs(savedir, exist_ok=True)
    mapping = np.reshape(mapping, IMG_SHAPE)
    plt.imsave(
        os.path.join(
            savedir,
            '{}-mode-rate{}-nx.png'.format(MODE, rate),
        ),
        np.mean(mapping, axis=-1).squeeze(),
        cmap='Reds',
        vmin=0.0,
        vmax=1.0,
        format='png',
    )


def store_collected_results(mappings, index, node, pred, fname, rates,
                            weights=None, perm=None, order=None):
    savedir = os.path.join('results', fname)
    os.makedirs(savedir, exist_ok=True)
    mappings = np.reshape(mappings, [len(rates)]+IMG_SHAPE)
    plt.imsave(
        os.path.join(
            savedir,
            '{}-mode-rates-averaged-nx.png'.format(MODE),
        ),
        np.mean(np.mean(mappings, axis=-1), axis=0).squeeze(),
        cmap='Reds',
        vmin=0.0,
        vmax=1.0,
        format='png',
    )
    if order is not None:
        order = np.reshape(order, IMG_SHAPE)
        plt.imsave(
            os.path.join(
                savedir,
                '{}-mode-rates-ordered-nx.png'.format(MODE),
            ),
            np.mean(order, axis=-1).squeeze(),
            cmap='Reds',
            vmin=0.0,
            vmax=1.0,
            format='png',
        )
    np.savez_compressed(
        os.path.join(
            savedir,
            '{}-mode-rates-nx.npz'.format(MODE),
        ),
        **{
            'mapping': np.average(mappings, weights=weights, axis=0).squeeze(),
            'mappings': mappings,
            'rates': rates,
            'index': index,
            'mode': MODE,
            'node': node,
            'prediction': pred,
            'rank': RANK,
            'weights': weights,
            'perm': perm,
            'order': order,
        }
    )


# squared distance distortion objective
def get_distortion(x, mean=mean, covariance=covariance, model=model,
                   adfmodel=adfmodel, mode=MODE):
    x_tensor = tf.constant(x, dtype=tf.float32)
    m_tensor = tf.constant(mean, dtype=tf.float32)
    c_tensor = tf.constant(covariance, dtype=tf.float32)

    print(c_tensor.shape)

    s_flat = tf.placeholder(tf.float32, (np.prod(x_tensor.shape),))
    s_tensor = tf.reshape(s_flat, x.shape)
    pred = model.predict(x)
    node = np.argpartition(pred[0, ...], -2)[-1]
    target = pred[0, node]
    mean_in = s_tensor*x_tensor + (1-s_tensor)*m_tensor
    if mode == 'diag':
        covariance_in = tf.square(1-s_tensor)*c_tensor
    elif mode == 'half':
        covariance_in = c_tensor*(1-s_tensor)
    elif mode == 'full':
        covrank = len(c_tensor.get_shape().as_list())
        perm = ([0] + list(range((covrank-1)//2+1, covrank))
                + list(range(1, (covrank-1)//2+1)))
        covariance_in = c_tensor*(1-s_tensor)
        covariance_in = K.permute_dimensions(
            covariance_in,
            perm,
        )
        covariance_in = covariance_in*(1-s_tensor)
        covariance_in = K.permute_dimensions(
            covariance_in,
            perm,
        )
    out_mean, out_covariance = adfmodel([mean_in, covariance_in])
    if mode == 'diag':
        loss = 1/2*(K.mean(K.square(out_mean[..., node]-target))
                    + K.mean(out_covariance[..., node]))
    elif mode == 'half':
        out_covariance = K.sum(K.square(out_covariance), axis=1)
        loss = 1/2*(K.mean(K.square(out_mean[..., node]-target))
                    + K.mean(out_covariance[..., node]))
    elif mode == 'full':
        loss = 1/2*(K.mean(K.square(out_mean[..., node]-target))
                    + K.mean(out_covariance[..., node, node]))
    gradient = K.gradients(loss, [s_flat])[0]
    f_out = K.function([s_flat], [loss])
    f_gradient = K.function([s_flat], [gradient])
    return lambda s: f_out([s])[0], lambda s: f_gradient([s])[0], node, pred
