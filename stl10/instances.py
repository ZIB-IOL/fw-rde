import numpy as np
import importlib
import os

# GENERAL PARAMETERS
DATAPATH = os.path.join('data', 'keras_generators.py')
STATISTICSPATH = os.path.join(os.path.split(DATAPATH)[0], 'statistics')


def load_generator(path=DATAPATH, class_mode=None):
    generator_path = os.path.expanduser(path)
    generator_spec = importlib.util.spec_from_file_location('', generator_path)
    generator_module = importlib.util.module_from_spec(generator_spec)
    generator_spec.loader.exec_module(generator_module)
    return generator_module.load_test_data(class_mode=class_mode)


def load_statistics(path=STATISTICSPATH, mode='diag', rank=30):
    print('Dataset statistics will be loaded now.')
    try:
        if mode == 'half':
            loaded = np.load(
                os.path.join(
                    path,
                    '{}-mode-rank-{}.npz'.format(mode, rank)
                )
            )
        else:
            loaded = np.load(
                os.path.join(
                    path,
                    '{}-mode.npz'.format(mode))
            )
        return loaded['mean'], loaded['covariance']
    except Exception:
        if mode == 'half':
            print('No precomputed dataset statistics for {} mode with rank {} '
                  'were found at {}.'.format(mode, path))
        else:
            print('No precomputed dataset statistics for {} mode were '
                  'found at {}.'.format(mode, path))
