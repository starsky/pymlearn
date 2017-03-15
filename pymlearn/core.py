backend = 'theano'

_available_backends = ['theano', 'numpy_vectorized', 'numpy_semi-vectorized']


def set_backend(new_backend):
    """
    Sets new computation backend.
    :param new_backend: 'theano', 'numpy_vectorized', 'numpy_semi-vectorized'
    :return: None
    """
    global backend
    if new_backend not in _available_backends:
        raise ValueError('Unsupported backend: %s. Possible backends are %s' %
                         (new_backend, ', '.join(_available_backends)))
    backend = new_backend