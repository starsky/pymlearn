solver_backend = 'theano'
loss_backend = 'theano'

_available_backends = ['theano', 'numpy_vectorized', 'numpy_semi-vectorized']
_available_loss_backends = _available_backends
_available_solver_backends = ['theano', 'python']


def set_loss_backend(new_backend):
    global loss_backend
    if new_backend not in _available_loss_backends:
        raise ValueError('Unsupported backend: %s. Possible backends are %s' %
                         (new_backend, ', '.join(_available_loss_backends)))
    loss_backend = new_backend


def set_solver_backend(new_backend):
    global solver_backend
    if new_backend not in _available_solver_backends:
        raise ValueError('Unsupported backend: %s. Possible backends are %s' %
                         (new_backend, ', '.join(_available_solver_backends)))
    solver_backend = new_backend
