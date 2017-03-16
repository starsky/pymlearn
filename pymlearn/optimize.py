# -*- coding: utf-8 -*-
import numpy as np
import scipy.optimize
from functools import partial
import core
import theano
from lasagne.updates import sgd
__author__ = 'mkopersk'


def gradient_descend_theano(fun, x0, args=None, learning_rate=1e-3, tol=1e-3, max_iter=3000, verbose=True):
    funct, trainable_params, non_trainable_params = fun
    updates = sgd(funct, trainable_params, learning_rate=learning_rate)
    train_fun = theano.function(non_trainable_params, funct, updates=updates)
    loss_fn_compiled = theano.function(non_trainable_params, funct)
    old_loss = np.inf
    for i in range(max_iter):
        train_fun(*args)
        curr_loss = loss_fn_compiled(*args)
        if abs(curr_loss - old_loss) < tol:
            break
        old_loss = curr_loss
        _print_optimizer_iteration_info(verbose, i, old_loss)
    _print_optimizer_final_info(verbose, i, old_loss, 'Lasagne Gradient Descend')
    params_optimal = trainable_params[0].get_value()
    return {'x': params_optimal}

 
def gradient_descend(fun, x0, args=None, jac=None, learning_rate=1e-3, tol=1e-3, max_iter=1000, verbose=True):    
    old_loss = np.inf
    if jac is None:
        jac = lambda params: compute_partial_derivatives_numerical(fun, params, args=args)
    x0 = x0.astype(np.float64)
    for i in range(max_iter):
        loss = fun(x0, *args)
        if np.abs(old_loss - loss) < tol:
            break
        old_loss = loss
        x0 -= jac(x0, *args) * learning_rate
        _print_optimizer_iteration_info(verbose, i, old_loss)
    _print_optimizer_final_info(verbose, i, old_loss, 'Python Gradient Descend')
    return {'x': x0}


def _print_optimizer_iteration_info(verbose, iter_num, loss_val):
    if verbose and iter_num % 100 == 0:
        print 'Iteration %03d:\tloss value: %f' % (iter_num, loss_val)


def _print_optimizer_final_info(verbose, iter_num, loss_val, msg):
    if verbose:
        print 'Optimizer %s, converged' % msg
        print 'Iterations count: %d' % iter_num
        print 'Loss value: %f' % loss_val


def compute_partial_derivatives_numerical(func, parameters, args=None, step=1e-5):
    current_function_value = func(parameters, *args)
    partial_step = np.eye(parameters.ravel().shape[0]) * step
    new_parameters = parameters.ravel()[:, np.newaxis] + partial_step
    derivatives = map(lambda new_partial_param:
                      (func(new_partial_param.reshape(parameters.shape), *args)
                       - current_function_value) / step, new_parameters.T)
    return np.vstack(derivatives).sum(axis=1).reshape(parameters.shape)


def solve(solver, loss_func, jac=None, tol=1e-5, max_iter=1000, verbose=False, params=None):
    if solver == 'GD' and core.solver_backend == 'theano':
        fun = (loss_func, [params[0]], params[1:])
        r = partial(gradient_descend_theano, fun, max_iter=max_iter, tol=tol, verbose=verbose)
        return r
    elif solver == 'BFGS':
        opts = {'disp': verbose, 'maxiter': max_iter}
        r = partial(scipy.optimize.minimize, loss_func, method='BFGS', jac=jac,
                    options=opts, tol=tol)
        return r
    elif solver == 'GD':
        r = partial(gradient_descend, loss_func, jac=jac, max_iter=max_iter, tol=tol,
                    verbose=verbose)
        return r
    else:
        raise ValueError('Wrong solver name')
