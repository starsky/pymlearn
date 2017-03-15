import core
import _loss_func_theano
import _loss_func_semi_vectorized
import _loss_func_vectorized


_available_implementations = {'theano': _loss_func_theano.get_loss_function,
                              'numpy_semi-vectorized': _loss_func_semi_vectorized.get_loss_function,
                              'numpy-vectorized': _loss_func_vectorized.get_loss_function}


def get_loss_function(loss, penalty, init_params):
    """
    Returns loss function and loss function derivative composed of selected loss and penalty function.
    Both loss_function and loss_function_derivateve input parameters are W,X,Y,reg. Where
    W - model params vector. It is a vector obtained from matrix <#labels>X<#features>, <#features> = features size + bias
    X - Input features as matrix <#examples>X<#features>
    Y - Input labels as matrix <#examples>X<#labels> (one hot vector)
    reg - regularization param manges influence of penalty function as follows:
        final_loss_function = loss_function + reg * penalty_function
    The actual implementation of loss function depends on backend selected via core.set_backend
    :param loss: 'softmax' or 'hinge'
    :param penalty: 'L1' or 'L2'
    :return: loss_function, loss_function_derivative (both returned objects are python functions)
    """
    if core.solver_backend == 'theano':
        return _loss_func_theano.get_loss_function_not_compiled(loss, penalty, init_params)
    else:
        f, f_der = _available_implementations[core.loss_backend](loss, penalty)
        return f, f_der, None


def get_theano_not_compiled_loss(loss, penalty, init_params):
    return _loss_func_theano.get_loss_function_not_compiled(loss, penalty, init_params)
