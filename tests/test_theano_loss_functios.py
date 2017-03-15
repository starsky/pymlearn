# -*- coding: utf-8 -*-

from context import _loss_func_semi_vectorized
from context import _loss_func_theano
import unittest
import sklearn.preprocessing
import theano
import numpy as np


class TheanoLossFunctionsTestSuite(unittest.TestCase):
    """Advanced test cases."""

    def test_hinge_loss(self):
        W = np.random.random((10, 1000)).astype(theano.config.floatX)
        X = np.random.random((10000, 1000)).astype(theano.config.floatX)
        Y = np.random.randint(0, 10, 10000).astype(np.int32)[:, np.newaxis]
        to_binary_label = sklearn.preprocessing.MultiLabelBinarizer()
        Y = to_binary_label.fit_transform(Y).astype(theano.config.floatX).T

        reference_loss = _loss_func_semi_vectorized.hinge_loss(W, X, Y)
        reference_gradient = _loss_func_semi_vectorized.hinge_loss_derivatives(W, X, Y)

        hinge_loss, hinge_loss_derivatives = _loss_func_theano._compile_hinge_loss_func(compile_=True)

        loss = hinge_loss(W, X, Y)
        gradient = hinge_loss_derivatives(W, X, Y)

        np.testing.assert_almost_equal(reference_loss, loss)
        np.testing.assert_array_almost_equal(reference_gradient, gradient)

    def test_softmax_loss(self):
        W = np.random.random((10, 1000)).astype(theano.config.floatX)
        X = np.random.random((10000, 1000)).astype(theano.config.floatX)
        Y = np.random.randint(0, 10, 10000).astype(np.int32)[:, np.newaxis]
        to_binary_label = sklearn.preprocessing.MultiLabelBinarizer()
        Y = to_binary_label.fit_transform(Y).astype(theano.config.floatX).T

        reference_loss = _loss_func_semi_vectorized.softmax_loss(W, X, Y)
        reference_gradient = _loss_func_semi_vectorized.softmax_loss_derivatives(W, X, Y)

        softmax_loss, softmax_loss_derivatives = _loss_func_theano._compile_softmax_loss_func(compile_=True)
        loss = softmax_loss(W, X, Y)
        gradient = softmax_loss_derivatives(W, X, Y)

        np.testing.assert_almost_equal(reference_loss, loss)
        np.testing.assert_array_almost_equal(reference_gradient, gradient)

    def test_l1_penalty(self):
        W = np.random.random((10, 1000))

        reference_loss = _loss_func_semi_vectorized.l1_penalty(W)
        reference_gradient = _loss_func_semi_vectorized.l1_penalty_der(W)

        l1_penalty, l1_penalty_der = _loss_func_theano._compile_l1_penalty_func(compile_=True)
        loss = l1_penalty(W)
        gradient = l1_penalty_der(W)

        np.testing.assert_almost_equal(reference_loss, loss)
        np.testing.assert_array_almost_equal(reference_gradient, gradient)

    def test_l2_penalty(self):
        W = np.random.random((10, 1000))

        reference_loss = _loss_func_semi_vectorized.l2_penalty(W)
        reference_gradient = _loss_func_semi_vectorized.l2_penalty_der(W)
        loss_func, gradient_func = _loss_func_theano._compile_l2_penalty_func(compile_=True)
        loss = loss_func(W)
        gradient = gradient_func(W)

        np.testing.assert_almost_equal(reference_loss, loss)
        np.testing.assert_array_almost_equal(reference_gradient, gradient)


    def test_get_loss_function(self):
        W = np.random.random((10, 1000)).astype(theano.config.floatX).ravel()
        X = np.random.random((10000, 1000)).astype(theano.config.floatX)
        Y = np.random.randint(0, 10, 10000).astype(np.int32)[:, np.newaxis]
        to_binary_label = sklearn.preprocessing.MultiLabelBinarizer()
        Y = to_binary_label.fit_transform(Y).astype(theano.config.floatX).T
        reg_values = [0, 0.5]
        loss_values = ['softmax', 'hinge']
        penalty_values = ['L1', 'L2']

        for reg in reg_values:
            for loss_fn_name in loss_values:
                for penalty in penalty_values:
                    loss_ref_fun, loss_der_ref_fun = _loss_func_semi_vectorized.get_loss_function(loss_fn_name, penalty)
                    reference_loss = loss_ref_fun(W, X, Y, reg)
                    reference_gradient = loss_der_ref_fun(W, X, Y, reg)

                    loss_fun, loss_der_fun = _loss_func_theano.get_loss_function(loss_fn_name, penalty)
                    loss = loss_fun(W, X, Y, reg)
                    gradient = loss_der_fun(W, X, Y, reg)
                    np.testing.assert_almost_equal(reference_loss, loss)
                    np.testing.assert_array_almost_equal(reference_gradient, gradient)

if __name__ == '__main__':
    unittest.main()
