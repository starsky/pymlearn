# -*- coding: utf-8 -*-

from context import _loss_func_semi_vectorized
from context import _loss_func_vectorized
import unittest
import sklearn.preprocessing
import theano
import numpy as np


class VectorizedLossFunctionsTestSuite(unittest.TestCase):
    """Advanced test cases."""

    def test_hinge_loss(self):
        W = np.random.random((10, 1000)).astype(theano.config.floatX)
        X = np.random.random((10000, 1000)).astype(theano.config.floatX)
        Y = np.random.randint(0, 10, 10000).astype(np.int32)[:, np.newaxis]
        to_binary_label = sklearn.preprocessing.MultiLabelBinarizer()
        Y = to_binary_label.fit_transform(Y).astype(theano.config.floatX).T

        reference_loss = _loss_func_semi_vectorized.hinge_loss(W, X, Y)
        reference_gradient = _loss_func_semi_vectorized.hinge_loss_derivatives(W, X, Y)
        loss = _loss_func_vectorized.hinge_loss(W, X, Y)
        gradient = _loss_func_vectorized.hinge_loss_derivatives(W, X, Y)

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
        loss = _loss_func_vectorized.softmax_loss(W, X, Y)
        gradient = _loss_func_vectorized.softmax_loss_derivatives(W, X, Y)

        np.testing.assert_almost_equal(reference_loss, loss)
        np.testing.assert_array_almost_equal(reference_gradient, gradient)

    def test_l1_penalty(self):
        W = np.random.random((10, 1000))

        reference_loss = _loss_func_semi_vectorized.l1_penalty(W)
        reference_gradient = _loss_func_semi_vectorized.l1_penalty_der(W)
        loss = _loss_func_vectorized.l1_penalty(W)
        gradient = _loss_func_vectorized.l1_penalty_der(W)

        np.testing.assert_almost_equal(reference_loss, loss)
        np.testing.assert_array_almost_equal(reference_gradient, gradient)

    def test_l2_penalty(self):
        W = np.random.random((10, 1000))

        reference_loss = _loss_func_semi_vectorized.l2_penalty(W)
        reference_gradient = _loss_func_semi_vectorized.l2_penalty_der(W)
        loss = _loss_func_vectorized.l2_penalty(W)
        gradient = _loss_func_vectorized.l2_penalty_der(W)

        np.testing.assert_almost_equal(reference_loss, loss)
        np.testing.assert_array_almost_equal(reference_gradient, gradient)

if __name__ == '__main__':
    unittest.main()
