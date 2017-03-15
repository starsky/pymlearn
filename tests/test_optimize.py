# -*- coding: utf-8 -*-

from context import _loss_func_semi_vectorized
from context import _loss_func_theano
from context import optimize
import unittest
import sklearn.preprocessing
import theano
import numpy as np


class TheanoLossFunctionsTestSuite(unittest.TestCase):
    pass


if __name__ == '__main__':
    unittest.main()
