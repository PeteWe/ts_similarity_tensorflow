# -*- coding: utf-8 -*-
"""
Created on Mon Jan  21 16:57:0 2019

@author: Peter Wolf:
Mail: wolf_p@outlook.com.
"""
import tensorflow as tf
from keras import backend as K

class EDTF():
    @staticmethod
    def ED(X,Y):
		"""
		Calculates the Euclidean distance of multivariate time series
		of the same length.
		
		:param X: 1st multivariate time series of dimension m x n.
		:param Y: 2nd multivariate time series of dimension m x p.
		:return: Euclidean distance matrix of dimension n x p.
		"""
        diff_per_feat = K.expand_dims(X, axis = 1) - Y
        diff_square = K.square(diff_per_feat)
        diff_sum_time = K.sum(diff_square, axis=2)
        sim_result = K.sum(diff_sum_time, axis=2)
        return tf.sqrt(sim_result)