# -*- coding: utf-8 -*-
"""
Created on Thu Jan  3 13:54:04 2019

@author: Peter Wolf
"""
import tensorflow as tf
from keras import backend as K

class CIDTF():
    @staticmethod
    def euclidean(X,clusters):
        """
		Calculates the Euclidean distance of two time series
		of the same length.
		
		:param X: 1st multivariate time series of dimension m x n.
		:param Y: 2nd multivariate time series of dimension m x p.
		:return: Euclidean distance matrix of dimension n x p.
		"""
        diff_per_feat = K.expand_dims(X, axis = 1) - clusters
        diff_square = K.square(diff_per_feat)
        diff_sum_time = K.sum(diff_square, axis=2)
        sim_result = K.sum(diff_sum_time, axis=2)
        return tf.sqrt(sim_result)
    
    @staticmethod
    def complexity_factor(X):
        """
        Calculates the complexity factor (CF), i.e. length of a stretched time
        series.
        
        :param X: time series of dimension m x 1.
        :return: scalar CE of a time series X.
        """
        ew_diff = X[:,:-1,:] - X[:,1:,:]
        sum_X = tf.reduce_sum(tf.square(ew_diff), axis=1)
        return tf.sqrt(sum_X)
    
    @staticmethod
    def CID_tf(X,Y):
        """
        Calculates the complexity invariant distance (CID) of two time series X and
        Y. CID is the Euclidean distance corrected by a complexity factor.
        
        Note: Currently, this is an implementation for univariate time series
        only. Multivariate CF calculation will follow sometime.
        """
        ed = CIDTF.euclidean(X,Y)        
        ce_x = CIDTF.complexity_factor(X)
        ce_y = CIDTF.complexity_factor(Y)
        ce_x_mat = tf.expand_dims(tf.tile(ce_x, [1,ce_y.shape[0]]),axis=2)
        ce_y_mat = tf.expand_dims(tf.transpose(tf.tile(ce_y, [1,tf.shape(ce_x)[0]])),axis=2)
        min_max_mat = tf.concat([ce_x_mat,ce_y_mat], axis=2)

        ce_max = tf.reduce_max(min_max_mat,axis=2)
        ce_min = tf.reduce_min(min_max_mat, axis=2)
        ce = tf.divide(ce_max,ce_min)
        
        return tf.multiply(ed,ce)
