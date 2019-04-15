"""
Created on Thu Jan  3 13:54:04 2019

@author: Peter Wolf
Mail: wolf_p@outlook.com.

Source: Cuturi et al. (2017) - Soft-DTW: a Differentiable Loss Function
for Time-Series.

Implemented for Tensorflow 1.12.0.
"""
import tensorflow as tf
import sys

DBL_MAX = sys.float_info.max

class SoftDTWTF():

    @staticmethod 
    def _squared_euclidean_tf(X, Y, squared=True):
        """
        Computes pairwise distances between each element of X and each
        element of Y.
        Args:
        X,    [s,m,d] matrix
        Y,    [p,n,d] matrix
        Returns:
        D,    [s,p,m,n] matrix of pairwise distances
        """
        # squared norms of each row in X and Y. sum over feature axis.
        nx = tf.reduce_sum(tf.square(X), 2)
        ny = tf.reduce_sum(tf.square(Y), 2)
        # nx as column vector for every sample in X (-> expand_axis=2).
        nx = tf.expand_dims(tf.expand_dims(nx, axis=2), axis=1)
        # ny as row vector for every cluster in Y.
        ny = tf.expand_dims(tf.expand_dims(ny, axis=0), axis=2)

        # Dot product for each sample in X and each cluster in Y.
        XY_matmul = tf.einsum('mno,pho->mpnh',X,Y)
        
        # return matrix of pairwise euclidean differences.
        if squared:
            D = tf.maximum(nx - 2*XY_matmul + ny, 0.0)
        else:
            D = tf.sqrt(tf.maximum(nx - 2*XY_matmul + ny, 0.0))
        
        return tf.cast(D, dtype=tf.float64)
    
    @staticmethod
    def _squared_euclidean_tf_eff(X, squared=True):
        """
        Computes pairwise distances of element in diagonale of X. Calculations
        reduced to diagnoale to decrease computational cost. Required for
        normalization of SDTW.
        
        Args:
        X,    [s,m,d] matrix
        Returns:
        D,    [s,m] matrix of distances of elements on diagonale of X.
        """
        # squared norms of each row in X and Y. sum over feature axis.
        nx = tf.reduce_sum(tf.square(X), 2)
        # nx as column vector for every sample in X (-> expand_axis=2).
        nx_t = tf.expand_dims(nx,axis=2)
        # ny as row vector for every cluster in Y.
        ny = tf.expand_dims(nx, axis=1)

        # dot product for each sample in X and each cluster in Y.
        # M = p
        XX_matmul = tf.matmul(X,X, False,True)

        # return matrix of pairwise euclidean differences.
        if True:
            D = tf.maximum(nx_t - 2*XX_matmul + ny, 0.0)
        else:
            D = tf.sqrt(tf.maximum(nx_t - 2*XX_matmul + ny, 0.0))

        return tf.cast(D, dtype=tf.float64)
    
    @staticmethod    
    def _softmin3_tf(a, b, c, gamma=0.01):
        """
        Softmin as defined in by Cuturi et al. (2017) - Soft-DTW: a
        Differentiable Loss Function for Time-Series. a,b,c are elements in
        cost matrix R.
        
        Args:
        a,    R[:,:,i-1,j]
        b,    R[:,:,i-1,j-1]
        c,    R[:,:,i,j-1]
        Returns:
        softmin,,    Soft minimum of inputs.
        """
        a = tf.divide(a, -gamma)
        b = tf.divide(b, -gamma)
        c = tf.divide(c, -gamma)
        max_val = tf.maximum(tf.maximum(a,b),c)
        
        tmp = tf.add(tf.constant(0.0, dtype=tf.float64), tf.exp(a - max_val))
        tmp = tf.add(tmp, tf.exp(b - max_val))
        tmp = tf.add(tmp, tf.exp(c - max_val))
        
        return -gamma * (tf.log(tmp) + max_val)
    
    @staticmethod
    def _cond_dp_rec2(j,i,n,D,R):
        # Condiction for loop 2.
        return tf.less(j, n+1) 

    @staticmethod
    def _cond_dp_rec1(i,m,n,D,R):
        # Condition for loop 1.
        return tf.less(i, m+1)

    @staticmethod 
    def _dp_rec1(i, m, n, D, R):
        j = tf.constant(1)
        j, i, n, D, R = tf.while_loop(SoftDTWTF._cond_dp_rec2,
                                      SoftDTWTF._dp_rec2,
                                      [j, i, n, D, R])
        return tf.add(i,1), m, n, D, R

    @staticmethod 
    def _dp_rec2(j, i, n, D, R):   
        """
        Dynamic programming loop of dynamic time warping to calculate cost
        matrix R.
        
        Args:
        j,    loop variable of inner (this) loop.
        i,    loop variable of outer loop.
        n,    temporal dimension.
        D,    pairwise distance matrix D.
        R,    Prefilled cost matrix R with 0 and inf.
        Returns:
        j,    incremented loop variable.
        i,    outer loop variable.
        n,    temporal dimension.
        D,    pairwise distance matrix D.
        R,    Cost matrix R filled with new element.
        """
        # Extract all distance elements for current step.
        shape = R.shape
        a = R[:,:,i-1,j]
        b = R[:,:,i-1,j-1]
        c = R[:,:,i,j-1]
        d = D[:,:,i-1,j-1]
        # Get softmin of all elements and add it to current step.
        soft_min_ij = tf.add(d, SoftDTWTF._softmin3_tf(a, b, c))
        # Concat new element in R for calc of next step.
        R_bef = R[:,:,:i,:]
        R_row = tf.concat([R[:,:,i,:j],
                           tf.expand_dims(soft_min_ij,axis=2),
                           R[:,:,i,j+1:]], axis=2)
        R_aft = R[:,:,i+1:,:]        
        R = tf.concat([R_bef,tf.expand_dims(R_row,axis=2),R_aft], axis=2)
        # Set shape of R again.
        R.set_shape(shape)
        return tf.add(j,1), i, n, D, R
    
    @staticmethod
    def _soft_dtw_core(D):
        """
        Core sdtf function which calculates cost matrix R based on pairwise
        distance matrix D.
        
        Args:
        D,    pairwise distance matrix D.
        Returns:
        R,    Final cost matrix R..
        """
        s, c, m, n = D.get_shape().as_list() # m = n here.
        zeros = lambda: tf.zeros([m+2, n+2], dtype=tf.float64)
        
        # Generate single initial R matrix.
        A = tf.Variable(zeros, dtype=tf.float64)
        B = tf.Variable(zeros, dtype=tf.float64)
        A = A[1:m+1,0].assign(tf.ones_like(A[1:m+1,0])*DBL_MAX)
        B = B[0,1:n+1].assign(tf.ones_like(B[0,1:n+1])*DBL_MAX)
        R_single = A + B
        
        # Replicate initial R s-times.
        R = tf.ones([tf.shape(D)[0],tf.shape(D)[1], 1, 1],
                     dtype=tf.float64) * R_single
        
        i = tf.constant(1)
        loop1_vars = [i, tf.constant(m), tf.constant(n), D, R]
        i, m, n, D, R = tf.while_loop(SoftDTWTF._cond_dp_rec1,
                                      SoftDTWTF._dp_rec1,
                                      loop1_vars)      
        return R

    @staticmethod
    def _sdtw_tf_mat(x,y):
        """
        SDTW calc for all elements in x and y.
        
        Args:
        x,    [s,m,d] matrix of time series with s samples of length m, and
              dimension d.
        y,    [p,n,d] matrix of time series with s samples of length m, and
              dimension d.
        Returns:
        dists_norm,    normalized distances of all elements in x and y.
        """
        # Get D for xy, xx, yy.
        D = SoftDTWTF._squared_euclidean_tf(x,y)
        D_xx = SoftDTWTF._squared_euclidean_tf_eff(x)
        D_yy = SoftDTWTF._squared_euclidean_tf_eff(y)
        # Expand dimension 0 to make D_.._diagn of same dimension as D.
        D_xx_diag = tf.expand_dims(D_xx,axis=0)
        D_yy_diag = tf.expand_dims(D_yy,axis=0)
        # Set shapes od D to avoid errors in initialization of graph.
        D.set_shape((None,None,y.shape[1],y.shape[1]))
        D_xx_diag.set_shape((None,None,y.shape[1],y.shape[1]))
        D_yy_diag.set_shape((None,None,y.shape[1],y.shape[1]))
        # Calculate cost matric for all D.
        R = SoftDTWTF._soft_dtw_core(D)
        R_xx = SoftDTWTF._soft_dtw_core(D_xx_diag)
        R_yy = SoftDTWTF._soft_dtw_core(D_yy_diag)
        # Extract xx and yy for normalization.
        s, s, m, n = D.shape
        xx = R_xx[:,:, m, n]
        yy = R_yy[:,:, m, n]
        # Normalize R of xy.
        dists_norm = R[:,:, m, n] - 0.5 * (tf.reshape(xx, [-1,1]) + yy)
        
        return dists_norm
    
    
    @staticmethod 
    def sdtw_tf(x,y):
        """
        Entry point for tensorflow version of normalized soft-dtw for
        multivariate time series.
        
        Args:
        x,    [s,m,d] matrix of time series with s samples of length m, and
              dimension d.
        y,    [p,n,d] matrix of time series with s samples of length m, and
              dimension d.
        Returns:
        SoftDTWTF._sdtw_tf_mat(x,y),    normalized distances of all elements
                                        in x and y.
        """
        return SoftDTWTF._sdtw_tf_mat(x,y)
