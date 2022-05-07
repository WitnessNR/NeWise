from numba import njit
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, GlobalAveragePooling2D, Lambda
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, InputLayer, BatchNormalization, Reshape
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import mnist, cifar10
import tensorflow as tf


from utils import generate_data_myself, generate_data
import time
import datetime
from activations import *
linear_bounds = None

import random

def fn(correct, predicted):
        return tf.nn.softmax_cross_entropy_with_logits(labels=correct,
                                                       logits=predicted)
class CNNModel:
    def __init__(self, model, inp_shape = (28,28,1)):
        print('-----------', inp_shape, '---------')
        temp_weights = [layer.get_weights() for layer in model.layers]

        self.weights = []
        self.biases = []
        self.shapes = []
        self.pads = []
        self.strides = []
        self.model = model
        
        cur_shape = inp_shape
        self.shapes.append(cur_shape)
        for layer in model.layers:
            print(cur_shape)
            weights = layer.get_weights()
            if type(layer) == Conv2D:
                print('conv')
                if len(weights) == 1:
                    W = weights[0].astype(np.float32)
                    b = np.zeros(W.shape[-1], dtype=np.float32)
                else:
                    W, b = weights
                    W = W.astype(np.float32)
                    b = b.astype(np.float32)
                padding = layer.get_config()['padding']
                stride = layer.get_config()['strides']
                pad = (0,0,0,0) #p_hl, p_hr, p_wl, p_wr
                if padding == 'same':
                    desired_h = int(np.ceil(cur_shape[0]/stride[0]))
                    desired_w = int(np.ceil(cur_shape[0]/stride[1]))
                    total_padding_h = stride[0]*(desired_h-1)+W.shape[0]-cur_shape[0]
                    total_padding_w = stride[1]*(desired_w-1)+W.shape[1]-cur_shape[1]
                    pad = (int(np.floor(total_padding_h/2)),int(np.ceil(total_padding_h/2)),int(np.floor(total_padding_w/2)),int(np.ceil(total_padding_w/2)))
                cur_shape = (int((cur_shape[0]+pad[0]+pad[1]-W.shape[0])/stride[0])+1, int((cur_shape[1]+pad[2]+pad[3]-W.shape[1])/stride[1])+1, W.shape[-1])
                self.strides.append(stride)
                self.pads.append(pad)
                self.shapes.append(cur_shape)
                self.weights.append(W)
                self.biases.append(b)
            elif type(layer) == GlobalAveragePooling2D:
                print('global avg pool')
                b = np.zeros(cur_shape[-1], dtype=np.float32)
                W = np.zeros((cur_shape[0],cur_shape[1],cur_shape[2],cur_shape[2]), dtype=np.float32)
                for f in range(W.shape[2]):
                    W[:,:,f,f] = 1/(cur_shape[0]*cur_shape[1])
                pad = (0,0,0,0)
                stride = ((1,1))
                cur_shape = (1,1,cur_shape[2])
                self.strides.append(stride)
                self.pads.append(pad)
                self.shapes.append(cur_shape)
                self.weights.append(W)
                self.biases.append(b)
            elif type(layer) == AveragePooling2D:
                print('avg pool')
                b = np.zeros(cur_shape[-1], dtype=np.float32)
                pool_size = layer.get_config()['pool_size']
                stride = layer.get_config()['strides']
                W = np.zeros((pool_size[0],pool_size[1],cur_shape[2],cur_shape[2]), dtype=np.float32)
                for f in range(W.shape[2]):
                    W[:,:,f,f] = 1/(pool_size[0]*pool_size[1])
                pad = (0,0,0,0) #p_hl, p_hr, p_wl, p_wr
                if padding == 'same':
                    desired_h = int(np.ceil(cur_shape[0]/stride[0]))
                    desired_w = int(np.ceil(cur_shape[0]/stride[1]))
                    total_padding_h = stride[0]*(desired_h-1)+pool_size[0]-cur_shape[0]
                    total_padding_w = stride[1]*(desired_w-1)+pool_size[1]-cur_shape[1]
                    pad = (int(np.floor(total_padding_h/2)),int(np.ceil(total_padding_h/2)),int(np.floor(total_padding_w/2)),int(np.ceil(total_padding_w/2)))
                cur_shape = (int((cur_shape[0]+pad[0]+pad[1]-pool_size[0])/stride[0])+1, int((cur_shape[1]+pad[2]+pad[3]-pool_size[1])/stride[1])+1, cur_shape[2])
                self.strides.append(stride)
                self.pads.append(pad)
                self.shapes.append(cur_shape)
                self.weights.append(W)
                self.biases.append(b)
            elif type(layer) == Activation:
                print('activation')
            elif type(layer) == Lambda:
	            print('lambda')
            elif type(layer) == InputLayer:
                print('input')
            elif type(layer) == BatchNormalization:
                print('batch normalization')
                gamma, beta, mean, std = weights
                std = np.sqrt(std+0.001) #Avoids zero division
                a = gamma/std
                b = -gamma*mean/std+beta
                self.weights[-1] = a*self.weights[-1]
                self.biases[-1] = a*self.biases[-1]+b
            elif type(layer) == Dense:
                print('FC')
                W, b = weights
                b = b.astype(np.float32)
                W = W.reshape(list(cur_shape)+[W.shape[-1]]).astype(np.float32)
                cur_shape = (1,1,W.shape[-1])
                self.strides.append((1,1))
                self.pads.append((0,0,0,0))
                self.shapes.append(cur_shape)
                self.weights.append(W)
                self.biases.append(b)
            elif type(layer) == Dropout:
                print('dropout')
            elif type(layer) == MaxPooling2D:
                print('pool')
                pool_size = layer.get_config()['pool_size']
                stride = layer.get_config()['strides']
                pad = (0,0,0,0) #p_hl, p_hr, p_wl, p_wr
                if padding == 'same':
                    desired_h = int(np.ceil(cur_shape[0]/stride[0]))
                    desired_w = int(np.ceil(cur_shape[0]/stride[1]))
                    total_padding_h = stride[0]*(desired_h-1)+pool_size[0]-cur_shape[0]
                    total_padding_w = stride[1]*(desired_w-1)+pool_size[1]-cur_shape[1]
                    pad = (int(np.floor(total_padding_h/2)),int(np.ceil(total_padding_h/2)),int(np.floor(total_padding_w/2)),int(np.ceil(total_padding_w/2)))
                cur_shape = (int((cur_shape[0]+pad[0]+pad[1]-pool_size[0])/stride[0])+1, int((cur_shape[1]+pad[2]+pad[3]-pool_size[1])/stride[1])+1, cur_shape[2])
                self.strides.append(stride)
                self.pads.append(pad)
                self.shapes.append(cur_shape)
                self.weights.append(np.full(pool_size+(1,1),np.nan,dtype=np.float32))
                self.biases.append(np.full(1,np.nan,dtype=np.float32))
            elif type(layer) == Flatten:
                print('flatten')
            elif type(layer) == Reshape:
                print('reshape')
            else:
                print(str(type(layer)))
                raise ValueError('Invalid Layer Type')
        print(cur_shape)

        for i in range(len(self.weights)):
            self.weights[i] = np.ascontiguousarray(self.weights[i].transpose((3,0,1,2)).astype(np.float32))
            self.biases[i] = np.ascontiguousarray(self.biases[i].astype(np.float32))
    def predict(self, data):
        return self.model(data)


@njit
def conv(W, x, pad, stride):
    p_hl, p_hr, p_wl, p_wr = pad
    s_h, s_w = stride
    y = np.zeros((int((x.shape[0]-W.shape[1]+p_hl+p_hr)/s_h)+1, int((x.shape[1]-W.shape[2]+p_wl+p_wr)/s_w)+1, W.shape[0]), dtype=np.float32)
    for a in range(y.shape[0]):
        for b in range(y.shape[1]):
            for c in range(y.shape[2]):
                for i in range(W.shape[1]):
                    for j in range(W.shape[2]):
                        for k in range(W.shape[3]):
                            if 0<=s_h*a+i-p_hl<x.shape[0] and 0<=s_w*b+j-p_wl<x.shape[1]:
                                y[a,b,c] += W[c,i,j,k]*x[s_h*a+i-p_hl,s_w*b+j-p_wl,k]
    return y

@njit
def pool(pool_size, x0, pad, stride):
    p_hl, p_hr, p_wl, p_wr = pad
    s_h, s_w = stride
    y0 = np.zeros((int((x0.shape[0]+p_hl+p_hr-pool_size[0])/s_h)+1, int((x0.shape[1]+p_wl+p_wr-pool_size[1])/s_w)+1, x0.shape[2]), dtype=np.float32)
    for x in range(y0.shape[0]):
        for y in range(y0.shape[1]):
            for r in range(y0.shape[2]):
                cropped = LB[s_h*x-p_hl:pool_size[0]+s_h*x-p_hl, s_w*y-p_wl:pool_size[1]+s_w*y-p_wl,r]
                y0[x,y,r] = cropped.max()
    return y0

@njit
def conv_bound(W, b, pad, stride, x0, eps, p_n):
    y0 = conv(W, x0, pad, stride)
    UB = np.zeros(y0.shape, dtype=np.float32)
    LB = np.zeros(y0.shape, dtype=np.float32)
    for k in range(W.shape[0]):
        if p_n == 105: # p == "i", q = 1
            dualnorm = np.sum(np.abs(W[k,:,:,:]))
        elif p_n == 1: # p = 1, q = i
            dualnorm = np.max(np.abs(W[k,:,:,:]))
        elif p_n == 2: # p = 2, q = 2
            dualnorm = np.sqrt(np.sum(W[k,:,:,:]**2))
        mid = y0[:,:,k]+b[k]
        UB[:,:,k] = mid+eps*dualnorm
        LB[:,:,k] = mid-eps*dualnorm
    return LB, UB

@njit
def conv_full(A, x, pad, stride):
    p_hl, p_hr, p_wl, p_wr = pad
    s_h, s_w = stride
    y = np.zeros((A.shape[0], A.shape[1], A.shape[2]), dtype=np.float32)
    for a in range(y.shape[0]):
        for b in range(y.shape[1]):
            for c in range(y.shape[2]):
                for i in range(A.shape[3]):
                    for j in range(A.shape[4]):
                        for k in range(A.shape[5]):
                            if 0<=s_h*a+i-p_hl<x.shape[0] and 0<=s_w*b+j-p_wl<x.shape[1]:
                                y[a,b,c] += A[a,b,c,i,j,k]*x[s_h*a+i-p_hl,s_w*b+j-p_wl,k]
    return y

@njit
def conv_bound_full(A, B, pad, stride, x0, eps, p_n):
    y0 = conv_full(A, x0, pad, stride)
    UB = np.zeros(y0.shape, dtype=np.float32)
    LB = np.zeros(y0.shape, dtype=np.float32)
    for a in range(y0.shape[0]):
        for b in range(y0.shape[1]):
            for c in range(y0.shape[2]):
                if p_n == 105: # p == "i", q = 1
                    dualnorm = np.sum(np.abs(A[a,b,c,:,:,:]))
                elif p_n == 1: # p = 1, q = i
                    dualnorm = np.max(np.abs(A[a,b,c,:,:,:]))
                elif p_n == 2: # p = 2, q = 2
                    dualnorm = np.sqrt(np.sum(A[a,b,c,:,:,:]**2))
                mid = y0[a,b,c]+B[a,b,c]
                UB[a,b,c] = mid+eps*dualnorm
                LB[a,b,c] = mid-eps*dualnorm
    return LB, UB

@njit
def upper_bound_conv(A, B, pad, stride, W, b, inner_pad, inner_stride, inner_shape, LB, UB, method):
    A_new = np.zeros((A.shape[0], A.shape[1], A.shape[2], inner_stride[0]*(A.shape[3]-1)+W.shape[1], inner_stride[1]*(A.shape[4]-1)+W.shape[2], W.shape[3]), dtype=np.float32)
    B_new = np.zeros(B.shape, dtype=np.float32)
    A_plus = np.maximum(A, 0)
    A_minus = np.minimum(A, 0)
    alpha_u, alpha_l, beta_u, beta_l = linear_bounds(LB, UB, method)
    assert A.shape[5] == W.shape[0]

    for x in range(A_new.shape[0]):
        for y in range(A_new.shape[1]):
            for t in range(A_new.shape[3]):
                for u in range(A_new.shape[4]):
                    if 0<=t+stride[0]*inner_stride[0]*x-inner_stride[0]*pad[0]-inner_pad[0]<inner_shape[0] and 0<=u+stride[1]*inner_stride[1]*y-inner_stride[1]*pad[2]-inner_pad[2]<inner_shape[1]:
                        for p in range(A.shape[3]):
                            for q in range(A.shape[4]):
                                if 0<=t-inner_stride[0]*p<W.shape[1] and 0<=u-inner_stride[1]*q<W.shape[2] and 0<=p+stride[0]*x-pad[0]<alpha_u.shape[0] and 0<=q+stride[1]*y-pad[2]<alpha_u.shape[1]:
                                    for z in range(A_new.shape[2]):
                                        for v in range(A_new.shape[5]):
                                            for r in range(W.shape[0]):
                                                A_new[x,y,z,t,u,v] += W[r,t-inner_stride[0]*p,u-inner_stride[1]*q,v]*alpha_u[p+stride[0]*x-pad[0],q+stride[1]*y-pad[2],r]*A_plus[x,y,z,p,q,r]
                                                A_new[x,y,z,t,u,v] += W[r,t-inner_stride[0]*p,u-inner_stride[1]*q,v]*alpha_l[p+stride[0]*x-pad[0],q+stride[1]*y-pad[2],r]*A_minus[x,y,z,p,q,r]
                                                
    B_new = conv_full(A_plus,alpha_u*b+beta_u,pad,stride) + conv_full(A_minus,alpha_l*b+beta_l,pad,stride)+B
    return A_new, B_new


@njit
def lower_bound_conv(A, B, pad, stride, W, b, inner_pad, inner_stride, inner_shape, LB, UB, method):
    A_new = np.zeros((A.shape[0], A.shape[1], A.shape[2], inner_stride[0]*(A.shape[3]-1)+W.shape[1], inner_stride[1]*(A.shape[4]-1)+W.shape[2], W.shape[3]), dtype=np.float32)
    B_new = np.zeros(B.shape, dtype=np.float32)
    A_plus = np.maximum(A, 0)
    A_minus = np.minimum(A, 0)
    alpha_u, alpha_l, beta_u, beta_l = linear_bounds(LB, UB, method)
    assert A.shape[5] == W.shape[0]
    for x in range(A_new.shape[0]):
        for y in range(A_new.shape[1]):
            for t in range(A_new.shape[3]):
                for u in range(A_new.shape[4]):
                    if 0<=t+stride[0]*inner_stride[0]*x-inner_stride[0]*pad[0]-inner_pad[0]<inner_shape[0] and 0<=u+stride[1]*inner_stride[1]*y-inner_stride[1]*pad[2]-inner_pad[2]<inner_shape[1]:
                        for p in range(A.shape[3]):
                            for q in range(A.shape[4]):
                                if 0<=t-inner_stride[0]*p<W.shape[1] and 0<=u-inner_stride[1]*q<W.shape[2] and 0<=p+stride[0]*x-pad[0]<alpha_u.shape[0] and 0<=q+stride[1]*y-pad[2]<alpha_u.shape[1]:
                                    for z in range(A_new.shape[2]):
                                        for v in range(A_new.shape[5]):
                                            for r in range(W.shape[0]):
                                                A_new[x,y,z,t,u,v] += W[r,t-inner_stride[0]*p,u-inner_stride[1]*q,v]*alpha_l[p+stride[0]*x-pad[0],q+stride[1]*y-pad[2],r]*A_plus[x,y,z,p,q,r]
                                                A_new[x,y,z,t,u,v] += W[r,t-inner_stride[0]*p,u-inner_stride[1]*q,v]*alpha_u[p+stride[0]*x-pad[0],q+stride[1]*y-pad[2],r]*A_minus[x,y,z,p,q,r]
    B_new = conv_full(A_plus,alpha_l*b+beta_l,pad,stride) + conv_full(A_minus,alpha_u*b+beta_u,pad,stride)+B
    return A_new, B_new


@njit
def pool_linear_bounds(LB, UB, pad, stride, pool_size):
    p_hl, p_hr, p_wl, p_wr = pad
    s_h, s_w = stride
    alpha_u = np.zeros((pool_size[0], pool_size[1], int((UB.shape[0]+p_hl+p_hr-pool_size[0])/s_h)+1, int((UB.shape[1]+p_wl+p_wr-pool_size[1])/s_w)+1, UB.shape[2]), dtype=np.float32)
    beta_u = np.zeros((int((UB.shape[0]+p_hl+p_hr-pool_size[0])/s_h)+1, int((UB.shape[1]+p_wl+p_wr-pool_size[1])/s_w)+1, UB.shape[2]), dtype=np.float32)
    alpha_l = np.zeros((pool_size[0], pool_size[1], int((LB.shape[0]+p_hl+p_hr-pool_size[0])/s_h)+1, int((LB.shape[1]+p_wl+p_wr-pool_size[1])/s_w)+1, LB.shape[2]), dtype=np.float32)
    beta_l = np.zeros((int((LB.shape[0]+p_hl+p_hr-pool_size[0])/s_h)+1, int((LB.shape[1]+p_wl+p_wr-pool_size[1])/s_w)+1, LB.shape[2]), dtype=np.float32)

    for x in range(alpha_u.shape[2]):
        for y in range(alpha_u.shape[3]):
            for r in range(alpha_u.shape[4]):
                cropped_LB = LB[s_h*x-p_hl:pool_size[0]+s_h*x-p_hl, s_w*y-p_wl:pool_size[1]+s_w*y-p_wl,r]
                cropped_UB = UB[s_h*x-p_hl:pool_size[0]+s_h*x-p_hl, s_w*y-p_wl:pool_size[1]+s_w*y-p_wl,r]

                max_LB = cropped_LB.max()
                idx = np.where(cropped_UB>=max_LB)
                u_s = np.zeros(len(idx[0]), dtype=np.float32)
                l_s = np.zeros(len(idx[0]), dtype=np.float32)
                gamma = np.inf
                for i in range(len(idx[0])):
                    l_s[i] = cropped_LB[idx[0][i],idx[1][i]]
                    u_s[i] = cropped_UB[idx[0][i],idx[1][i]]
                    if l_s[i] == u_s[i]:
                        gamma = l_s[i]

                if gamma == np.inf:
                    gamma = (np.sum(u_s/(u_s-l_s))-1)/np.sum(1/(u_s-l_s))
                    if gamma < np.max(l_s):
                        gamma = np.max(l_s)
                    elif gamma > np.min(u_s):
                        gamma = np.min(u_s)
                    weights = ((u_s-gamma)/(u_s-l_s)).astype(np.float32)
                else:
                    weights = np.zeros(len(idx[0]), dtype=np.float32)
                    w_partial_sum = 0
                    num_equal = 0
                    for i in range(len(idx[0])):
                        if l_s[i] != u_s[i]:
                            weights[i] = (u_s[i]-gamma)/(u_s[i]-l_s[i])
                            w_partial_sum += weights[i]
                        else:
                            num_equal += 1
                    gap = (1-w_partial_sum)/num_equal
                    if gap < 0.0:
                        gap = 0.0
                    elif gap > 1.0:
                        gap = 1.0
                    for i in range(len(idx[0])):
                        if l_s[i] == u_s[i]:
                            weights[i] = gap

                for i in range(len(idx[0])):
                    t = idx[0][i]
                    u = idx[1][i]
                    alpha_u[t,u,x,y,r] = weights[i]
                    alpha_l[t,u,x,y,r] = weights[i]
                beta_u[x,y,r] = gamma-np.dot(weights, l_s)
                growth_rate = np.sum(weights)
                if growth_rate <= 1:
                    beta_l[x,y,r] = np.min(l_s)*(1-growth_rate)
                else:
                    beta_l[x,y,r] = np.max(u_s)*(1-growth_rate)
    return alpha_u, alpha_l, beta_u, beta_l

@njit
def upper_bound_pool(A, B, pad, stride, pool_size, inner_pad, inner_stride, inner_shape, LB, UB):
    A_new = np.zeros((A.shape[0], A.shape[1], A.shape[2], inner_stride[0]*(A.shape[3]-1)+pool_size[0], inner_stride[1]*(A.shape[4]-1)+pool_size[1], A.shape[5]), dtype=np.float32)
    B_new = np.zeros(B.shape, dtype=np.float32)
    A_plus = np.maximum(A, 0)
    A_minus = np.minimum(A, 0)
    alpha_u, alpha_l, beta_u, beta_l = pool_linear_bounds(LB, UB, inner_pad, inner_stride, pool_size)

    for x in range(A_new.shape[0]):
        for y in range(A_new.shape[1]):
            for t in range(A_new.shape[3]):
                for u in range(A_new.shape[4]):
                    inner_index_x = t+stride[0]*inner_stride[0]*x-inner_stride[0]*pad[0]-inner_pad[0]
                    inner_index_y = u+stride[1]*inner_stride[1]*y-inner_stride[1]*pad[2]-inner_pad[2]
                    if 0<=inner_index_x<inner_shape[0] and 0<=inner_index_y<inner_shape[1]:
                        for p in range(A.shape[3]):
                            for q in range(A.shape[4]):
                                if 0<=t-inner_stride[0]*p<alpha_u.shape[0] and 0<=u-inner_stride[1]*q<alpha_u.shape[1] and 0<=p+stride[0]*x-pad[0]<alpha_u.shape[2] and 0<=q+stride[1]*y-pad[2]<alpha_u.shape[3]:
                                    A_new[x,y,:,t,u,:] += A_plus[x,y,:,p,q,:]*alpha_u[t-inner_stride[0]*p,u-inner_stride[1]*q,p+stride[0]*x-pad[0],q+stride[1]*y-pad[2],:]
                                    A_new[x,y,:,t,u,:] += A_minus[x,y,:,p,q,:]*alpha_l[t-inner_stride[0]*p,u-inner_stride[1]*q,p+stride[0]*x-pad[0],q+stride[1]*y-pad[2],:]
    B_new = conv_full(A_plus,beta_u,pad,stride) + conv_full(A_minus,beta_l,pad,stride)+B
    return A_new, B_new

@njit
def lower_bound_pool(A, B, pad, stride, pool_size, inner_pad, inner_stride, inner_shape, LB, UB):
    A_new = np.zeros((A.shape[0], A.shape[1], A.shape[2], inner_stride[0]*(A.shape[3]-1)+pool_size[0], inner_stride[1]*(A.shape[4]-1)+pool_size[1], A.shape[5]), dtype=np.float32)
    B_new = np.zeros(B.shape, dtype=np.float32)
    A_plus = np.maximum(A, 0)
    A_minus = np.minimum(A, 0)
    alpha_u, alpha_l, beta_u, beta_l = pool_linear_bounds(LB, UB, inner_pad, inner_stride, pool_size)

    for x in range(A_new.shape[0]):
        for y in range(A_new.shape[1]):
            for t in range(A_new.shape[3]):
                for u in range(A_new.shape[4]):
                    inner_index_x = t+stride[0]*inner_stride[0]*x-inner_stride[0]*pad[0]-inner_pad[0]
                    inner_index_y = u+stride[1]*inner_stride[1]*y-inner_stride[1]*pad[2]-inner_pad[2]
                    if 0<=inner_index_x<inner_shape[0] and 0<=inner_index_y<inner_shape[1]:
                        for p in range(A.shape[3]):
                            for q in range(A.shape[4]):
                                if 0<=t-inner_stride[0]*p<alpha_u.shape[0] and 0<=u-inner_stride[1]*q<alpha_u.shape[1] and 0<=p+stride[0]*x-pad[0]<alpha_u.shape[2] and 0<=q+stride[1]*y-pad[2]<alpha_u.shape[3]:
                                    A_new[x,y,:,t,u,:] += A_plus[x,y,:,p,q,:]*alpha_l[t-inner_stride[0]*p,u-inner_stride[1]*q,p+stride[0]*x-pad[0],q+stride[1]*y-pad[2],:]
                                    A_new[x,y,:,t,u,:] += A_minus[x,y,:,p,q,:]*alpha_u[t-inner_stride[0]*p,u-inner_stride[1]*q,p+stride[0]*x-pad[0],q+stride[1]*y-pad[2],:]
    B_new = conv_full(A_plus,beta_l,pad,stride) + conv_full(A_minus,beta_u,pad,stride)+B
    return A_new, B_new

@njit
def compute_bounds(weights, biases, out_shape, nlayer, x0, eps, p_n, strides, pads, LBs, UBs, method):
    print('nlayer: ', nlayer)
    pad = (0,0,0,0)
    stride = (1,1)
    modified_LBs = LBs + (np.ones(out_shape, dtype=np.float32),)
    modified_UBs = UBs + (np.ones(out_shape, dtype=np.float32),)
    for i in range(nlayer-1, -1, -1):
        if not np.isnan(weights[i]).any(): #Conv
            if i == nlayer-1:
                A_u = weights[i].reshape((1, 1, weights[i].shape[0], weights[i].shape[1], weights[i].shape[2], weights[i].shape[3]))*np.ones((out_shape[0], out_shape[1], weights[i].shape[0], weights[i].shape[1], weights[i].shape[2], weights[i].shape[3]), dtype=np.float32)
                B_u = biases[i]*np.ones((out_shape[0], out_shape[1], out_shape[2]), dtype=np.float32)
                A_l = A_u.copy()
                B_l = B_u.copy()
            else:
                A_u, B_u = upper_bound_conv(A_u, B_u, pad, stride, weights[i], biases[i], pads[i], strides[i], modified_UBs[i].shape, modified_LBs[i+1], modified_UBs[i+1], method)
                A_l, B_l = lower_bound_conv(A_l, B_l, pad, stride, weights[i], biases[i], pads[i], strides[i], modified_LBs[i].shape, modified_LBs[i+1], modified_UBs[i+1], method) 
        else: #Pool
            if i == nlayer-1:
                A_u = np.eye(out_shape[2]).astype(np.float32).reshape((1,1,out_shape[2],1,1,out_shape[2]))*np.ones((out_shape[0], out_shape[1], out_shape[2], 1,1,out_shape[2]), dtype=np.float32)
                B_u = np.zeros(out_shape, dtype=np.float32)
                A_l = A_u.copy()
                B_l = B_u.copy()
            A_u, B_u = upper_bound_pool(A_u, B_u, pad, stride, weights[i].shape[1:], pads[i], strides[i], modified_UBs[i].shape, np.maximum(modified_LBs[i],0), np.maximum(modified_UBs[i],0))
            A_l, B_l = lower_bound_pool(A_l, B_l, pad, stride, weights[i].shape[1:], pads[i], strides[i], modified_LBs[i].shape, np.maximum(modified_LBs[i],0), np.maximum(modified_UBs[i],0))
        pad = (strides[i][0]*pad[0]+pads[i][0], strides[i][0]*pad[1]+pads[i][1], strides[i][1]*pad[2]+pads[i][2], strides[i][1]*pad[3]+pads[i][3])
        stride = (strides[i][0]*stride[0], strides[i][1]*stride[1])
    LUB, UUB = conv_bound_full(A_u, B_u, pad, stride, x0, eps, p_n)
    LLB, ULB = conv_bound_full(A_l, B_l, pad, stride, x0, eps, p_n)
    return LLB, ULB, LUB, UUB, A_u, A_l, B_u, B_l, pad, stride

def find_output_bounds(weights, biases, shapes, pads, strides, x0, eps, p_n, method='NeWise'):
    LB, UB = conv_bound(weights[0], biases[0], pads[0], strides[0], x0, eps, p_n)
    LBs = [x0-eps, LB]
    UBs = [x0+eps, UB]
    
    for i in range(2,len(weights)+1):
        print('find_output_bounds ', i)
        LB, _, _, UB, A_u, A_l, B_u, B_l, pad, stride = compute_bounds(tuple(weights), tuple(biases), shapes[i], i, x0, eps, p_n, tuple(strides), tuple(pads), tuple(LBs), tuple(UBs), method)
        UBs.append(UB)
        LBs.append(LB)
    # return LBs, UBs, A_u, A_l, B_u, B_l, pad, stride
    return LBs[-1], UBs[-1], A_u, A_l, B_u, B_l, pad, stride

def warmup(model, x, eps_0, p_n, fn):
    print('Warming up...')
    weights = model.weights[:-1]
    biases = model.biases[:-1]
    shapes = model.shapes[:-1]
    W, b, s = model.weights[-1], model.biases[-1], model.shapes[-1]
    last_weight = np.ascontiguousarray((W[0,:,:,:]).reshape([1]+list(W.shape[1:])),dtype=np.float32)
    weights.append(last_weight)
    biases.append(np.asarray([b[0]]))
    shapes.append((1,1,1))
    print('enter fn...')
    fn(weights, biases, shapes, model.pads, model.strides, x, eps_0, p_n)
    
ts = time.time()
timestr = datetime.datetime.fromtimestamp(ts).strftime('%Y%m%d_%H%M%S')
#Prints to log file
def printlog(s):
    print(s, file=open("logs/cnn_bounds_full_core_with_LP_"+timestr+".txt", "a"))

def run_certified_bounds_core(file_name, n_samples, p_n, q_n, data_from_local=True, method='NeWise', cnn_cert_model=False, activation = 'sigmoid', mnist=False, cifar=False, fashion_mnist=False, gtsrb=False):
    np.random.seed(1215)
    random.seed(1215)
    if activation == 'atan':
        keras_model = load_model(file_name, custom_objects={'atan': tf.atan})
    else:
        keras_model = load_model(file_name, custom_objects={'fn':fn, 'tf':tf})

    if cifar:
        model = CNNModel(keras_model, inp_shape = (32,32,3))
    elif gtsrb:
        print('gtsrb')
        model = CNNModel(keras_model, inp_shape = (48,48,3))
    else:
        model = CNNModel(keras_model)
    print('--------abstracted model-----------')
    
    global linear_bounds
    if activation == 'sigmoid':
        linear_bounds = sigmoid_linear_bounds
    elif activation == 'tanh':
        linear_bounds = tanh_linear_bounds
    elif activation == 'atan':
        linear_bounds = atan_linear_bounds
    
    upper_bound_conv.recompile()
    lower_bound_conv.recompile()
    compute_bounds.recompile()

    dataset = ''
    
    if cifar:
        dataset = 'cifar10'
        inputs, targets, true_labels, true_ids, img_info = generate_data('cifar10', samples=n_samples, data_from_local=data_from_local, targeted=True, random_and_least_likely = True, target_type = 0b0010, predictor=model.model.predict, start=0, cnn_cert_model=cnn_cert_model)
    elif fashion_mnist:
        dataset = 'fashion_mnist'
        inputs, targets, true_labels, true_ids, img_info = generate_data('fashion_mnist', samples=n_samples, data_from_local=data_from_local, targeted=True, random_and_least_likely = True, target_type = 0b0010, predictor=model.model.predict, start=0, cnn_cert_model=cnn_cert_model)
    else:
        dataset = 'mnist'
        inputs, targets, true_labels, true_ids, img_info = generate_data('mnist', samples=n_samples, data_from_local=data_from_local, targeted=True, random_and_least_likely = True, target_type = 0b0010, predictor=model.model.predict, start=0, cnn_cert_model=cnn_cert_model)
        
    #0b01111 <- all
    #0b0010 <- random
    #0b0001 <- top2 
    #0b0100 <- least
        
    print('----------generated data---------')

    printlog('===========================================')
    printlog("model name = {}".format(file_name))
    
    total_images = 0
    steps = 15
    eps_0 = 0.05
    summation = 0
    
    warmup(model, inputs[0].astype(np.float32), eps_0, p_n, find_output_bounds)
    
    NeWise_start_time = time.time()
    for i in range(len(inputs)):
        # printlog('--- ' + method + ' relaxation: Computing eps for input image ' + str(i)+ '---')
        predict_label = np.argmax(true_labels[i])
        target_label = np.argmax(targets[i])
        
        #Perform binary search
        log_eps = np.log(eps_0)
        log_eps_min = -np.inf
        log_eps_max = np.inf
        for j in range(steps):
            LB_total, UB_total, _, _, _, _, _, _ = find_output_bounds(model.weights, model.biases, model.shapes, model.pads, model.strides, inputs[i].astype(np.float32), np.exp(log_eps), p_n, method)
            distance_bt_pre_tar = LB_total[0][0][predict_label] - UB_total[0][0][target_label]
            # print("Step {}, eps = {:.5f}, f_c_min - f_t_max = {:.6s}".format(j,np.exp(log_eps),str(distance_bt_pre_tar)))
            
            # print("Step {}, eps = {:.5f}, {:.6s} <= f_c - f_t <= {:.6s}".format(j,np.exp(log_eps),str(np.squeeze(LB)),str(np.squeeze(UB))))
            if distance_bt_pre_tar > 0: #Increase eps
                log_eps_min = log_eps
                log_eps = np.minimum(log_eps+1, (log_eps_max+log_eps_min)/2)
            else: #Decrease eps
                log_eps_max = log_eps
                log_eps = np.maximum(log_eps-1, (log_eps_max+log_eps_min)/2)
        
        # printlog("[L1] method = {}-{}, model = {}, image no = {}, true_id = {}, target_label = {}, true_label = {}, robustness = {:.5f}".format(method, activation,file_name, i, true_ids[i],target_label,predict_label,np.exp(log_eps_min)))
        summation += np.exp(log_eps_min)
    
    eps_avg = summation/len(inputs)
    aver_time = (time.time()-NeWise_start_time)/len(inputs)
    # printlog("[L0] method = {}-{}, model = {}, total images = {}, avg robustness = {:.5f}, avg runtime = {:.2f}".format(method, activation,file_name,len(inputs),eps_avg,aver_time))
    printlog("[L0] method = {}-{}, total images = {}, avg robustness = {:.5f}, avg runtime = {:.2f}".format(method, activation, len(inputs),eps_avg,aver_time))

def run_verified_robustness_core(file_name, n_samples, eps_0, p_n, q_n, method='NeWise', cnn_cert_model=False, activation = 'sigmoid', mnist=False, cifar=False, fashion_mnist=False, gtsrb=False):
    np.random.seed(1215)
    random.seed(1215)
    keras_model = load_model(file_name, custom_objects={'fn':fn, 'tf':tf})

    if cifar:
        model = CNNModel(keras_model, inp_shape = (32,32,3))
    elif gtsrb:
        print('gtsrb')
        model = CNNModel(keras_model, inp_shape = (48,48,3))
    else:
        model = CNNModel(keras_model)
    print('--------abstracted model-----------')
    
    global linear_bounds
    if activation == 'sigmoid':
        linear_bounds = sigmoid_linear_bounds
    elif activation == 'tanh':
        linear_bounds = tanh_linear_bounds
    
    upper_bound_conv.recompile()
    lower_bound_conv.recompile()
    compute_bounds.recompile()

    dataset = ''
    
    if cifar:
        dataset = 'cifar10'
        inputs, targets, true_labels, true_ids = generate_data_myself('cifar10', model.model, samples=n_samples, start=0)
    elif gtsrb:
        dataset = 'gtsrb'
        inputs, targets, true_labels, true_ids = generate_data_myself('gtsrb', model.model, samples=n_samples, start=0)
    elif fashion_mnist:
        dataset = 'fashion_mnist'
        inputs, targets, true_labels, true_ids = generate_data_myself('fashion_mnist', model.model, samples=n_samples, start=0)
    else:
        dataset = 'mnist'
        inputs, targets, true_labels, true_ids = generate_data_myself('mnist', model.model, samples=n_samples, start=0, cnn_cert_model=cnn_cert_model)
        
    print('----------generated data---------')

    printlog('===========================================')
    printlog("model name = {}".format(file_name))
    printlog("eps = {:.5f}".format(eps_0))
    
    warmup(model, inputs[0].astype(np.float32), eps_0, p_n, find_output_bounds)
    
    NeWise_start_time = time.time()
    NeWise_robust_number = 0
    NeWise_unknown_number = 0 
    NeWise_robust_img_id = []
    total_images = 0

    for i in range(len(inputs)):
        total_images += 1
        
        # printlog("----------------image id = {}----------------".format(i))
        print('------------image id = {}--------------'.format(i))
        predict_label = np.argmax(true_labels[i])
        # printlog("image predict label = {}".format(predict_label))
        
        NeWise_robust_flag = True
        each_image_start_time = time.time()
        
        LB_total, UB_total, _, _, _, _, _, _ = find_output_bounds(model.weights, model.biases, model.shapes, model.pads, model.strides, inputs[i].astype(np.float32), eps_0, p_n, method)
        # printlog('{} relaxation'.format(method))
        
        for j in range(i*9,i*9+9):
            target_label = targets[j]
            # printlog("target label = {}".format(target_label))
            
            # printlog('LB_total[0, 0, predict_label] - UB_total[0, 0, target_label]: {}'.format(LB_total[0, 0, predict_label] - UB_total[0, 0, target_label]))
            if LB_total[0, 0, predict_label] - UB_total[0, 0, target_label] > 0:
                continue
            
            weights = model.weights[:-1]
            biases = model.biases[:-1]
            shapes = model.shapes[:-1]
            W, b, s = model.weights[-1], model.biases[-1], model.shapes[-1]
            last_weight = (W[predict_label,:,:,:]-W[target_label,:,:,:]).reshape([1]+list(W.shape[1:]))
            weights.append(last_weight)
            biases.append(np.asarray([b[predict_label]-b[target_label]]))
            shapes.append((1,1,1))

            LB, UB, A_u, A_l, B_u, B_l, pad, stride = find_output_bounds(weights, biases, shapes, model.pads, model.strides, inputs[i].astype(np.float32), eps_0, p_n, method)
            # printlog('the interval of predict_label - target_label')
            # printlog('[ {}, {} ]'.format(LB, UB))
            
            if LB > 0:
                continue 
            
            NeWise_robust_flag = False
            break
        
        if NeWise_robust_flag:
            NeWise_robust_number += 1
            NeWise_robust_img_id.append(i)
            # printlog("{}: robust".format(method))
        else:
            # printlog("{}: unknown".format(method))
            NeWise_unknown_number += 1
            
        # printlog("runtime: {:.3f}".format(time.time()-each_image_start_time))
    
    NeWise_total_time = (time.time()-NeWise_start_time)
    printlog("{} time: {:.5f}".format(method, NeWise_total_time))
    NeWise_aver_time = NeWise_total_time / total_images
    printlog("[L0] method = {}, eps = {}, total images = {}, robust = {}, unknown = {}, average runtime = {:.3f}".format(method, eps_0, total_images, NeWise_robust_number, NeWise_unknown_number, NeWise_aver_time))
    # printlog("[L0] {} robust images id: {}".format(method, NeWise_robust_img_id))
   
    print('------------------')
    print('------------------')