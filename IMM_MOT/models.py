#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  5 14:37:27 2025

@author: tanvibansal
"""
import numpy as np
from scipy.stats import norm

class KalmanFilter:
    """
    F: dynamics prediction matrix
        F @ [x1, v1, a1] -> [x2, v2, a2]
        3x3: 
            x = x + v*t + a*t^2 / 2
            v =     v +   a*t
            a =           a
    H: state estimation conversion matrix
        H @ [x, v, a] -> [x]

    predict(u | z, P; F, B, Q):
        z = Fz + Bu
        P = FPF^T + Q

    update(x | z, P; H, R):
        y = x - Hz
        S = HPH^T + R

        K = PH^TS^-1
        J = I - KH

        z = z + Ky
        P = JPJ^T + KRK^T

    """
    def __init__(self, x=0, B=None, H=None, Q=None, R=None, z=None, P=None, dt=None, max_age=None,uuid=None):
        self.dt = dt or 1  # default time step
        self.max_age = max_age  # maximum age of the filter between measurement and predict time in seconds
        self._update_F(self.dt)  # transition matrix
        self.uuid = np.random.randint(100000,1000000) if uuid is None else uuid
        # constants
        self.n = self.F.shape[1]  # state dimension
        self.I = np.eye(self.n)  # identity matrix

        # parameters
        self.H = np.eye(self.n)[:1] if H is None else H  # measurement function (state to measurement space)
        self.B = 0 if B is None else B  # control input
        self.Q = np.eye(self.n) if Q is None else Q  # process noise
        self.R = np.eye(self.n) if R is None else R  # measurement noise

        # state
        self.z = np.zeros((self.n, 1)) if z is None else z  # state estimate
        self.P = np.eye(self.n) if P is None else P  # state covariance
        self.tx = 0  # last measurement time
        self.tp = 0  # last prediction time

        # initialize filter position
        if x is not None:
            self.set_x(x)

    def _update_F(self, dt):
        '''Set state transition matrix using time delta'''
        # x = x + v*dt + a*dt^2 / 2
        # v =     v +   a*dt
        self.F = np.array([
            [1, dt, 0.5 * dt**2], 
            [0,  1,       dt], 
            [0,  0,        1]
        ])

    def _update_time(self, t):
        '''set prediction time and update transition matrix using time difference'''
        dt = t - self.tp if self.tp is not None else 1
        self.tp = t  # update prediction time
        self._update_F(dt)

    def predict(self, t, u=0):
        self._update_time(t)  # update timestamp and transition matrix

        self.z = self.F @ self.z + self.B * u  # state prediction  [n, 1]
        self.P = self.F @ self.P @ self.F.T + self.Q  # covariance prediction  [n, n]
        x_h = self.H @ self.z  # measurement prediction  [1]
        return x_h

    def estimate(self, x):
        y = x - self.H @ self.z  # measurement residual  [1]
        S = self.H @ self.P @ self.H.T + self.R  # residual covariance  [1, 1]
        return y, S

    def apply(self, y, S, x=None):
        self.tx = self.tp  # update measurement time to latest prediction time
        
        K = self.P @ self.H.T @ np.linalg.inv(S)  # Kalman gain  [n, 1]
        J = self.I - K @ self.H  # [n, n]
        self.z = self.z + K @ y  # state update  [n, 1] + [n, 1] @ [1]
        self.P = J @ self.P @ J.T + K @ self.R @ K.T  # covariance update  [n, n] + [n, n]

        # set the filter position to the ground truth value to avoid drift
        if x is not None:
            self.set_x(x)
        return self.z, self.P

    def cost(self, y, S):
        '''Get how likely the measurement is given the state'''
    
        c = np.abs(y)/ np.log(np.diag(np.abs(S)))
        return c
    
    def expired(self):
        '''Check if the filter hasn't been updated for a while'''
        return self.max_age and self.tx is not None and self.tp - self.tx > self.max_age

    def set_x(self, x):
        self.z[0] = x
    
    def pdf(self, x):
        rv = norm(loc = self.H @ self.z, scale = self.R)
        return rv.pdf(x)


class IMM:
    def __init__(self, filters, new_filter):
        self.filters = filters
        self.new_filter = new_filter

    @property
    def n(self):
        return len(self.filters)

    def predict(self, u=0):
        x_hs = np.zeros((self.n, 1, 1))
        for i in range(self.n):
            x_hs[i] = self.filters[i].predict(u)
        return x_hs

    def estimate(self, x):
        ys = np.zeros((self.n, 1, 1))
        Ss = np.zeros((self.n, 1, 1))
        for i in range(self.n):
            y, S = self.filters[i].estimate(x)
            ys[i], Ss[i] = y, S

        return ys, Ss

    def cost(self, ys, Ss):
        cost = np.zeros((self.n, 1, 3))
        for i in range(self.n):
            cost[i] = self.filters[i].cost(ys[i], Ss[i])
            #if Ss[i] > 10000000000:
             #   cost[i] = np.inf
        return cost

    def apply(self, i, y, S, x):
        self.filters[i].apply(y, S, x)

        # drop stale filters
        self.filters = [f for f in self.filters if not f.expired()]

    def missed(self, x, t):
        # add new filter
        f = self.new_filter(x)
        self.filters.append(f)
        # warmup
        f.predict(t)
        y, S = f.estimate(x)
        f.apply(y, S)
        return y, S


def MOT(base_track_filters, noise_track_Q, noise_track_R, track_expiration_thresh, measurements, gating_thresh):
    """
    MOT implementation for IMM prediction with 3-dimension state and 1-dimension observation 
    
    Args:
    base_track_filters: list of Kalman filters to be used in the IMM for predicting flood profiles
    measurements: pandas dataframe of continuous timeseries signals with columns: time (sorted ascending),
        t_elapsed (time - time[0]), dt (difference between t_elapsed in seconds)
    noise_track_Q: numpy array of shape (3,3) with noise event process covariance
    noise_track_R: numpy array of shape (1,1) with noise event measurement noise covariance
    track_expiration_thresh: float encoding expiration threshold to retire inactive tracks (3/60 = 3 measurements) 
    gating_thresh: float or int encoding the gating threshold for the prediction cost
    """
    # create IMM with filters for base track (rest/flood)    
    im = IMM(base_track_filters, (lambda x: KalmanFilter(x,Q=noise_track_Q,R = noise_track_R, max_age=track_expiration_thresh)))  # noise filters with different noise parameters
    
    #run the model over each measurement in the timeseries    
    predictions = [[0,0]]
    uncertainties = [[0,0]]
    costs = []
    filters = [[0,0]]
    best_filters = [0]
    for i in range(1,len(measurements)):
        timestamp = measurements.t_elapsed.iloc[i]/60
        x = measurements.depth_raw_mm.iloc[i]
        # PREDICT next location for all active filters
        x_h = im.predict(timestamp)
        x_h = x_h[:, 0, 0]
    
        # EVALUATE prediction error
        y, S = im.estimate(x)
        S = np.abs(S)
        U = 3 * np.sqrt(S[:, 0,0])
        cost = im.cost(y, S)
        f = np.argmin(cost.mean((1,2)))  # get the best filter

        costs.append(cost[:,:,0])
    
        # DATA ASSOCIATION 
        if x <=30: #check if the measurement is within the expected region for measurement noise of the base track (30mm)
            #apply rest filter at x =x
            for f in [0,1]:
                im.apply(f, y[f], S[f], x)
                x_h[f] = x
            best_filters.append(0)
        elif (cost[f] < gating_thresh).all() or (0 <= x < x_h[f]): #if not, find if any active tracks are feasible and select the most likely one
            best_filters.append(im.filters[f].uuid)
            im.apply(f, y[f], S[f], x)
            x_h[f] = x
        else:  # if no active tracks are likely, assume new object has entered and create a new filter/track           
            #print("new filter spawned: t = %s"%(timestamp))
            y, S = im.missed(x, timestamp)
    
            # append new filter prediction
            x_h = np.concatenate([x_h, [x]])
            U = np.concatenate([U, [3 * np.sqrt(S[0,0])]])
            best_filters.append(im.filters[-1].uuid)
    
        # SAVE RESULTS 
        predictions.append(x_h)
        uncertainties.append(U)
        filters.append([f.uuid for f in im.filters])
        #best_filters.append(bf)
    costs.append(cost[:,:,0])
    
            
    return predictions, uncertainties, filters, costs #, best_filters
       