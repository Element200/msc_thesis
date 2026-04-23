# -*- coding: utf-8 -*-
"""
simulation_methods.py.

Created on Wed Jun 11 14:42:20 2025
v1.3

@author: sane
Contains methods to integrate the Langevin equation, effectively simulating what the setup defined in Kumar and Bechhoefer (2020) does
"""

import numpy as np
from tqdm import tqdm
import pandas as pd
import xarray as xr
import scipy
import sympy as sym

import sys
import os
directory = os.path.dirname(__file__)
sys.path.append(directory)
try:
    import quench_methods
except ImportError:
    print("Something went wrong when importing dependencies! Make sure local imports are in the same directory as this file")
    raise ImportError


dt = 1e-5
expt_length = 6e-2
F = lambda x: -x
x_min = -1
x_max = 3
D = 150 # px^2/s
k_BT_b = 1
gamma = k_BT_b/D #0.02
E_barrier,E_tilt, F_left, x_well, force_asymmetry = 2,1.3,50,0.5,1 # for testing purposes

def fast_CDF_inverter(f, n_x=512):
    x = np.linspace(0,1, n_x)
    f_vals = f(x)
    f_inv = scipy.interpolate.interp1d(f_vals, x) # Swap y argument and x argument to get an approximate inverse function
    return f_inv
    

def inverse_transform_sampler(num_samples, CDF, interpolation_mesh_size=100, n_x=512):
    uniform_samples = np.random.random(num_samples) # Drawn from U[0,1]
    CDF_inv = fast_CDF_inverter(CDF, n_x=n_x)
    return CDF_inv(uniform_samples)
    

def langevin_simulation(x_0, dt=dt, gamma=1/150, expt_length = expt_length, force = F, temperature_function = lambda t: 1, k_BT_b=1, measurement_noise_std=0, clip_quench=False, max_force=np.inf):
    """
    Integrate (the Ito way) the SDE dx = F(x)/gamma * dt + D(x,t)*eta(t), where eta is a normally distributed random number, from t=0 to t=expt_length.

    Parameters
    ----------
    x_0 : numeric or vector of numerics
        Initial position(s) (at t=0). The shape will determine the number of output trajectories.
    dt : numeric, optional
        Time interval. The default is 1e-5.
    gamma : numeric, optional
        Viscosity. The default is roughly 5.88e-3.
    expt_length : numeric, optional
        Time interval to integrate over. The default is 6e-2.
    force : vectorised function, optional
        Vectorised function, deterministic term. The default is F(x)=-x.
    temperature_function : function, optional
        For time-varying quenches, the dimensionless function (scaled by k_BT_b) describing how the bath temperature evolves. The default is lambda t: 1, which is an instantaneous quench.
    k_BT_b : numeric, optional
        Equilibrium temperature. The default is 1.
    measurement_noise_std : numeric, optional
        Measurement noise standard deviation. The default is 0.

    Returns
    -------
    x : 2D vector of shape (expt_length//dt, *x_0.shape). 
        If x_0 is 1D, this is a 2D array with the first axis corresponding to the timestep and the second axis corresponding to the 'particle number' in the trajectory. If x_0 is a single numeric, x will only have one useful dimension corresponding to the timestep. x is a vector of all of the trajectories we get from stochastically integrating the Langevin equation, and forms an ensemble that we can plug into the Ensemble object defined earlier.

    """
    max_displacement = max_force/gamma
    # if type(x_0) != np.array:
    #     x_0 = np.array([x_0])
    num_particles = x_0.shape[0]
    timesteps = round(expt_length/dt)
    initial_measurement_noise = np.random.normal(0, measurement_noise_std, num_particles)
    x_i = x_0 # "true" position
    X_i = x_0.copy()+initial_measurement_noise # Measured position (which has measurement nosie)
    X = np.zeros((*x_0.shape, timesteps)) # Preallocating space and mutating the array is more efficient than appending data
    X[...,0] = X_i.copy() # We can only measure the measured position, not the true position (you may have guessed this from the name)
    t = np.arange(0,expt_length, dt)
    D = k_BT_b*temperature_function(t)/gamma
    thermal_fluctuation_std = np.sqrt(2*D*dt)
    stochastic_displacement = np.random.normal(0,thermal_fluctuation_std, size=(*x_0.shape, timesteps)) # Precalculate the noise terms we will use in the integral -- speeds up code. One array of noise (same shape as x_0) is needed per timestep. 
    measurement_noise = np.random.normal(0, measurement_noise_std, size=stochastic_displacement.shape) 
    for i in tqdm(range(1, timesteps)): # tqdm for progress bar
        # We try to minimise indexing within the hot loop
        deterministic_displacement = force(x_i, i*dt)*dt/gamma # Since we're using feedback forces, the force depends on the *measured* position, which includes measurement noise, not the true position
        x_i = x_i + (deterministic_displacement + stochastic_displacement[..., i])
        X_i = x_i + measurement_noise[...,i]
        X[...,i] = X_i # 'Append' to the preallocated array
        # Appending to an array is like measurement, and the noise is added in this step.
    return X

def langevin_simulation_virtualPotential(x_0, dt=dt, gamma=1/150, expt_length = expt_length, force = F, temperature_function = lambda t: 1, k_BT_b=1, measurement_noise_std=0):
    """
    Integrate (the Ito way) the SDE dx = F(x)/gamma * dt + D(x,t)*eta(t), where eta is a normally distributed random number, from t=0 to t=expt_length.

    Parameters
    ----------
    x_0 : numeric or vector of numerics
        Initial position(s) (at t=0). The shape will determine the number of output trajectories.
    dt : numeric, optional
        Time interval. The default is 1e-5.
    gamma : numeric, optional
        Viscosity. The default is roughly 5.88e-3.
    expt_length : numeric, optional
        Time interval to integrate over. The default is 6e-2.
    force : vectorised function, optional
        Vectorised function, deterministic term. The default is F(x)=-x.
    temperature_function : function, optional
        For time-varying quenches, the dimensionless function (scaled by k_BT_b) describing how the bath temperature evolves. The default is lambda t: 1, which is an instantaneous quench.
    k_BT_b : numeric, optional
        Equilibrium temperature. The default is 1.
    measurement_noise_std : numeric, optional
        Measurement noise standard deviation. The default is 0.

    Returns
    -------
    x : 2D vector of shape (expt_length//dt, *x_0.shape). 
        If x_0 is 1D, this is a 2D array with the first axis corresponding to the timestep and the second axis corresponding to the 'particle number' in the trajectory. If x_0 is a single numeric, x will only have one useful dimension corresponding to the timestep. x is a vector of all of the trajectories we get from stochastically integrating the Langevin equation, and forms an ensemble that we can plug into the Ensemble object defined earlier.

    """
    # if type(x_0) != np.array:
    #     x_0 = np.array([x_0])
    num_particles = x_0.shape[0]
    timesteps = round(expt_length/dt)
    initial_measurement_noise = np.random.normal(0, measurement_noise_std, num_particles)
    x_i = x_0 # "true" position
    X_i = x_0.copy()+initial_measurement_noise # Measured position (which has measurement nosie)
    X_i_ = X_i.copy() # Lagging "true" position
    X = np.zeros((*x_0.shape, timesteps)) # Preallocating space and mutating the array is more efficient than appending data
    X[...,0] = X_i.copy() # We can only measure the measured position, not the true position (you may have guessed this from the name)
    t = np.arange(0,expt_length, dt)
    D = k_BT_b*temperature_function(t)/gamma
    thermal_fluctuation_std = np.sqrt(2*D*dt)
    stochastic_displacement = np.random.normal(0,thermal_fluctuation_std, size=(*x_0.shape, timesteps)) # Precalculate the noise terms we will use in the integral -- speeds up code. One array of noise (same shape as x_0) is needed per timestep. 
    measurement_noise = np.random.normal(0, measurement_noise_std, size=stochastic_displacement.shape) 
    for i in tqdm(range(1, timesteps)): # tqdm for progress bar
        # We try to minimise indexing within the hot loop
        deterministic_displacement = (i>1)*force(X_i_, i*dt)*dt/gamma # The force is zero at t=1 since the force is evaluated at the n-2'th timestep
        X_i_ = X_i.copy()
        x_i = x_i + (deterministic_displacement + stochastic_displacement[..., i])
        X_i = x_i + measurement_noise[...,i]
        X[...,i] = X_i # 'Append' to the preallocated array
        # Appending to an array is like measurement, and the noise is added in this step.
    return X

def run_mpemba_simulations(k_BTs, num_particles, potential, quench_protocol = None, num_allowed_initial_positions=100_000, dt=1e-5, expt_length=1e-1, save_memory=False, gamma=gamma, transformed_time=False, k_BT_b=1, measurement_noise_std=0, initial_position_tolerance=0, use_virtual_potential=True):
    """
    Simulate the experiment in Bechhoefer and Kumar (2020) by choosing an initial position from a Boltzmann distribution and integrating the Langevin equation that it corresponds to.

    Parameters
    ----------
    k_BTs : Vector of numerics. At least one of these should be 1.
    
    
    num_particles : int
        Number of particles in the ensemble (preferably >1000).
    potential : Potential object
        An object that is equipped with methods to calculate the potential, forces, etc.
    quench_protocol : function, optional
        Quench function to use. The default is lambda k_BT: k_BT.
    num_allowed_initial_positions : int, optional
        The size of the array we'll draw initial positions from. Ideally this should be large. The default is 100_000.
    dt : numeric, optional
        Timestep for Ito integration. The default is 1e-5.
    expt_length : numeric, optional
        Interval to integrate over. The default is 1e-1.
    transformed_time : bool, optional
        Use a time transformation trick to avoid issues with large diffusivities in the quench protocol. The default is False.
    measurement_noise_std : numeric, optional
        The standard deviation of the measurement noise. The default is 0.
    initial_position_tolerance : numeric, optional
        In an experiment, the initial position will never be exactly equal to the position drawn from the PDF, but will instead be within some tolerance of the sampled position. This parameter accounts for that by adding a random Gaussian number with std=this value to the position. The default is 0.

    Returns
    -------
    xr.DataArray with coordinates T (temperatures), n (particle number), t (time)
        An ensemble containing the particle trajectories, and appropriately labelled. This can be passed into an Ensemble object later.

    """
    # potential_params ordering: [E_barrier, E_tilt, F_left, x_well, x_min,x_max,force_asymmetry]
    if use_virtual_potential:
        simulation_function = langevin_simulation_virtualPotential
    else:
        simulation_function = langevin_simulation
    if quench_protocol is None:
        quench_protocol = quench_methods.InstantaneousQuench()
    if not k_BT_b in k_BTs:
        raise ValueError("At least one reference temperature must exist!")
    active_range = np.linspace(potential.x_min, potential.x_max, num_allowed_initial_positions)
    results = []
    initial_distro = xr.DataArray(potential.boltzmann_PMF(active_range,k_BTs).T, coords=(k_BTs, active_range), dims=(['T','x'])) # Slow -- needs fixing
    # (len(k_BTs) x num_allowed_initial_positions) array to draw positions from
    
    times = np.arange(0,expt_length,dt)
    
    if transformed_time:
        t_list = []
        
    for i, k_BT in enumerate(k_BTs):
        
        quench_protocol.set_a(k_BT/k_BT_b-1)
        p_arr = initial_distro.loc[k_BT]
        initial_noise = np.random.uniform(-initial_position_tolerance, initial_position_tolerance, num_particles)
        
        if len(p_arr.shape)>1:
            p_arr = p_arr[0,:] # If there are multiple redundant temperatures, pick the first of the identical probability distros
        x = np.random.choice(active_range, num_particles, p=p_arr) # draw initial position from its probability distribution
        x += initial_noise # The true position always has measurement noise!
        if transformed_time:
            t_list.append(quench_protocol.t(times)) # Integrate time to find the transformation
            time_dependent_force = lambda x, t: potential.F(x)/quench_protocol.h_1(t)
            results.append(simulation_function(x, force= time_dependent_force, temperature_function=lambda t: k_BT_b, expt_length=expt_length, gamma=gamma, dt=dt, measurement_noise_std=measurement_noise_std)) # Transform the time using this trick and reset the quench protocol to be D(t) = k_BT_b, the instantaneous quench
        
        else:
            results.append(simulation_function(x, force=lambda x, t: potential.F(x), temperature_function=lambda t: quench_protocol.h(t), expt_length=expt_length, gamma=gamma, k_BT_b=k_BT_b, dt=dt, measurement_noise_std=measurement_noise_std))
    
    if save_memory:
        results = np.array(results, dtype=np.float32) # Change to single-precision floating point to save some memory
    if transformed_time:
        output = xr.concat([xr.DataArray(results[i], 
                                         coords=[('n', np.arange(0,num_particles,1)), 
                                                 ('t', t_list[i])]
                                         ) for i in range(len(k_BTs))], dim='T')
        output['T'] = k_BTs
        return output # This is going to have a lot of NaNs where the times don't line up
        
    return xr.DataArray(results, coords = [('T', k_BTs), ('n', np.arange(0,num_particles,1)),('t', times)])
