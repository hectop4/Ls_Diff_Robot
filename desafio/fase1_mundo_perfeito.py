import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({
    "font.size": 32,
    "axes.titlesize": 26,
    "axes.labelsize": 24,
    "xtick.labelsize": 32,
    "ytick.labelsize": 32,
    "legend.fontsize": 20,
})

r = 0.25
s = 1.0

def phi_l_dot(t):
  return 8.0
def phi_r_dot(t):
  return 16.0

phi_l_dot = phi_l_dot(0)
phi_r_dot = phi_r_dot(0)

v= r / 2 * (phi_l_dot + phi_r_dot)
omega = r / s * (phi_r_dot - phi_l_dot)

def theta(t, theta_i):
  return 2 * t + theta_i

def x_t(t, x_i, theta_i):
  return (v/omega) * (np.sin(2 * t + theta_i) - np.sin(theta_i)) 

def y_t(t, y_i, theta_i):
  return (-v/omega) * (np.cos(2 * t + theta_i) - np.cos(theta_i)) + y_i 

def observations(t, x_i, y_i, theta_i, obs):
  x = x_t(t, x_i, theta_i)
  y = y_t(t, y_i, theta_i)
  return np.sqrt((x - obs[:, 0]) ** 2 + (y - obs[:, 1]) ** 2)

def A(x, y, theta):
  return np.array([[0,0,-f1.v*np.sin(theta)],[0,0,f1.v*np.cos(theta)],[0,0,0]])

x_vect_true = np.array([3,3,0.5])

t_sim=np.arange(0, 10,0.01)

x=np.array([[x_t(t, x_vect_true[0], x_vect_true[2]), y_t(t, x_vect_true[1], x_vect_true[2]), theta(t, x_vect_true[2])] for t in t_sim])
t_obs = np.arange(0,10,2)
rho_beacons = np.array([[0, 0], [10, 0], [0, 10]])

real_observations = np.array([observations(t, x_vect_true[0], x_vect_true[1], x_vect_true[2], rho_beacons) for t in t_obs])
#! Noising

noisy_observations = real_observations + np.random.normal(0, 0.05, real_observations.shape)


if __name__ == "__main__":
    print("This module defines the estimated world model and generates observations.")
    print("=============State Vector===================")
    print(x)


    print("=============Real Observations===================")
    print(real_observations)  


    print("=============Noisy Observations===================")
    print(noisy_observations)

    plt.figure()
    plt.plot(t_obs, real_observations[:, 0], 'o-', label='Beacon 1 (Real)')
    plt.plot(t_obs, real_observations[:, 1], 'o-', label='Beacon 2 (Real)')
    plt.plot(t_obs, real_observations[:, 2], 'o-', label='Beacon 3 (Real)')
    plt.plot(t_obs, noisy_observations[:, 0], 'x--', label='Beacon 1 (Noisy)')
    plt.plot(t_obs, noisy_observations[:, 1], 'x--', label='Beacon 2 (Noisy)')
    plt.plot(t_obs, noisy_observations[:, 2], 'x--', label='Beacon 3 (Noisy)')

    plt.xlabel('Time [s]')
    plt.ylabel('Distance')
    plt.title('Observations over Time')
    plt.legend()
    plt.grid()
    plt.show()  

    plt.figure()
    plt.plot(t_sim, x[:, 0], label='x(t)')
    plt.plot(t_sim, x[:, 1], label='y(t)')
    plt.plot(t_sim, x[:, 2], label='theta(t)')
    plt.xlabel('Time [s]')
    plt.ylabel('State Variables')
    plt.title('State Variables over Time')
    plt.legend()
    plt.grid()
    plt.show()