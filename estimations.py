import numpy as np
def theta(t, theta_i):
  return 2 * t + theta_i

def x_t(t, x_i, theta_i):
  return (3/2) * (np.sin(2 * t + theta_i) - np.sin(theta_i)) + x_i

def y_t(t, y_i, theta_i):
  return (-3/2) * (np.cos(2 * t + theta_i) - np.cos(theta_i)) + y_i

def observations(t, x_i, y_i, theta_i, obs):
  x = x_t(t, x_i, theta_i)
  y = y_t(t, y_i, theta_i)
  return np.sqrt((x - obs[:, 0]) ** 2 + (y - obs[:, 1]) ** 2)

def A(x, y, theta):
  return np.array([[0,0,-3*np.sin(theta)],[0,0,3*np.cos(theta)],[0,0,0]])
x_est = np.array([2.9,3.1,0.4])

print('============Measurements===================')
print('x_t:', [x_t(t, x_est[0], x_est[2]) for t in np.arange(0, 6, 1)])
print('y_t:', [y_t(t, x_est[1], x_est[2]) for t in np.arange(0, 6, 1)])
print('theta_t:', [theta(t, x_est[2]) for t in np.arange(0, 6, 1)])

X = np.array([[x_t(t, x_est[0], x_est[2]), y_t(t, x_est[1], x_est[2]), theta(t, x_est[2])] for t in np.arange(0, 6, 1)])
print('============State Vector===================')
print(X)

rho1=observations(1, x_est[0], x_est[1], x_est[2], np.array([[0, 0], [10, 0], [0, 10]]))
print('============Observations===================')
print('rho1:', rho1)

rho2 = observations(2, x_est[0], x_est[1], x_est[2], np.array([[0, 0], [10, 0], [0, 10]]))
print('rho2:', rho2)

rho3 = observations(3, x_est[0], x_est[1], x_est[2], np.array([[0, 0], [10, 0], [0, 10]]))
print('rho3:', rho3)

H_1 = np.array([[x_t(1, x_est[0], x_est[2]) / rho1[0], y_t(1, x_est[1], x_est[2]) / rho1[0], 0],[x_t(1, x_est[0], x_est[2]) / rho1[2], (y_t(1, x_est[1], x_est[2]) - 10) / rho1[2], 0],[(x_t(1, x_est[0], x_est[2]) - 10) / rho1[1], y_t(1, x_est[1], x_est[2]) / rho1[1], 0]])
print('============H_1===================')
print(H_1)

A= A(x_est[0], x_est[1], x_est[2])
print('============A===================')
print(A)

def phi_i():
  return np.array([[1,0,(3/2)*(np.cos(X[1][2])-np.cos(x_est[2]))],[0,1,(3/2)*(np.sin(X[1][2])-np.sin(x_est[2]))],[0,0,1]])

print('============phi_i===================')
print(phi_i())

H= H_1@phi_i()
print('============H===================')
print(H)

W=np.eye(3)*400

nambla=np.transpose(H)@W@H
print('============Nambla===================')
print(nambla)