import numpy as np
import estimations
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
x_vect = np.array([3,3,0.5])

print('============Measurements===================')
print('x_t:', [x_t(t, x_vect[0], x_vect[2]) for t in np.arange(0, 6, 1)])
print('y_t:', [y_t(t, x_vect[1], x_vect[2]) for t in np.arange(0, 6, 1)])
print('theta_t:', [theta(t, x_vect[2]) for t in np.arange(0, 6, 1)])

X = np.array([[x_t(t, x_vect[0], x_vect[2]), y_t(t, x_vect[1], x_vect[2]), theta(t, x_vect[2])] for t in np.arange(0, 6, 1)])
print('============State Vector===================')
print(X)

rho1=observations(1, x_vect[0], x_vect[1], x_vect[2], np.array([[0, 0], [10, 0], [0, 10]])) + np.random.normal(0, 0.1, 3)
print('============Observations===================')
print('rho1:', rho1)

rho2 = observations(2, x_vect[0], x_vect[1], x_vect[2], np.array([[0, 0], [10, 0], [0, 10]])) + np.random.normal(0, 0.1, 3)
print('rho2:', rho2)

rho3 = observations(3, x_vect[0], x_vect[1], x_vect[2], np.array([[0, 0], [10, 0], [0, 10]])) + np.random.normal(0, 0.1, 3)
print('rho3:', rho3)



stations = np.array([[0, 0], [10, 0], [0, 10]])
x_ref = estimations.x_est

def H_tilde(t, xs):
  x, y = x_t(t, xs[0], xs[2]), y_t(t, xs[1], xs[2])
  rho = observations(t, xs[0], xs[1], xs[2], stations)
  return np.column_stack([(x - stations[:, 0]) / rho, (y - stations[:, 1]) / rho, np.zeros(3)])

def phi(t, xs):
  th = theta(t, xs[2])
  return np.array([[1, 0, (3/2) * (np.cos(th) - np.cos(xs[2]))],
                   [0, 1, (3/2) * (np.sin(th) - np.sin(xs[2]))],
                   [0, 0, 1]])

A = A(x_ref[0], x_ref[1], x_ref[2])
print('============A===================')
print(A)

W = np.eye(3) * 400

nambla = np.zeros((3, 3))
N = np.zeros(3)
for t, rho_obs, rho_est in [(1, rho1, estimations.rho1), (2, rho2, estimations.rho2), (3, rho3, estimations.rho3)]:
  H = H_tilde(t, x_ref) @ phi(t, x_ref)
  res = rho_obs - rho_est
  print(f'============t={t}===================')
  print('H:\n', H)
  print('Residual:', res)
  nambla += H.T @ W @ H
  N += H.T @ W @ res

print('============Nambla===================')
print(nambla)
print('============N===================')
print(N)

x_res_0 = np.linalg.solve(nambla, N)
print('============x_res_0===================')
print(x_res_0)
print('============x_est + x_res_0===================')
print(x_ref + x_res_0)

cov_est=np.linalg.inv(nambla)
print('============Covariance Estimation===================')
print(cov_est)

print('============Covariance Estimation Diagonal===================')
print(np.sqrt(np.diag(cov_est)))
