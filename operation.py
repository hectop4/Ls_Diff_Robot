import numpy as np

def y_estimate(t,x0):
  
  x_0, y_0, th_0 = x0
  return (-3/2)*(np.cos(2*t+th_0)-np.cos(th_0)) + y_0

def th_estimate(t,x0):
  
  x_0, y_0, th_0 = x0
  return 2*t + th_0

def x_estimate(t,x0):
  
  x_0, y_0, th_0 = x0
  return (3/2)*(np.sin(2*t+th_0)-np.sin(th_0)) + x_0

x0=np.array([2.9, 3.31, 0.4])           # estado inicial verdadero [x0, y0, theta0]

x_estimations = np.array([x_estimate(t,x0) for t in np.arange(0,5,1)])
y_estimations = np.array([y_estimate(t,x0) for t in np.arange(0,5,1)])
th_estimations = np.array([th_estimate(t,x0) for t in np.arange(0,5,1)])
print('============Estimaciones===================')
print(x_estimations)
print(y_estimations)
print(th_estimations)

obs_1=np.array([[0.0, 0.0],     # O1
                [10.0, 0.0],     # O2
                [0, 10.0]])    # O3

def observations(t,x0,obs):
  x_0, y_0, th_0 = x0
  x = x_estimate(t,x0)
  y = y_estimate(t,x0)
  return np.sqrt((x - obs[:, 0]) ** 2 + (y - obs[:, 1]) ** 2)

print('============Observaciones Estimada===================')

obs = np.array([observations(t,x0,obs_1) for t in np.arange(0,5,1)])
print(obs)

x_0_real=np.array([3.0, 3.0, 0.5])           # estado inicial verdadero [x0, y0, theta0]
x_estimations_real = np.array([x_estimate(t,x_0_real) for t in np.arange(0,5,1)])
y_estimations_real = np.array([y_estimate(t,x_0_real) for t in np.arange(0,5,1)])
th_estimations_real = np.array([th_estimate(t,x_0_real) for t in np.arange(0,5,1)])
print('===============Estimacion Real================')
print(x_estimations_real)
print(y_estimations_real)
print(th_estimations_real)
print('============Observaciones Real===================')
real_obs = np.array([observations(t,x_0_real,obs_1) for t in np.arange(0,5,1)])
print(real_obs)
print('============Diferencia===================')
y=real_obs-obs
print(y)

H_vir_1=np.array([[x_estimations[1]/obs[1][0], y_estimations[1]/obs[1][0], 0],
                 [(x_estimations[1])/obs[1][2], (y_estimations[1]-10)/obs[1][2], 0],
                 [(x_estimations[1]-10)/obs[1][1], (y_estimations[1])/obs[1][1], 0]]  )
print(H_vir_1)

stm=np.array([[1, 0, -2.4877],
              [0, 1, 0.4291],
              [0, 0, 1]])
print('============STM===================')
print(stm)

H=H_vir_1@stm
print('============H===================')
print(H)

w=np.eye(3)*400

nambla=np.transpose(H)@w@H

print('============Nambla===================')
print(nambla)

N=np.transpose(H)@w@y[1]
print('============N===================')
print(N)

x_res_0=np.linalg.inv(nambla)@N
print('============x_res_0===================')
print(x_res_0)

x_new_est=np.array([x_estimations[0], y_estimations[0], th_estimations[0]]) + x_res_0
print('============x_new_est===================')
print(x_new_est)