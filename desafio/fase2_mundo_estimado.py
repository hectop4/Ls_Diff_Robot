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
import fase1_mundo_perfeito as f1

Y=f1.noisy_observations
print('============Noisy Observations===================')
print(Y)

X_est_init=np.array([-3,-3,-1.5])
X_est_prop=np.array([[f1.x_t(t, X_est_init[0], X_est_init[2]), f1.y_t(t, X_est_init[1], X_est_init[2]), f1.theta(t, X_est_init[2])] for t in f1.t_sim])
print('============Initial Estimated State===================')
print(X_est_init)
print('============Propagated Estimated State===================')
print(X_est_prop)

def comparasion_real_estimated(real, estimated,file_name=None):
    plt.figure(figsize=(24,24))
    plt.plot(real[:,1], real[:, 0], label='Real x', color='blue', linewidth=10)
    plt.plot(estimated[:,1], estimated[:, 0], label='Estimated x', color='#FFA500AB', linestyle='--', linewidth=8)
    plt.plot(f1.rho_beacons[:, 1], f1.rho_beacons[:, 0], 'ro', label='Beacons', markersize=30)
    plt.xlabel('Time (s)')
    plt.ylabel('X Position')
    plt.title('Real vs Estimated XY Position')
    plt.legend(fontsize=24)
    plt.tight_layout()
    plt.savefig(file_name) if file_name else None
    plt.show()



class BatchLeastSquares:
    def __init__(self, X_est_init, Y):
        self.X_est_init = X_est_init
        self.Y = Y
        self.X_est = X_est_init.copy()
        self.n_iter = 10
        self.W = np.eye(3) * 400
        self.nambla = np.zeros((3, 3))
        self.N= np.zeros(3)
      

    def A(self, x, y, theta):
        return np.array([[0,0,-f1.v*np.sin(theta)],[0,0,f1.v*np.cos(theta)],[0,0,0]])
    def phi(self, t, xs):
        th = f1.theta(t, xs[2])
        return np.array([[1, 0, (f1.v/f1.omega) * (np.cos(th) - np.cos(xs[2]))],
                         [0, 1, (f1.v/f1.omega) * (np.sin(th) - np.sin(xs[2]))],
                         [0, 0, 1]])
    def H_tilde(self, t, xs):
        x, y = f1.x_t(t, xs[0], xs[2]), f1.y_t(t, xs[1], xs[2])
        rho = f1.observations(t, xs[0], xs[1], xs[2], f1.rho_beacons)
        return np.column_stack([(x - f1.rho_beacons[:, 0]) / rho, (y - f1.rho_beacons[:, 1]) / rho, np.zeros(3)])
    def run(self):
        total_delta_x=np.array([])

        for i in range(self.n_iter):
            self.nambla = np.zeros((3, 3))
            self.N = np.zeros(3)
            for k, t in enumerate(f1.t_obs):
                H_total = self.H_tilde(t, self.X_est) @ self.phi(t, self.X_est)
                residual = self.Y[k] - f1.observations(t, *self.X_est, f1.rho_beacons)
                self.nambla += H_total.T @ self.W @ H_total
                self.N += H_total.T @ self.W @ residual
            delta_X = np.linalg.solve(self.nambla, self.N)
            total_delta_x=np.append(total_delta_x, delta_X[0])

            print(f"Iteration {i+1}: delta_X = {delta_X}")
            self.X_est += delta_X
            print(f"Iteration {i+1}: X_est = {self.X_est}")
        self.P0=np.linalg.inv(self.nambla)
        print(f"Final estimated state: {self.X_est}")
        print(f"Final estimated covariance: {self.P0}")
        print(f"Final estimated standard deviation: {np.sqrt(np.diag(self.P0))}")
        return self.X_est, self.P0, np.sqrt(np.diag(self.P0)), total_delta_x
  

    


if __name__ == "__main__":
    print("This module implements the Batch Least Squares algorithm for state estimation.")
    comparasion_real_estimated(f1.x, X_est_prop, file_name="real_vs_estimated_init.png")
    bsls = BatchLeastSquares(X_est_init, Y)
    estimation=bsls.run()[0]
    delta_x=bsls.run()[3]
    print('============Final delta State===================')
    print(delta_x)

    comparasion_real_estimated(f1.x, np.array([[f1.x_t(t, estimation[0], estimation[2]), f1.y_t(t, estimation[1], estimation[2]), f1.theta(t, estimation[2])] for t in f1.t_sim]), file_name="real_vs_final_estimated_final.png")

