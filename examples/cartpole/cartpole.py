import time, numpy as np, zmq, tact
from scipy.optimize import minimize
import stable_baselines3 as sb3

def mpc_cont(m, q0, qd0):
    horizon = 20
    tau_max = 200
        
    def cost_func(tau, q, qd):
        cost = 0
        ctx = None                       # cold-start each rollout; warm-thread within it
        for i in range(horizon):
            q, qd, y, ctx = m.step(q, qd, np.array([tau[i], 0]), ctx=ctx)
            cost += 1.0*q[1]**2  #+ abs(q[0])
        return cost

    bounds = [(-tau_max, tau_max) for i in range(horizon)]
    tau_ini = np.zeros(horizon)        

    result = minimize(cost_func, tau_ini, args=(q0, qd0), bounds=bounds)
    return result.x[0:1]

class Controller:
    def __init__(self, env, ymlname, prefix=None, rate=None, verbose=False):
        self.n_y = 4 #number of outputs
        self.n_u = 2 #number of control input

        self.rate = rate          # control-loop ticks/sec from runner (start passes rate=)
        self.verbose = verbose
        
        self.m = env.m
        self.shift(0)
        self.T = 0

        self.trj = tact.MovingAverageWaypointSmoother(100)
        self.pid = tact.PIDController(100, 1, 0, 0.001)
        #self.model = sb3.PPO.load('out')
        
    def shift(self, s):
        self.s = self.next_s = s
        self.t = 0

    def one_step_forward(self):
        if self.s != self.next_s: self.shift(self.next_s)
        else: self.t += 1
        self.T += 1
        
    def msgproc(self, w):
        if w[0] in ['home', 'zero', 'test1', 'test2']: self.shift(w[0])

    def update(self, y):
        q, qd = y[:2], y[2:4]
        tau = np.zeros(2)

        if self.s == 'zero':
            if self.t == 0: self.trj.target(np.array([[0.0]]), [1000], q[0], self.T)
            tau[0] = self.pid.update(self.trj.generate(), q[:1], qd[:1])[0]
            
        elif self.s == 'home':
            if self.t == 0: self.trj.target(np.array([[1.0]]), [1000], q[0], self.T)
            tau[0] = self.pid.update(self.trj.generate(), q[:1], qd[:1])[0]

            #tau[0] = self.pid.update(np.array([1.0]), q[:1], qd[:1])[0]

        elif self.s == 'test1':
            tau[0] = mpc_cont(self.m, q, qd)[0]

        elif self.s == 'test2':
            obs = np.zeros(4)
            obs[0:2] = np.tanh(q)
            obs[2:4] = np.tanh(qd)
            action, _ = self.model.predict(obs)
            tau[0] = 100 * action[0]

        if self.verbose:
            print(self.T, q)
            
        self.one_step_forward()
        return tau, None, None

    
'''
    #tau_opt, cost = minimize2(m, q0, qd0) 
    #return tau_opt[0:1]


def minimize2(m, q0, qd0):
    #np.random.seed(int(time.time()%1*10000))
    N = 10
    min_cost = 100000 #just big number
    out = np.zeros(N)    
    
    for i in range(100000):
        if i == 0: tau = np.zeros(N)
        else: tau = 200*np.random.rand(N) -100

        cost = 0
        q = q0
        qd = qd0
        
        for j in range(N):
            #q, qd, y = test_model(q, qd, tau[j])
            q, qd, y = m.step(q, qd, np.array([tau[j], 0]))            
            cost += 1.0*q[1]**2
            
        if cost < min_cost:
            min_cost = cost
            for j in range(N): out[j] = tau[j]

    return out, min_cost
'''

    
'''
class model:
    def __init__(self):
        self.jtype = [2, 1] #prismatic - revolute
        self.parent = [None, 0]
        self.active = [1, 0]
        self.m = np.array([1.0, 1.0])
        self.dt = 0.005
        
        self.I = np.zeros((2, 3, 3))
        self.I[0] = np.diag([0.01, 1/12, 1/12])  #np.zeros((3, 3)) 
        self.I[1] = np.diag([0.01, 1/12, 1/12]) # np.zeros((3, 3)) 

        self.Ti = np.zeros((2, 4, 4)) 
        self.Ti[0] = tact.T_rot_y(np.pi/2) 
        self.Ti[1] = tact.T_rot_y(-np.pi/2) 

        self.Bi = np.zeros((2, 4, 4))
        self.Bi[0] = tact.T_trans([0, 0, 0])
        self.Bi[1] = tact.T_trans([0, 0.5, 0])
        
        self.X = tact.get_spatial_transform(self.Ti)
        self.I6 = tact.get_spatial_inertia(self.Bi, self.m, self.I)

        #self.X = np.zeros((2, 6, 6))
        #self.X[0] = tact.X_trans([0, 0, 0])
        #self.X[1] = tact.X_trans([1, 0, 0])
        
        #self.I6 = np.zeros((2, 6, 6))
        #self.I6[0] = tact.mcI(self.m[0], [1, 0, 0], self.I[0])
        #self.I6[1] = tact.mcI(self.m[1], [1, 0, 0], self.I[1])

        self.view = [90, -90]
        self.lim = [[-2, 2], [-2, 2], [-2, 2]]
        self.vtype = [2, 1, 'o']

        #joint friction factors
        self.ff = np.array([0.1, 0.1])

        self.q0 = np.array([0, np.pi])
        self.qd0 = np.zeros(2)
        self.tau0 = np.zeros(2)
        
        np.set_printoptions(precision=6)
        np.set_printoptions(suppress=True)

    def skeleton(self, q):
        T, B = tact._fk(self.Ti, self.Bi, self.parent, self.jtype, q)
        cart1 = T[0] @ tact.T_trans([0, 0, -0.5])
        cart2 = T[0] @ tact.T_trans([0, 0, 0.5])
        EE = (T[1] @ tact.T_trans([0, 1, 0]))[:3, 3]
        L1 = np.concatenate((cart1[:3, 3], cart2[:3, 3]))
        L2 = np.concatenate((T[1][:3, 3], EE))
        return [L1, L2, EE]
    
    def step(self, q, qd, tau):
        g = [0, -9.81, 0]
        f_ext= np.zeros((2, 6))

        #add some friction
        tau = tau - self.ff * qd

        q_next, qd_next, _ = tact.euler_step(self.X, self.I6, self.parent, self.jtype, q, qd, tau, f_ext, g, self.dt)
        #q_next, qd_next, _ = tact.rk4_step(self.X, self.I6, self.parent, self.jtype, q, qd, tau, f_ext, g, self.dt)

        y = np.concatenate((q_next, qd_next))
        return q_next, qd_next, y

    def inertia(self, q):
        H = tact.crb_featherstone(self.X, self.I6, self.parent, self.jtype, q)
        return H

    def inertia2(self, q):
        T, B = tact._fk(self.Ti, self.Bi, self.parent, self.jtype, q)
        H = tact.inertia_lagrange(T, B, self.parent, self.jtype, self.m, self.I)
        return H
    
    def bias(self, q, qd, g=[0, 0, 0]):
        qdd = np.zeros(2)
        b = tact.rne_featherstone(self.X, self.I6, self.parent, self.jtype, q, qd, qdd, g)
        return b
'''

    
'''
    def step2(self, q, qd, tau):
        g = [0, -9.81, 0]
        #g = [0, 0, 0]

        H = self.inertia(q)
        b = self.bias(q, qd, g)

        #add some friction
        tau = tau - self.ff * qd
        
        H_inv = np.linalg.inv(H)
        qdd = H_inv @ (tau - b)
        
        q_next, qd_next = tact.euler_method(q, qd, qdd, self.dt)
        y = np.concatenate((q_next, qd_next))

        return q_next, qd_next, y
'''

    
'''
    def test(self, q, qd, tau):
        g = [0, -9.81, 0]
        #g = [0, 0, 0]

        H1 = self.inertia(q)
        H2 = self.inertia2(q)
        b = self.bias(q, qd, g)
        
        H_inv = np.linalg.inv(H2)
        qdd1 = H_inv @ (tau - b)

        f_ext= np.zeros((2, 6))
        qdd2, a, v, f = tact.aba_featherstone(self.X, self.I6, self.parent, self.jtype, q, qd, tau, f_ext, g)

        print(qdd1)
        print(qdd2)

        print(H1)
        print(H2)
'''
        
