import numpy as np, tact
import wmpc    # state '2': scipy MPC + legacy Wbc (backup-faithful)
#import wmpc2   # state '3': OSQP MPC + WbcQp hybrid (archived to backup/wmpc2.py)

class Controller:
    n_y = 28  # number of outputs
    n_u = 3   # number of control input

    def __init__(self, env, ymlname, prefix='', rate=None, verbose=False):
        self.verbose = verbose
        self.rate = rate    # control loop ticks/sec; rate-aware logic TBD

        # Fixed-base model: used by HopRaibert (body-frame Jacobian) and the joint/task
        # PID test states. WBC+MPC uses m_fb (floating-base) instead.
        self.m = tact.Model(ymlname, fixed_base=True)
        self.m_fb = tact.Model(ymlname, fixed_base=False)
        self.env = env
        self.prefix = ''

        self.ee = {'foot': '3d'}
        self.pid = tact.PIDController(10, 0, 0, 0.001)
        self.jtc = tact.JacobianTransposeController(self.m, self.ee, np.array([50, 50, 50]), np.zeros(3))
        self.trj = tact.MovingAverageWaypointSmoother(100)

        #self.shift('1')
        self.shift('2')
        self.T = 0

        self.v1 = np.zeros(3)
        #self.fout = open('./log.txt', 'w')

        self.vest = tact.ContactAidedEKF(
            dt=self.m.dt, nfeet=1,
            contact_threshold=10.0,
            sigma_a=0.2, sigma_ba=0.002, sigma_kin=0.05,
            t_warmup=2000,
        )
        self._v_sqerr_sum = np.zeros(3)
        self._rmse_n = 0

    def shift(self, s):
        self.s = self.next_s = s
        self.t = 0

    def one_step_forward(self):
        if self.s != self.next_s: self.shift(self.next_s)
        else: self.t += 1
        self.T += 1

    def msgproc(self, w):
        if w[0] in ['0', 't', 'T', '1', '2']: self.shift(w[0])  # '3' disabled (wmpc2 archived)
        elif self.s == '1': self.ho1.msgproc(w)
        elif self.s == '2': self.ho2.msgproc(w)
        #elif self.s == '3': self.ho3.msgproc(w)

    #def estimate_v(self, q, qd, R, a, w):
    #    self.v1 += 0.001*(R @ a - np.array([0, 0, 9.81]))
    #    v2 = -R @ np.cross(w, self.m.fk(self.ee, q)) - R @ self.m.jacob(self.ee, q) @ qd
    #    if self.s=='1' and self.t > 0 and self.ho1.s == 2 and self.ho1.t == 0: self.v1 = 1.0*v2
    #    return self.v1

    def update(self, y):
        q, qd, f, R, a, w, v, p = y[:3], y[3:6], y[6:12], tact.quat_to_rotation(y[12:16]), y[16:19], y[19:22], y[22:25], y[25:28]

        v_hat = self.vest.step(R, w, a, q, qd, f, self.m.fk(self.ee, q), self.m.jacob(self.ee, q))
        #self._v_sqerr_sum += (v - v_hat) ** 2
        #self._rmse_n += 1
        #if self._rmse_n == 200:
        #    rmse = np.sqrt(self._v_sqerr_sum / self._rmse_n)
        #    print(f'T={self.T:6d}'
        #          f'  v    = [{v[0]:+.4f} {v[1]:+.4f} {v[2]:+.4f}]'
        #          f'  v_hat= [{v_hat[0]:+.4f} {v_hat[1]:+.4f} {v_hat[2]:+.4f}]'
        #          f'  rmse = [{rmse[0]:.4f} {rmse[1]:.4f} {rmse[2]:.4f}]')
        #    self._v_sqerr_sum[:] = 0
        #    self._rmse_n = 0

        tau = np.zeros(3)

        if self.s == '0': #test joint space
            if self.t == 0: self.trj.target(np.zeros((1, 3)), [1000], q, self.T)
            tau = self.pid.update(self.trj.generate(), q, qd)

        elif self.s == 't': #test joint space
            if self.t == 0: self.trj.target(np.array([[0.2, 0.2, -0.1]]), [1000], q, self.T)
            tau = self.pid.update(self.trj.generate(), q, qd)

        elif self.s == 'T': #test task space
            tau = self.jtc.update(np.array([0.0, 0.2, -0.1]), q, qd)

        elif self.s == '1':  # Raibert's style hopper
            if self.t == 0: self.ho1 = HopRaibert(self.m, self.env, self.ee)
            tau = self.ho1.update(q, qd, f, R, v)

        elif self.s == '2':  # original scipy-MPC + legacy Wbc (backup-faithful)
            if self.t == 0: self.ho2 = wmpc.Hop(self.m_fb, self.env, self.ee)
            tau = self.ho2.update(q, qd, f, R, v, w, p)

        #elif self.s == '3':  # OSQP-MPC + WbcQp hybrid (archived to backup/wmpc2.py)
        #    if self.t == 0: self.ho3 = wmpc2.Hop(self.m_fb, self.env, self.ee)
        #    tau = self.ho3.update(q, qd, f, R, v, w, p)

        if R[2][2] < 0.5 and self.s != 'fallen':
            print('FALLEN at T=%d  s=%s  R22=%.3f' % (self.T, self.s, R[2][2]))
            self.next_s = 'fallen'
        self.one_step_forward()
        return tau, None, None, None, None


class HopRaibert:
    def __init__(self, m, env, ee):
        self.pid = tact.PIDController(5, 0, 0, 0.001)
        self.jtc = tact.JacobianTransposeController(m, ee, [50, 50, 50], [0, 0, 0])

        self.v_d = [0, 0] #velocity desired (in global)
        self.H_d = 140 #height desired (tick)

        self.flag = None
        self.shift(2)

    def shift(self, s):
        self.s = self.next_s = s
        self.t = 0

    def one_step_forward(self):
        if self.s != self.next_s: self.shift(self.next_s)
        else: self.t += 1

    def msgproc(self, w):
        if   w[0] == 'a': self.v_d[0] += 0.1
        elif w[0] == 'd': self.v_d[0] -= 0.1
        elif w[0] == 'w': self.v_d[1] -= 0.1
        elif w[0] == 's': self.v_d[1] += 0.1
        elif w[0] == 'e': self.v_d[0] = 0; self.v_d[1] = 0
        elif w[0] == '1': self.H_d -= 30
        elif w[0] == '2': self.H_d += 30
        print(self.v_d, self.H_d)

    def fly(self, q, qd, f, R, v):
        #if t < 5: return [jt1.update([0, 0, -0.5], q, qd)]

        b = np.zeros(2)
        b[0] = -0.05*(self.v_d[0] - v[0])
        b[1] = -0.05*(self.v_d[1] - v[1])

        x = np.zeros(3)
        x[0] = 0.15*v[0] + b[0]
        x[1] = 0.15*v[1] + b[1]
        x[2] = -0.50
        x = np.transpose(R) @ x

        tau = self.jtc.update(x, q, qd)
        return tau

    def land(self, q, qd, f, R):
        ref = np.array([0, 0, -0.50])
        tau = self.jtc.update(ref, q, qd)
        return tau

    def fire(self, q, qd, f, R):
        ref = np.array([0, 0, -0.50])
        tau = self.jtc.update(ref, q, qd)
        if self.t < 30: tau[2] = 20.0
        return tau

    def update(self, q, qd, f, R, v):
        if self.s == 0:
            tau = self.fly(q, qd, f, R, v)
            if f[2] > 10: self.next_s = 1

        elif self.s == 1:
            tau = self.land(q, qd, f, R)
            if self.t == 60: self.next_s = 2 #; print(v)

        elif self.s == 2:
            tau = self.fire(q, qd, f, R)
            if f[2] < 1: self.next_s = 0

        self.one_step_forward()
        return tau


'''
class mlib:
    def fk(self, _, q):
        s = np.array(list(map(math.sin, q)))
        c = np.array(list(map(math.cos, q)))
        x = np.zeros(3)

        x[0] = -0.5*s[1]
        x[1] =  0.5*c[1]*s[0]
        x[2] = -0.5*c[0]*c[1]
        return x

    def jacob(self, _, q):
        s = np.array(list(map(math.sin, q)))
        c = np.array(list(map(math.cos, q)))
        J = np.zeros((3, 3))

        J[0][0] = 0
        J[0][1] = -0.5*c[0]*c[0]*c[1] - 0.5*c[1]*s[0]*s[0]
        J[0][2] = 0
        J[1][0] = 0.5*c[0]*c[1]
        J[1][1] = -0.5*s[0]*s[1]
        J[1][2] = 0
        J[2][0] = 0.5*c[1]*s[0]
        J[2][1] = 0.5*c[0]*s[1]
        J[2][2] = 0
        return J


    def error(self, _, q, x_d):
        x = self.fk(_, q)
        e = x_d - x
        return e
'''

'''
    def gravity(self, q, g):
        s = np.array(list(map(math.sin, q)))
        c = np.array(list(map(math.cos, q)))
        tau = np.zeros(3)

        tau[0] = 0.05*g[1]*c[0]*c[1]+0.05*g[2]*c[1]*s[0]
        tau[1] = 0.05*g[2]*c[0]*s[1]-0.2*g[0]*(0.25*c[0]*c[0]*c[1]+0.25*c[1]*s[0]*s[0])-0.05*g[1]*s[0]*s[1]
        return tau


    def com(self, q):
        s = np.array(list(map(math.sin, q)))
        c = np.array(list(map(math.cos, q)))
        com = np.zeros(3)

        com[0] = -0.0384615*s[1]
        com[1] = 0.0384615*c[1]*s[0]
        com[2] = 0.153846-0.0384615*c[0]*c[1]
        return com
'''


'''
    #kick phase-1
    def ground(self, q, qd, f, R):
        ref = np.array([0, 0, -0.45])
        tau = self.jtc.update(ref, q, qd)

        if 100 <= self.t < 200:
            tau = self.jtc.update(ref, q, qd)
            tau[2] = 40
        return tau

    #kick phase-2
    def flying(self, q, qd, f, R, v):
        target = np.array([0, 1, -0.3])
        tau = self.pid.update(target, q, qd)

        if 100 < self.t < 300:
            x = np.zeros(3)
            x[0] = 0.2*v[0]
            x[1] = 0.2*v[1]
            x[2] = -0.50

            x = np.matmul(np.transpose(R), x)
            tau = self.jtc.update(x, q, qd)

        return tau
'''

'''
        elif self.s == 11:
            tau = self.ground(q, qd, f, R)
            if f[2] < 1: self.next_s = 12

        elif self.s == 12:
            tau = self.flying(q, qd, f, R, v)
            if f[2] > 5: self.next_s = 1

'''
