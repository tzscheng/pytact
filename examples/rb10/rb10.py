import ctypes, numpy as np, tact
from math import pi as pi

'''
def scan3(env, x):
    L = np.zeros(1024*3)
    _L = (ctypes.c_double*len(L))(*L)
    _x = (ctypes.c_double*len(tau))(*x)
    env.scan3(_x, _L)
    p = np.array(_L)
    pc = []
    for i in range(n): pc.append([p[3*i+0], p[3*i+1], p[3*i+2]])
    return pc
'''

class Controller:
    def __init__(self, env, ymlname, prefix=None, verbose=False):
        self.n_y = 12 #number of outputs
        self.n_u = 6 #number of control input

        self.verbose = verbose

        self.env = env
        self.m = tact.Model('rb10')
        self.InverseKinematics = InverseKinematics()

        self.shift(None)
        self.T = 0

        k_p = np.array([500, 500, 500, 300, 300, 50], dtype=float)
        k_d = np.array([10, 10, 10, 3, 3, 0], dtype=float)
        
        self.pid = tact.PIDController(k_p, k_d, 0, 0.001)
        self.trj1 = tact.MovingAverageWaypointSmoother(100) #joint space trajectory generator
        
    def shift(self, s):
        self.s = self.next_s = s
        self.t = 0

    def one_step_forward(self):
        if self.s != self.next_s: self.shift(self.next_s)
        else: self.t += 1
        self.T += 1
        
    def msgproc(self, w):
        self.shift(w[0])
        self.v = np.array(w[1:], dtype=float)

    #def err_norm(self, q):
    #    if len(self.pd4.ft) == 0: norm = 10000.0 #just large number
    #    else: norm = np.linalg.norm(self.pd4.ft - q)
    #    return norm

    def ik(self, x, q):
        S = self.InverseKinematics.RB10_invKine(x[0], x[1], x[2], x[3], x[4], x[5])[0]
        if len(S) == 0: print('no IK solution for', x); exit(0)
        E = np.zeros(len(S))
        
        for i in range(len(S)):
            for j in range(6):
                E[i] += abs(S[i][j]-q[j])

            #avoid abnormal pose
            if S[i][2] < 0: E[i] += 100

        idx = np.argmin(E)
        return S[idx]
        
    def update(self, y):
        q, qd = y[:6], y[6:12]
        tau = np.zeros(6)

        #if self.T % 1000 == 0:
        #    if type(self.env) == ctypes.CDLL: pc = [] # scan3(0, 0, 2, 0, np.pi/2, 0)
        #    else: pc = self.env.scan3(0, 0, 2, 0, np.pi/2, 0)
        #    msg = 'pc'
        #    for i in range(len(pc)): msg += ' %f %f %f' %(pc[i][0], pc[i][1], pc[i][2])
        #    print(msg)
            
        if self.s == 'zero':
            if self.t == 0: self.trj1.target(np.zeros((1, 6)), [2000], q, self.T)
            q_d = self.trj1.generate()
            tau = self.pid.update(q_d, q, qd) + self.m.gravity(q)

        elif self.s == 'home':
            if self.t == 0: self.trj1.target(np.array([[0, -0.54, 2.12, 0, 1.57, 0]]), [2000], q, self.T)
            q_d = self.trj1.generate()
            tau = self.pid.update(q_d, q, qd) + self.m.gravity(q)

        elif self.s == 'pid':
            if self.t == 0: self.trj1.target(self.v.reshape((1, 6)), [1000], q, self.T) #self.env.push(('jall %f %f %f %f %f %f' %(q_d[0], q_d[1], q_d[2], q_d[3], q_d[4], q_d[5])).encode())
            q_d = self.trj1.generate()
            tau = self.pid.update(q_d, q, qd) + self.m.gravity(q)

        elif self.s == 'ik1':
            if self.t == 0:
                q_d = self.ik(self.v, q)
                self.trj1.target(q_d.reshape((1, 6)), [1000], q, self.T) #self.env.push(('jall %f %f %f %f %f %f' %(q_d[0], q_d[1], q_d[2], q_d[3], q_d[4], q_d[5])).encode())
            x_d = self.trj1.generate()
            tau = self.pid.update(x_d, q, qd) + self.m.gravity(q)

        elif self.s == 'iktest':
            if self.t == 0:
                q_d = self.m.ik({'tcp':'6d'}, q, np.array([0.5, 0.0, 0.4, np.pi/2, 0, np.pi/2]))
                self.trj1.target(q_d.reshape((1, 6)), [2000], q, self.T)
            tau = self.pid.update(self.trj1.generate(), q, qd) + self.m.gravity(q)
            
        #elif self.s == 'path':
        #    if self.t == 0:
        #        num = int(len(self.v)/7)
        #        q_d = np.zeros((num, 6))
        #        tick = np.zeros(num, dtype=int)
                
        #        for i in range(num):
        #            q_d[i] = self.ik(np.array(self.v[7*i+1:7*i+7], dtype=float), q)
        #            tick[i] = int(self.v[7*i])
                    
        #        self.pd4.target(q_d, tick, q, self.T)
        #        #self.env.push(('mvpb2 %f %f %f %f %f %f  %f %f %f %f %f %f' %(self.v[1], self.v[2], self.v[3], self.v[4], self.v[5], self.v[6],  self.v[7], self.v[8], self.v[9], self.v[10], self.v[11], self.v[12])).encode())
        #    tau = self.pd4.update(q, d) + self.m.gravity(q)
            
        self.one_step_forward()
        return tau, None, None


class InverseKinematics:
    def __init__(self):    
        self.d0 = 0.1970
        d1 = 0
        a2 = -0.6127
        a3 = -0.57015
        d4 =  0.15615
        d5 =  0.11715
        d6 =  0.1153
        gripper_offset = 0

        self.d = np.array([d1, 0, 0, d4, d5, d6 + gripper_offset])        
        self.a = np.array([0, a2, a3, 0, 0, 0])                         
        self.alpha = np.array([pi/2, 0, 0, pi/2, -pi/2, 0], dtype=np.float32)
        self.theta = np.zeros(6, dtype=np.float32)
        self.config = np.array([0, -pi/2, 0, -pi/2, 0, 0]) # DH parameter theta_config
        self.number_of_joints = 6

    def RB10_invKine(self, px, py, pz, ox, oy, oz):
        rot_x = np.matrix([[1, 0, 0, 0], [0, np.cos(ox), -np.sin(ox), 0], [0, np.sin(ox), np.cos(ox), 0], [0, 0, 0, 1]])
        rot_y = np.matrix([[np.cos(oy), 0, np.sin(oy), 0], [0, 1, 0, 0], [-np.sin(oy), 0, np.cos(oy), 0], [0, 0, 0, 1]])
        rot_z = np.matrix([[np.cos(oz), -np.sin(oz), 0, 0], [np.sin(oz), np.cos(oz), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

        Position_GE = [px, py, pz]
        Orientation_GE = rot_z*rot_y*rot_x 
    
        T_GE = rot_z*rot_y*rot_x + np.matrix([[0, 0, 0, px], [0, 0, 0, py], [0, 0, 0, pz], [0, 0, 0, 0]])
        T_0E = rot_z*rot_y*rot_x + np.matrix([[0, 0, 0, px], [0, 0, 0, py], [0, 0, 0, pz - self.d0], [0, 0, 0, 0]])
        
        theta_i = pi
        alpha_i = -pi/2
        a_i = 0
        d_i = 0
        A_i = self.homogeneous_transformation_i(theta_i, alpha_i, a_i, d_i)
        T_06 = np.matmul(T_0E, np.linalg.inv(A_i))
    
        pose = np.array([T_06[0, 3], T_06[1, 3], T_06[2, 3]])
        ee_orientation = np.array(T_06[0:3, 0:3])
        theta_RB10 = self.inverse_kine(pose, ee_orientation)
    
        if theta_RB10.shape[0] == 0:
            th_RB10_changed = theta_RB10
        
        else:
            th_RB10_changed = np.zeros((theta_RB10.shape[0], 6))
            th_RB10_changed[:, 0] =  (theta_RB10[:,0] - self.config[0]) 
            th_RB10_changed[:, 1] = -(theta_RB10[:,1] - self.config[1]) 
            th_RB10_changed[:, 2] = -(theta_RB10[:,2] - self.config[2]) 
            th_RB10_changed[:, 3] = -(theta_RB10[:,3] - self.config[3]) 
            th_RB10_changed[:, 4] =  (theta_RB10[:,4] - self.config[4]) 
            th_RB10_changed[:, 5] = -(theta_RB10[:,5] - self.config[5])
        
        return th_RB10_changed, Position_GE, Orientation_GE
    
    def inverse_kine(self, pose, orientation):
        pose_vector_form = np.reshape([pose[0], pose[1], pose[2], 1], (-1, 1))
        pose_full_form = np.hstack((np.vstack((orientation, np.zeros((1, 3)))), pose_vector_form))
        joints_ik = np.zeros((8, 6), dtype=np.float32) #thetas are the joint variables

        #computes wrist center-> frame5
        wrist_center = np.array(pose - self.d[5]*orientation[:, 2], dtype=np.float32)
        wrist_x = wrist_center[0]
        wrist_y = wrist_center[1]
        wrist_r = np.sqrt(wrist_x**2 + wrist_y**2)
    
        psi = np.arctan2(wrist_y, wrist_x)
        phi = np.arccos(self.d[3]/wrist_r)

        #solutions for theta1, either left or right shoulder
        theta1_possible_values = np.array([np.pi/2 + psi + phi, np.pi/2 + psi - phi], dtype=np.float32)
        joints_ik[0:4, 0] = theta1_possible_values[0] 
        joints_ik[4: , 0] = theta1_possible_values[1] 

        theta5_possible_values = np.zeros(2)
        theta6_possible_values = np.zeros(2*2)

        for i in range(len(theta1_possible_values)):
            T_01 = self.get_transformation_matrix(joint_variables=[theta1_possible_values[i], 0, 0, 0, 0, 0], start_joint=0, end_joint=0)
            T_10 = np.linalg.inv(T_01)
            pose_frame1 = np.matmul(T_10, pose_vector_form)
            pose_full_form_frame1 = np.matmul(T_10, pose_full_form)            
            start_indice = i*4

            #theta5
            theta5_possible_values[i] = np.arccos((pose_frame1[2] - self.d[3])/self.d[5])
            joints_ik[start_indice  :start_indice+2, 4] =  theta5_possible_values[i]
            joints_ik[start_indice+2:start_indice+4, 4] = -theta5_possible_values[i]

            #theta6
            theta6_possible_values[i*2  ] = np.arctan2(-pose_full_form_frame1[2, 1]/np.sin( theta5_possible_values[i]), pose_full_form_frame1[2, 0] / np.sin( theta5_possible_values[i]))
            theta6_possible_values[i*2+1] = np.arctan2(-pose_full_form_frame1[2, 1]/np.sin(-theta5_possible_values[i]), pose_full_form_frame1[2, 0] / np.sin(-theta5_possible_values[i]))
            joints_ik[start_indice  :start_indice+2, 5] = theta6_possible_values[i*2  ]
            joints_ik[start_indice+2:start_indice+4, 5] = theta6_possible_values[i*2+1]

        impossible_combinations = []
        for combination_theta1_theta6 in range(len(theta6_possible_values)):
            hypothesis_index = 2*combination_theta1_theta6
            T_01 = self.get_transformation_matrix(joint_variables=joints_ik[hypothesis_index, :], start_joint=0, end_joint=0)
            T_10 = np.linalg.inv(T_01)
            pose_full_form_frame1 = np.matmul(T_10, pose_full_form)
            T_56 = self.get_transformation_matrix(joint_variables=joints_ik[hypothesis_index, :], start_joint=5, end_joint=5)
            T_45 = self.get_transformation_matrix(joint_variables=joints_ik[hypothesis_index, :], start_joint=4, end_joint=4)
            T_64 = np.linalg.inv(np.matmul(T_45,T_56))
            T_14 = np.matmul(pose_full_form_frame1, T_64)
            P_13 = np.matmul(T_14, np.resize([0, -self.d[3], 0, 1], (4, 1)))[0:3]
            coeff = (np.linalg.norm(P_13)**2 - self.a[1]**2 - self.a[2]**2)/(2*self.a[1]*self.a[2])
            
            if coeff > 1 or coeff < -1:
                theta3 = np.nan
                impossible_combinations.insert(0, hypothesis_index)
                impossible_combinations.insert(0, hypothesis_index+1)
            else:
                theta3 = np.arccos((np.linalg.norm(P_13)**2 - self.a[1]**2 - self.a[2]**2)/(2*self.a[1]*self.a[2]))

            joints_ik[hypothesis_index  , 2] = -theta3
            joints_ik[hypothesis_index+1, 2] =  theta3

        possible_joints = copy.deepcopy(joints_ik)
        for i in impossible_combinations: possible_joints = np.delete(possible_joints, i, axis=0)
        joints_ik = copy.deepcopy(possible_joints)
        
        for hypothesis_index in range(joints_ik.shape[0]):
            T_01 = self.get_transformation_matrix(joint_variables=joints_ik[hypothesis_index, :], start_joint=0, end_joint=0)
            T_10 = np.linalg.inv(T_01)
            pose_full_form_frame1 = np.matmul(T_10, pose_full_form)
            T_56 = self.get_transformation_matrix(joint_variables=joints_ik[hypothesis_index, :], start_joint=5, end_joint=5)
            T_65 = np.linalg.inv(T_56)
            T_45 = self.get_transformation_matrix(joint_variables=joints_ik[hypothesis_index, :], start_joint=4, end_joint=4)
            T_54 = np.linalg.inv(T_45)
            T_14 = np.matmul(np.matmul(pose_full_form_frame1, T_65), T_54)
            P_13 = np.matmul(T_14,np.resize([0, -self.d[3], 0, 1], (4, 1)))[0:3]
            joints_ik[hypothesis_index, 1] = -np.arctan2(P_13[1], -P_13[0]) + np.arcsin(self.a[2]*np.sin(joints_ik[hypothesis_index, 2])/np.linalg.norm(P_13))
            T_23 = self.get_transformation_matrix(joint_variables=joints_ik[hypothesis_index, :], start_joint=2, end_joint=2)
            T_32 = np.linalg.inv(T_23)
            T_12 = self.get_transformation_matrix(joint_variables=joints_ik[hypothesis_index, :], start_joint=1, end_joint=1)
            T_21 = np.linalg.inv(T_12)
            T_34 = np.matmul(np.matmul(T_32, T_21), T_14)
            joints_ik[hypothesis_index, 3] = np.arctan2(T_34[1, 0], T_34[0, 0])

        joints_ik = joints_ik[~np.isnan(joints_ik).any(axis=1)]
        return joints_ik
    
    def homogeneous_transformation_i(self, theta_i, alpha_i, a_i, d_i):
        A_i = np.array([[np.cos(theta_i), -np.sin(theta_i)*np.cos(alpha_i),  np.sin(theta_i)*np.sin(alpha_i), a_i*np.cos(theta_i)], [np.sin(theta_i),  np.cos(theta_i)*np.cos(alpha_i), -np.cos(theta_i)*np.sin(alpha_i), a_i*np.sin(theta_i)], [0, np.sin(alpha_i), np.cos(alpha_i), d_i], [0, 0, 0, 1]], dtype=np.float32)        
        return A_i

    def homogeneous_transformation_i_var_theta(self, i, theta_i):
        return self.homogeneous_transformation_i(theta_i=theta_i, alpha_i=self.alpha[i], a_i=self.a[i], d_i=self.d[i])

    def get_transformation_matrix(self, joint_variables, start_joint, end_joint):
        T_matrix = np.eye(4)
        assert end_joint >= start_joint
        joint_index = start_joint
        
        while joint_index <= end_joint:
            theta_i = joint_variables[joint_index]
            A_i = self.homogeneous_transformation_i_var_theta(joint_index, theta_i)
            T_matrix = np.matmul(T_matrix, A_i)
            joint_index += 1

        return T_matrix

    
'''    
class InverseKinematics:
    def __init__(self):    
        self.d0 = 0.1970
        d1 = 0
        a2 = -0.6127
        a3 = -0.57015
        d4 =  0.15615
        d5 =  0.11715
        d6 =  0.1153
        gripper_offset = 0

        self.d = np.array([d1, 0, 0, d4, d5, d6 + gripper_offset])        
        self.a = np.array([0, a2, a3, 0, 0, 0])                         
        self.alpha = np.array([pi/2, 0, 0, pi/2, -pi/2, 0], dtype=np.float32)
        self.theta = np.zeros(6, dtype=np.float32)
        self.config = np.array([0, -pi/2, 0, -pi/2, 0, 0]) # DH parameter theta_config
        self.number_of_joints = 6
        
        # kin_model = kinematics.kinematics_model(ur_model='rb5', gripper_offset=0.0)   # kinematics_utils 모듈을 사용하는 경우에 주석 제거


    def RB10_invKine(self, px, py, pz, ox, oy, oz):
        #if abs(px) < 0.15 or abs(px) > 1.3:
        #    print("Error code1 : abs(Px) should be the number within 0.15 ~ 1.3")

        #if abs(py) < 0.15 or abs(py) > 1.3:
        #    print("Error code2 : abs(Py) should be the number within 0.15 ~ 1.3")

        #if abs(pz-self.d0) > 1.3:
        #    print("Error code3 : abs(Pz) should be smaller than 1.3 (<1.3)")

        #if abs(ox) > 2*pi or abs(oy) > 2*pi or abs(oz) > 2*pi:
        #    print("Error code4: Ox or Oy or Oz should be the number within -3.14 ~ 3.14")
    
        # Roll (x-axis rotation)
        rot_x = np.matrix([[1, 0, 0, 0],
                           [0, np.cos(ox), -np.sin(ox), 0],
                           [0, np.sin(ox), np.cos(ox), 0],
                           [0, 0, 0, 1]])
        
        # Pitch (y-axis rotation)
        rot_y = np.matrix([[np.cos(oy), 0, np.sin(oy), 0],
                           [0, 1, 0, 0],
                           [-np.sin(oy), 0, np.cos(oy), 0],
                           [0, 0, 0, 1]])

        # Yaw (z-axis rotation)
        rot_z = np.matrix([[np.cos(oz), -np.sin(oz), 0, 0],
                           [np.sin(oz), np.cos(oz), 0, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])

        Position_GE = [px, py, pz]
        # Position_OE = [px, py, pz - d0]
        Orientation_GE = rot_z*rot_y*rot_x 
        # Orientation_OE = Orientation_GE   
    
        T_GE = rot_z*rot_y*rot_x + np.matrix([[0, 0, 0, px],
                                              [0, 0, 0, py],
                                              [0, 0, 0, pz],
                                              [0, 0, 0, 0]]) # Desired Position/Orientation Matrix 
    
        T_0E = rot_z*rot_y*rot_x + np.matrix([[0, 0, 0, px],
                                              [0, 0, 0, py],
                                              [0, 0, 0, pz - self.d0],
                                              [0, 0, 0, 0]]) # Desired Position/Orientation Matrix 
        
        theta_i = pi
        alpha_i = -pi/2
        a_i = 0
        d_i = 0
        A_i = self.homogeneous_transformation_i(theta_i, alpha_i, a_i, d_i)
        T_06 = np.matmul(T_0E, np.linalg.inv(A_i))
    
        pose = np.array([T_06[0, 3], T_06[1, 3], T_06[2, 3]])
        ee_orientation = np.array(T_06[0:3, 0:3])
        theta_RB10 = self.inverse_kine(pose, ee_orientation)
    
        if theta_RB10.shape[0] == 0:
            th_RB10_changed = theta_RB10
        
        else:
            th_RB10_changed = np.zeros((theta_RB10.shape[0], 6))
            th_RB10_changed[:, 0] =  (theta_RB10[:,0] - self.config[0]) 
            th_RB10_changed[:, 1] = -(theta_RB10[:,1] - self.config[1]) 
            th_RB10_changed[:, 2] = -(theta_RB10[:,2] - self.config[2]) 
            th_RB10_changed[:, 3] = -(theta_RB10[:,3] - self.config[3]) 
            th_RB10_changed[:, 4] =  (theta_RB10[:,4] - self.config[4]) 
            th_RB10_changed[:, 5] = -(theta_RB10[:,5] - self.config[5])
    
            # print('IK Test - T_GE : ', T_GE)
            # print('IK Test - T_0E : ', T_0E)
            # print('IK Test - T_06 : ', T_06)
    
        return th_RB10_changed, Position_GE, Orientation_GE

    
    def inverse_kine(self, pose, orientation):
        #other forms of the pose representation:
        #only the position, but in the vector form (with the final 1)
        pose_vector_form = np.reshape([pose[0], pose[1], pose[2], 1], (-1, 1))

        #matrix that enflobes the orientation and the pose
        pose_full_form = np.hstack((np.vstack((orientation, np.zeros((1, 3)))), pose_vector_form))

        #one line means 1 option for each joint (6 joints), there are 8 possibilities for the same ee pose
        joints_ik = np.zeros((8, 6), dtype=np.float32) #thetas are the joint variables

        #####################
        # theta 1 (index 0) #
        #####################

        #computes wrist center-> frame5
        wrist_center = np.array(pose - self.d[5]*orientation[:, 2], dtype=np.float32)
        wrist_x = wrist_center[0]
        wrist_y = wrist_center[1]
        wrist_r = np.sqrt(wrist_x**2 + wrist_y**2)
    
        psi = np.arctan2(wrist_y, wrist_x)
        phi = np.arccos(self.d[3]/wrist_r)
        #solutions for theta1, either left or right shoulder
        theta1_possible_values = np.array([np.pi/2 + psi + phi, np.pi/2 + psi - phi], dtype=np.float32)
        joints_ik[0:4, 0] = theta1_possible_values[0] 
        joints_ik[4: , 0] = theta1_possible_values[1] 

        #####################
        # theta 5 and 6 (index 4 and 5) #
        #####################

        theta5_possible_values = np.zeros(2)
        theta6_possible_values = np.zeros(2*2)

        for i in range(len(theta1_possible_values)):
            #homogeneouos transformation between frames 0 and 1
            T_01 = self.get_transformation_matrix(joint_variables=[theta1_possible_values[i], 0, 0, 0, 0, 0], start_joint=0, end_joint=0)

            #homogeneous transformation 1->0 to obtain desired point in the frame 1
            T_10 = np.linalg.inv(T_01)
            
            #desired pose and orientation in the frame 1
            pose_frame1 = np.matmul(T_10, pose_vector_form)
            pose_full_form_frame1 = np.matmul(T_10, pose_full_form)
            
            start_indice = i*4

            #theta5
            theta5_possible_values[i] = np.arccos((pose_frame1[2] - self.d[3])/self.d[5])
            joints_ik[start_indice  :start_indice+2, 4] =  theta5_possible_values[i]
            joints_ik[start_indice+2:start_indice+4, 4] = -theta5_possible_values[i]

            #theta6
            theta6_possible_values[i*2  ] = np.arctan2(-pose_full_form_frame1[2, 1]/np.sin( theta5_possible_values[i]), pose_full_form_frame1[2, 0] / np.sin( theta5_possible_values[i]))
            theta6_possible_values[i*2+1] = np.arctan2(-pose_full_form_frame1[2, 1]/np.sin(-theta5_possible_values[i]), pose_full_form_frame1[2, 0] / np.sin(-theta5_possible_values[i]))
            joints_ik[start_indice  :start_indice+2, 5] = theta6_possible_values[i*2  ]
            joints_ik[start_indice+2:start_indice+4, 5] = theta6_possible_values[i*2+1]

        #####################
        # theta 3 (index 2) #
        #####################
    
        impossible_combinations = []
        for combination_theta1_theta6 in range(len(theta6_possible_values)):
            hypothesis_index = 2*combination_theta1_theta6

            ### To compute desired pose in frame 4
            #homogeneouos transformation between frames 0 and 1
            T_01 = self.get_transformation_matrix(joint_variables=joints_ik[hypothesis_index, :], start_joint=0, end_joint=0)

            #homogeneous transformation 1->0 to obtain desired point in the frame 1
            T_10 = np.linalg.inv(T_01)

            #desired pose and orientation in the frame 1
            pose_full_form_frame1 = np.matmul(T_10, pose_full_form)

            #transformation that converts from base 4 to frame 6 (T64)
            #homogeneouos transformation between frames 5 and 6
            T_56 = self.get_transformation_matrix(joint_variables=joints_ik[hypothesis_index, :], start_joint=5, end_joint=5)

            #homogeneouos transformation between frames 4 and 5
            T_45 = self.get_transformation_matrix(joint_variables=joints_ik[hypothesis_index, :], start_joint=4, end_joint=4)

            T_64 = np.linalg.inv(np.matmul(T_45,T_56))

            # desired pose in frame 4
            T_14 = np.matmul(pose_full_form_frame1, T_64)

            # desired pose in frame 3
            P_13 = np.matmul(T_14, np.resize([0, -self.d[3], 0, 1], (4, 1)))[0:3]

            #theta 3
            coeff = (np.linalg.norm(P_13)**2 - self.a[1]**2 - self.a[2]**2)/(2*self.a[1]*self.a[2])
            if coeff > 1 or coeff < -1:
                theta3 = np.nan
                #signals the lines to be removed 
                impossible_combinations.insert(0, hypothesis_index)
                impossible_combinations.insert(0, hypothesis_index+1)
            else:
                theta3 = np.arccos((np.linalg.norm(P_13)**2 - self.a[1]**2 - self.a[2]**2)/(2*self.a[1]*self.a[2]))

            #<------------------------
            joints_ik[hypothesis_index  , 2] = -theta3
            joints_ik[hypothesis_index+1, 2] =  theta3

        #deletes impossible hypotesis
        possible_joints = copy.deepcopy(joints_ik)
        for i in impossible_combinations:
            possible_joints = np.delete(possible_joints, i, axis=0)
        joints_ik = copy.deepcopy(possible_joints)
        
        #####################
        # theta 2 and 4 (index 1 and 3) #
        #####################

        for hypothesis_index in range(joints_ik.shape[0]):
            ### To compute desired pose in frame 4
            #homogeneouos transformation between frames 0 and 1
            T_01 = self.get_transformation_matrix(joint_variables=joints_ik[hypothesis_index, :], start_joint=0, end_joint=0)

            #homogeneous transformation 1->0 to obtain desired point in the frame 1
            T_10 = np.linalg.inv(T_01)

            #desired pose and orientation in the frame 1
            pose_full_form_frame1 = np.matmul(T_10, pose_full_form)

            #homogeneouos transformation between frames 5 and 6
            T_56 = self.get_transformation_matrix(joint_variables=joints_ik[hypothesis_index, :], start_joint=5, end_joint=5)

            #homogeneous transformation 6->5 to obtain desired point in the frame 6
            T_65 = np.linalg.inv(T_56)

            #homogeneouos transformation between frames 4 and 5
            T_45 = self.get_transformation_matrix(joint_variables=joints_ik[hypothesis_index, :], start_joint=4, end_joint=4)

            #homogeneous transformation 5->4 to obtain desired point in the frame 1
            T_54 = np.linalg.inv(T_45)

            # desired pose in frame 4
            T_14 = np.matmul(np.matmul(pose_full_form_frame1, T_65), T_54)

            # desired pose in frame 3
            P_13 = np.matmul(T_14,np.resize([0, -self.d[3], 0, 1], (4, 1)))[0:3]
        
            # theta 2
            joints_ik[hypothesis_index, 1] = -np.arctan2(P_13[1], -P_13[0]) + np.arcsin(self.a[2]*np.sin(joints_ik[hypothesis_index, 2])/np.linalg.norm(P_13))

            # theta 4
            #homogeneouos transformation between frames 2 and 3
            T_23 = self.get_transformation_matrix(joint_variables=joints_ik[hypothesis_index, :], start_joint=2, end_joint=2)

            #homogeneous transformation 3->2 to obtain desired point in the frame 3
            T_32 = np.linalg.inv(T_23)

            #homogeneouos transformation between frames 1 and 2
            T_12 = self.get_transformation_matrix(joint_variables=joints_ik[hypothesis_index, :], start_joint=1, end_joint=1)

            #homogeneous transformation 2->1 to obtain desired point in the frame 2
            T_21 = np.linalg.inv(T_12)

            T_34 = np.matmul(np.matmul(T_32, T_21), T_14)
            joints_ik[hypothesis_index, 3] = np.arctan2(T_34[1, 0], T_34[0, 0])

        #<-------------------
        #removes lines containing nan values
        joints_ik = joints_ik[~np.isnan(joints_ik).any(axis=1)]

        return joints_ik
    

    def RB10_forwKine(self, th_RB10_changed, c):
        # 회전 방향이 다르므로 1. 부호를 바꾸고, 2. config 자세 맞추기
        # 식: -(th_RB10_changed) + config = th_RB10
        th_RB10 = [[],[],[],[],[],[]]
        th_RB10[0] =  th_RB10_changed[c, 0] + self.config[0] 
        th_RB10[1] = -th_RB10_changed[c, 1] + self.config[1]
        th_RB10[2] = -th_RB10_changed[c, 2] + self.config[2]
        th_RB10[3] = -th_RB10_changed[c, 3] + self.config[3]
        th_RB10[4] =  th_RB10_changed[c, 4] + self.config[4]
        th_RB10[5] = -th_RB10_changed[c, 5] + self.config[5]
    
        # new_pose, new_ee_orientation, T_06 = kin_model.forward_kin(th_RB10)    # kinematics_utils 모듈 불러와서 실행
        new_pose, new_ee_orientation, T_06 = self.forward_kine(th_RB10)               # kinematics_utils 모듈을 내재화하여 실행
  
        theta_i = 0
        alpha_i = 0
        a_i = 0
        d_i = d0
        A_i = self.homogeneous_transformation_i(theta_i, alpha_i, a_i, d_i)
        T_G6 = np.matmul(A_i, T_06)
    
        theta_i = pi
        alpha_i = -pi/2
        a_i = 0
        d_i = 0
        A_i = self.homogeneous_transformation_i(theta_i, alpha_i, a_i, d_i)
        T_GE = np.matmul(T_G6, A_i)
   
        # print('FK Test - T_06 : ', T_06)
        # print('FK Test - T_0E : ', T_0E)
        # print('FK Test - T_GE : ', T_GE)
  
        return T_GE
        

    def homogeneous_transformation_i(self, theta_i, alpha_i, a_i, d_i):
        A_i = np.array([np.cos(theta_i), -np.sin(theta_i)*np.cos(alpha_i),  np.sin(theta_i)*np.sin(alpha_i), a_i*np.cos(theta_i), \
                        np.sin(theta_i),  np.cos(theta_i)*np.cos(alpha_i), -np.cos(theta_i)*np.sin(alpha_i), a_i*np.sin(theta_i), \
                        0              ,                  np.sin(alpha_i),                  np.cos(alpha_i), d_i                , \
                        0              ,                                0,                                0, 1                    ], dtype=np.float32)
        A_i = np.reshape(A_i, (4, 4))
        return A_i

    def homogeneous_transformation_i_var_theta(self, i, theta_i):
        return self.homogeneous_transformation_i(theta_i=theta_i, alpha_i=self.alpha[i], a_i=self.a[i], d_i=self.d[i])

    def homogeneous_transformation_i_var_alpha(self, i, alpha_i):
        return self.homogeneous_transformation_i(theta_i=self.theta[i], alpha_i=alpha_i, a_i=self.a[i], d_i=self.d[i])

    def homogeneous_transformation_i_var_a(self, i, a_i):
        return self.homogeneous_transformation_i(theta_i=self.theta[i], alpha_i=self.alpha[i], a_i=a_i, d_i=self.d[i])

    def homogeneous_transformation_i_var_d(self, i, d_i):
        return self.homogeneous_transformation_i(theta_i=self.theta[i], alpha_i=self.alpha[i], a_i=self.a[i], d_i=d_i)

    def get_transformation_matrix(self, joint_variables, start_joint, end_joint):
        T_matrix = np.eye(4)
        assert end_joint >= start_joint
        joint_index = start_joint
        
        while joint_index <= end_joint:
            theta_i = joint_variables[joint_index]
            A_i = self.homogeneous_transformation_i_var_theta(joint_index, theta_i)
            T_matrix = np.matmul(T_matrix, A_i)
            joint_index += 1

        return T_matrix

    def forward_kine(self, joint_variables):
        T_matrix = self.get_transformation_matrix(joint_variables=joint_variables, start_joint=0, end_joint=self.number_of_joints-1)
        pose = np.array(T_matrix[0:3, -1])
        ee_orientation = np.array(T_matrix[0:3, 0:3]) 

        return pose, ee_orientation, T_matrix
'''

    
'''
class model:
    def __init__(self):
        self.jtype = [1, 1, 1, 1, 1, 1]
        self.parent = [None, 0, 1, 2, 3, 4]
        self.active = [1, 1, 1, 1, 1, 1]
        self.m = np.array([2.0, 2.0, 4.0, 3.0, 1.0, 1.0])
        self.dt = 0.001
        
        self.I = np.zeros((6, 3, 3))
        self.I[0] = tact.Isph(self.m[0], 0.05)
        self.I[1] = tact.Isph(self.m[1], 0.05)
        self.I[2] = tact.Isph(self.m[2], 0.05)
        self.I[3] = tact.Isph(self.m[3], 0.05)
        self.I[4] = tact.Isph(self.m[4], 0.05)
        self.I[5] = tact.Isph(self.m[5], 0.05)
        
        self.Ti = np.zeros((6, 4, 4))
        self.Ti[0] = tact.T_trans([0, 0, 0.1970])
        self.Ti[1] = tact.T_trans([0, -0.1875, 0]) @ tact.T_rot_x(-np.pi/2) 
        self.Ti[2] = tact.T_trans([0, -0.6127, 0])
        self.Ti[3] = tact.T_trans([0, -0.5702, 0.1514])
        self.Ti[4] = tact.T_trans([0, 0, -0.1172]) @ tact.T_rot_x(np.pi/2)
        self.Ti[5] = tact.T_trans([0, 0, 0.1172]) @ tact.T_rot_x(-np.pi/2)
                
        self.Bi = np.tile(np.eye(4), (6, 1, 1))
        self.Bi[0] = tact.T_trans([0, -0.0937, 0])
        self.Bi[1] = tact.T_trans([0, -0.3063, 0])
        self.Bi[2] = tact.T_trans([0, -0.2851, 0])
        self.Bi[3] = tact.T_trans([0, 0, -0.1172])
        self.Bi[4] = tact.T_trans([0, 0, 0.1172])
        self.Bi[5] = tact.T_trans([0, 0, -0.10]) # tool case
        
        self.X = tact.get_spatial_transform(self.Ti)
        self.I6 = tact.get_spatial_inertia(self.Bi, self.m, self.I)
        
        self.EB = tact.T_trans([0, 0, 0.1514]) #elbow point in T[2]
        #self.EE = tact.T_trans([0, 0, -0.0947]) @ tact.T_rot_x(np.pi/2) #end-effector in T[5] : tool-tip
        self.EE = tact.T_trans([0, 0, -0.1947]) @ tact.T_rot_x(np.pi/2) #end-effector in T[5] : end-sphere
        self.EI = tact.get_spatial_inertia0(tact.T_trans([0, 0, -0.05]), 0.1, tact.Isph(0.1, 0.05)) # end inertia in T[5]

        #contact body
        self.CB = []
        self.CB.append([101, -1, tact.T_trans([0, 0, -0.05]), [1.2, 1.2, 0.1], [50000, 100, 50]]) #floor
        #self.CB.append([102,  5, self.EE, [0.05], [50000, 100, 100]])       #sphere tool tip
        #self.CB.append([103,  5, self.EE, [0.05, 0.20], [50000, 100, 100]]) #cylinder tool tip
        #self.CB.append([104,  5, self.EE, [0.05, 0.20], [50000, 100, 100]]) #capsule tool tip
        self.CB.append([100,  5, np.eye(4), [6], [30000, 10, 10]])              #mesh tool tip
        
        self.view = [20, 45]
        self.lim = [[-0.75, 0.75], [-0.75, 0.75], [-0.3, 1.2]]
        self.vtype = [2, 2, 2, 2, 2, 2,  1, 1, 1, 1, 1, 1, 1, 1, 'x', self.CB[0][0], self.CB[1][0]]
        
        #joint friction factors
        self.ff = np.array([0.3, 0.3, 0.2, 0.2, 0.1, 0.1])

        self.q0 = np.array([0, 0, 0, 0, 0, 0])
        self.qd0 = np.zeros(6)
        self.tau0 = np.zeros(6)

        #analytic inverse kinematics module
        self.InverseKinematics = InverseKinematics()
        
        np.set_printoptions(precision=6)
        np.set_printoptions(suppress=True)

    def skeleton(self, q):
        T, B = tact._fk(self.Ti, self.Bi, self.parent, self.jtype, q)
        Z1 = np.concatenate(((T[0] @ tact.T_trans([0, 0, -0.05]))[:3, 3], (T[0] @ tact.T_trans([0, 0, 0.05]))[:3, 3]))
        Z2 = np.concatenate(((T[1] @ tact.T_trans([0, 0, -0.05]))[:3, 3], (T[1] @ tact.T_trans([0, 0, 0.05]))[:3, 3]))
        Z3 = np.concatenate(((T[2] @ tact.T_trans([0, 0, -0.05]))[:3, 3], (T[2] @ tact.T_trans([0, 0, 0.05]))[:3, 3]))
        Z4 = np.concatenate(((T[3] @ tact.T_trans([0, 0, -0.05]))[:3, 3], (T[3] @ tact.T_trans([0, 0, 0.05]))[:3, 3]))
        Z5 = np.concatenate(((T[4] @ tact.T_trans([0, 0, -0.05]))[:3, 3], (T[4] @ tact.T_trans([0, 0, 0.05]))[:3, 3]))
        Z6 = np.concatenate(((T[5] @ tact.T_trans([0, 0, -0.05]))[:3, 3], (T[5] @ tact.T_trans([0, 0, 0.05]))[:3, 3]))

        L0 = np.concatenate((np.zeros(3), T[0][:3, 3]))
        L1 = np.concatenate((T[0][:3, 3], T[1][:3, 3]))
        L2 = np.concatenate((T[1][:3, 3], T[2][:3, 3]))
        L3 = np.concatenate((T[2][:3, 3], (T[2] @ self.EB)[:3, 3]))
        L4 = np.concatenate(((T[2] @ self.EB)[:3, 3], T[3][:3, 3]))
        L5 = np.concatenate((T[3][:3, 3], T[4][:3, 3]))
        L6 = np.concatenate((T[4][:3, 3], T[5][:3, 3]))
        L7 = np.concatenate((T[5][:3, 3], (T[5] @ self.EE)[:3, 3]))
        OG = np.zeros(3) #origin

        FL = np.concatenate((tact.homogeneous_to_xyzeuler(self.CB[0][2]), self.CB[0][3]))
        EE = np.concatenate((tact.homogeneous_to_xyzeuler(T[5] @ self.CB[1][2]), self.CB[1][3]))
        return [Z1, Z2, Z3, Z4, Z5, Z6, L0, L1, L2, L3, L4, L5, L6, L7, OG, FL, EE]

    def step(self, q, qd, tau):
        T, B = tact._fk(self.Ti, self.Bi, self.parent, self.jtype, q)

        #get external force
        #f_ext = np.zeros((6, 6))
        #f_ext = tact.ground_sphere_contact2(T, self.CB, self.dt)
        #f_ext = tact.ground_sphere_contact3(T, self.parent, self.jtype, qd, self.CB)
        f_ext = tact.contact_force(T, self.parent, self.jtype, qd, self.CB)
        
        #add some friction
        tau = tau - self.ff * qd

        q_next, qd_next, qdd = tact.euler_step(self.X, self.I6, self.parent, self.jtype, q, qd, tau, f_ext, [0, 0, -9.81], self.dt)
        #q_next, qd_next, qdd = tact.rk4_step(self.X, self.I6, self.parent, self.jtype, q, qd, tau, f_ext, [0, 0, -9.81], self.dt)
        
        #get end-effector force sensor
        #f = self.EI @ a[5] + tact.crf(v[5]) @ self.EI @ v[5]
        y = np.concatenate((q_next[:6], qd_next[:6], f_ext[5]))
        return q_next, qd_next, y

    def fk(self, th):
        T, B = tact._fk(self.Ti[:6], self.Bi[:6], self.parent[:6], self.jtype[:6], th)
        EE = T[5] @ self.EE
        return tact.homogeneous_to_xyzeuler(EE)

    def jacob(self, th):
        T, B = tact._fk(self.Ti[:6], self.Bi[:6], self.parent[:6], self.jtype[:6], th)
        EE = T[5] @ self.EE
        J = tact.jacob_whitney(T, EE, self.parent[:6], self.jtype[:6], 5)
        return J

    def gravity(self, th):
        T, B = tact._fk(self.Ti[:6], self.Bi[:6], self.parent[:6], self.jtype[:6], th)
        tau = tact.gravity_lagrange(T, B, self.parent[:6], self.jtype[:6], self.m[:6])
        return -tau

    def inertia(self, th):
        M = tact.crb_featherstone(self.X[:6], self.I6[:6], self.parent[:6], self.jtype[:6], th)
        return M
       
    def bias(self, th, thd):
        thdd = np.zeros(6)
        b = tact.rne_featherstone(self.X[:6], self.I6[:6], self.parent[:6], self.jtype[:6], th, thd, thdd)
        return b

    def error(self, th, x_d):
        T, B = tact._fk(self.Ti[:6], self.Bi[:6], self.parent[:6], self.jtype[:6], th)
        EE = T[5] @ self.EE
        EE_d = tact.xyzeuler_to_homogeneous(x_d)
        return tact.homogeneous_error(EE_d, EE)

    def ik(self, x, q):
        S = self.InverseKinematics.RB10_invKine(x[0], x[1], x[2], x[3], x[4], x[5])[0]
        if len(S) == 0: print('no IK solution for', x); exit(0)
        E = np.zeros(len(S))
        
        for i in range(len(S)):
            for j in range(6):
                E[i] += abs(S[i][j]-q[j])

            #avoid abnormal pose
            if S[i][2] < 0: E[i] += 100

        idx = np.argmin(E)
        return S[idx]         
'''

    
'''    
class mlib:
    def __init__(self, path):
        self.clib = ctypes.CDLL(path)
        self.InverseKinematics = InverseKinematics()
        
    def fk(self, q):
        x = np.zeros(6)
        _q = (ctypes.c_double*len(q))(*q)
        _x = (ctypes.c_double*len(x))(*x)
        self.clib.fk(_q, _x)
        x = np.array(_x)
        return x

    def error(self, q, x_d):
        e = np.zeros(6)
        _q = (ctypes.c_double*len(q))(*q)
        _x_d = (ctypes.c_double*len(x_d))(*x_d)
        _e = (ctypes.c_double*len(e))(*e)
        self.clib.error(_q, _x_d, _e)
        e = np.array(_e)
        return e

    def jacob(self, q):
        J = np.zeros(6*6)
        _q = (ctypes.c_double*len(q))(*q)
        _J = (ctypes.c_double*len(J))(*J)
        self.clib.jacob(_q, _J)
        J = np.array(_J).reshape((6, 6))
        return J

    def gravity(self, q, m, l):
        tau = np.zeros(6)
        _q = (ctypes.c_double*len(q))(*q)
        _u = (ctypes.c_double*len(tau))(*tau)
        _m = (ctypes.c_double)(m)
        _l = (ctypes.c_double)(l)
        self.clib.gravity(_q, _u, _m, _l)
        tau = np.array(_u)
        return -tau


    def ik(self, x, q):
        #S = ik_rb10.RB5_invKine(x[0], x[1], x[2], x[3], x[4], x[5])[0]
        S = self.InverseKinematics.RB10_invKine(x[0], x[1], x[2], x[3], x[4], x[5])[0]

        if len(S) == 0:
            print('no IK solution for', x)
            exit(0)
    
        E = np.zeros(len(S))
        
        for i in range(len(S)):
            for j in range(6):
                E[i] += abs(S[i][j]-q[j])

            #avoid abnormal pose
            if S[i][2] < 0: E[i] += 100

        idx = np.argmin(E)
        return S[idx]
        
    #def ik_N(self, x):
    #    #if m == 'rb5': S =  rb5ik.RB5_invKine(x[0], x[1], x[2], x[3], x[4], x[5])[0]
    #    #if m == 'rb10':
    #    S = ik_rb10.RB5_invKine(x[0], x[1], x[2], x[3], x[4], x[5])[0]
    #    return len(S)

'''
