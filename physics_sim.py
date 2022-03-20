import math
from math import atan2

import numpy as np
from scipy.interpolate import interp1d
from gym import spaces

eps = 1e-10


def C(x):
    return np.cos(x)


def S(x):
    return np.sin(x)


def body_to_earth_frame(PHI, THETA, PSI):
    # C^b_n
    R = [[C(PSI) * C(THETA) - S(PSI) * S(PHI) * S(THETA), S(PSI) * C(PHI),
          C(PSI) * S(THETA) + S(PSI) * S(PHI) * C(THETA)],
         [S(PSI) * C(THETA) + C(PSI) * S(PHI) * S(THETA), C(PSI) * C(PHI),
          S(PHI) * S(THETA) - C(PSI) * S(PHI) * C(THETA)],
         [-C(PHI) * S(THETA), S(PHI), C(PHI) * C(THETA)]]
    return np.array(R)


def euler_angular_v(PHI, THETA, PSI):
    Q = [[C(THETA), S(THETA) * np.tan(PHI), -S(THETA) / C(PHI)],
         [0, 1, 0],
         [S(THETA), -C(THETA) * np.tan(PHI), C(THETA) / C(PHI)]]
    return np.array(Q)


def earth_to_body_frame(ii, jj, kk):
    # C^n_b
    return np.transpose(body_to_earth_frame(ii, jj, kk))


def Constrain(x, x_min, x_max):
    if x > x_max:
        return x_max
    elif x < x_min:
        return x_min
    else:
        return x


def angle_correct(angle):
    return ((angle + np.pi) % (2 * np.pi)) - np.pi


class PhysicsSim(object):
    '''单涵道无人机垂直起降物理模型，参照了四旋翼的环境搭建'''

    def __init__(self, init_pose=None, init_V=None, init_AngRates=None, runtime=200.):
        self.init_pose = init_pose
        self.init_V = init_V
        self.init_AngRates = init_AngRates
        self.runtime = runtime
        self.speed_min = 0
        self.speed_max = 1225 * 1.732
        self.actuator_min = -0.3491
        self.actuator_max = 0.3491
        self.min_action = np.array(
            [self.speed_min, self.actuator_min, self.actuator_min, self.actuator_min, self.actuator_min])
        self.max_action = np.array(
            [self.speed_max, self.actuator_max, self.actuator_max, self.actuator_max, self.actuator_max])
        self.action_space = spaces.Box(self.min_action, self.max_action, dtype=np.float64)
        self.low_state = np.array(
            [-40, -40, -40, -2 * np.pi, -2 * np.pi, -2 * np.pi, -50, -50, -50, -np.pi, -np.pi / 2., -np.pi]
        )
        self.high_state = -self.low_state
        self.observation_space = spaces.Box(
            low=self.low_state, high=self.high_state, dtype=np.float64
        )
        self.mass = 1.5300
        self.g = 9.788
        self.Ix = 0.025483
        self.Iy = 0.025504
        self.Iz = 0.00562
        self.dt = 0.02
        self.disturbance_x = 0.0
        self.disturbance_y = 0.0
        self.disturbance_z = 0.0
        self.r_sm_X = np.array(
            [3.141592653000000, 1.570796327000000, 1.308996939000000, 1.047197551000000, 0.785398163000000,
             0.523598776000000, 0.261799388000000, 0.0])
        self.r_sm_Y = np.array([0.0, 0.0, 0.0, 0.0, 0.124000000000000, 0.330000000000000, 0.510000000000000, 1.0])
        self.k_rs_X = np.array(
            [3.1416, 2.8798, 2.6180, 2.3562, 2.0944, 1.8326, 1.5708, 1.3090, 1.0472, 0.7854, 0.5236, 0.2618, 0.0])
        self.k_rs_Y = np.array(
            [0.0, -0.6125, -1.2580, -1.6410, -2.2340, -2.7340, -3.8950, -2.7340, -2.2340, -1.6410, -1.2580, -0.6125,
             0.0])
        self.r_m_Y = np.array([1, 0.927000000000000, 0.794100000000000, 0.632600000000000, 0.479900000000000,
                               0.393300000000000, 0.369000000000000, 0.352700000000000])
        self.k_ra_X = np.array(
            [3.141592653000000, 1.570796327000000, 1.308996939000000, 1.047197551000000, 0.785398163000000,
             0.523598776000000, 0.261799388000000, 0])
        self.k_ra_Y = np.array(
            [0, 0.175500000000000, 0.260000000000000, 0.350000000000000, 0.436000000000000, 0.550000000000000,
             0.610100000000000, 0.650000000000000])
        self.k_as_X = np.array(
            [0, 0.0873, 0.1745, 0.2618, 0.3491, 0.4363, 0.5236, 0.6109, 0.6981, 0.7854, 0.8727, 1.0472,
             1.2217, 1.3090, 1.3963, 1.5708, 1.7453, 1.8326, 1.9199, 2.0944, 2.2689, 2.3562, 2.4435, 2.5307, 2.6180,
             2.7053, 2.7925, 2.8798, 2.9671, 3.0543, 3.1416])
        self.k_as_Y = np.array(
            [0, -0.0080, -0.0155, -0.0224, -0.0285, -0.0327, -0.0366, -0.0399, -0.0390, -0.0391, -0.0371, -0.0364,
             -0.0328, -0.0282, -0.0244, -0.0178, -0.0159, -0.0174, -0.0243, -0.0264, -0.0266, -0.0261, -0.0238, -0.0228,
             -0.0207, -0.0189, -0.0173, -0.0127, -0.0079, -0.0048, 0])
        self.k_ac_X = np.array([0., 0.087266, 0.174533, 0.261799, 0.349066, 0.436332, 0.523599, 0.610865,
                                0.698132, 0.785398, 0.872665, 1.047198, 1.22173, 1.308997, 1.396263, 1.570796,
                                1.745329, 1.832596, 1.919862, 2.094395, 2.268928, 2.356194, 2.443461, 2.530727,
                                2.617994, 2.70526, 2.792527, 2.879793, 2.96706, 3.054326, 3.141593])
        self.k_ac_Y = np.array([-4.550e-03, -4.410e-03, -3.840e-03, -3.000e-03, -2.500e-03, -1.300e-03,
                                -8.700e-05, 1.889e-03, 1.350e-03, 1.280e-04, -1.120e-03, -2.610e-03,
                                2.320e-04, 2.797e-03, 4.117e-03, 4.427e-03, 4.912e-03, 5.440e-03,
                                3.134e-03, 7.560e-03, 9.433e-03, 9.381e-03, 9.806e-03, 9.452e-03,
                                9.045e-03, 8.363e-03, 6.792e-03, 7.613e-03, 7.297e-03, 6.924e-03,
                                7.066e-03])
        self.statebuffer = []
        self.already_crash = False
        self.already_landing = False
        self.prev_state = None

    def get_time(self):
        return self.time

    def reset(self):
        self.time = 0.0
        self.pose = np.array([0.0, 0.0, 30.0, 0.0, 0.0, 0.0]) if self.init_pose is None else np.copy(
            self.init_pose)  # 初始位置即X,Y,Z,PITCH,ROLL,YAW
        self.vb = np.array([0.0, 0.0, 0.0]) if self.init_V is None else np.copy(self.init_V)  # 初始速度 机体坐标系
        self.angular_v = np.array([0.0, 0.0, 0.0]) if self.init_AngRates is None else np.copy(
            self.init_AngRates)  # 初始角速度 w=p,q,r
        self.done = False
        self.stepid = 0
        self.already_crash = False
        self.already_landing = False
        self.statebuffer = []
        x = np.concatenate((self.vb, self.angular_v, self.pose))
        return x

    def get_earth_V(self):
        return np.matmul(earth_to_body_frame(*list(self.pose[3:])), self.vb)

    def get_body_gravity(self):
        return np.matmul(earth_to_body_frame(*list(self.pose[3:])), np.array([0, 0, self.mass * self.g]))

    def calculate_alpha(self, Amplitude, Vc):
        if Amplitude < 1.0e-05:
            alpha = 0.0
        else:
            alpha = np.arccos(Vc / Amplitude)
        if alpha > 3.141:
            alpha = 3.14
        elif alpha < 0:
            alpha = 0.0
        return alpha

    def check_landing_success(self):
        Z_limit = 0.1
        V_limit = 1.5
        dis_limit = 4
        angle_limit = 10 / 180 * np.pi
        v_angle_limit = 10 / 180 * np.pi
        x, y, z, pitch, roll, _ = self.pose
        u , v , w=self.vb
        vpitch, vroll, vyaw = self.angular_v
        return True if z <= Z_limit and w < 0.5 and np.sqrt(u**2+v**2)<V_limit and np.sqrt(x ** 2 + y ** 2) < dis_limit and abs(
            pitch) < angle_limit and abs(roll) < angle_limit and abs(vpitch) < angle_limit and abs(
            vyaw) < v_angle_limit and abs(vyaw) < v_angle_limit else False

    def check_crash(self):
        Z_limit = 0.1
        V_limit = 50
        dis_limit = 4
        angle_limit = 10 / 180 * np.pi
        v_angle_limit = 10 / 180 * np.pi
        x, y, z, pitch, roll, _ = self.pose
        v_norm = np.linalg.norm(self.vb)
        vpitch, vroll, vyaw = self.angular_v
        crash = False
        if z >= 40:  # 高度上限
            crash = True
        if z <= Z_limit and v_norm >= V_limit:  # 速度限制
            crash = True
        if z <= Z_limit and np.sqrt(x ** 2 + y ** 2) >= dis_limit:  # 距离限制
            crash = True
        if z <= Z_limit and abs(pitch) >= angle_limit:  # 角度限制
            crash = True
        if z <= Z_limit and abs(roll) >= angle_limit:  # 角度限制
            crash = True
        if z <= Z_limit and abs(vyaw) >= v_angle_limit:
            crash = True
        if z <= Z_limit and abs(vroll) >= v_angle_limit:
            crash = True
        if z <= Z_limit and abs(vpitch) >= v_angle_limit:
            crash = True
        if roll >= np.pi / 2 or roll <= -np.pi / 2:
            crash = True
        return crash

    def calculate_reward(self):
        reward = 0
        x, y, z, pitch, roll, _ = self.pose
        u, v, w = self.vb
        p, q, r = self.angular_v
        reward += -0.1*(pitch**2+roll**2)
        reward += -0.1*np.sqrt(p**2+q**2+r**2)
        distnorm = np.sqrt(x**2+y**2+z**2)/34.
        vnorm = np.sqrt(u**2+v**2+w**2)/40
        St = 1 - (np.sqrt(distnorm)+np.sqrt(vnorm)/2)
        if self.prev_state is None:
            reward += St
        else:
            reward +=St-self.prev_state
        self.prev_state=St
        if self.already_crash:
            reward = -100
        if self.already_landing:
            reward += 500

        return reward

    def Calculate_Force(self, speed, c, state):  # 螺旋桨推力,speed是螺旋桨的转速
        f1 = interp1d(self.r_sm_X, self.r_sm_Y, kind='linear')
        f2 = interp1d(self.k_rs_X, self.k_rs_Y, kind='linear')
        f3 = interp1d(self.r_sm_X, self.r_m_Y, kind='linear')
        f4 = interp1d(self.k_ra_X, self.k_ra_Y, kind='linear')
        f5 = interp1d(self.k_as_X, self.k_as_Y, kind='linear')
        f6 = interp1d(self.k_ac_X, self.k_ac_Y, kind='linear')
        K_TS = 9.9796018325697625989171178675552e-6
        K_TV = -2.8620408163265306122448979591837e-4
        k_q0 = 0.290827
        k_q1 = -0.02182
        den = 1.225
        Sarea = 0.040828138126052952
        d_cs = 0.0149564
        c_b = -0.026179938
        d_MS = 1.1334291042e-7
        l_1 = 0.17078793
        l_2 = 0.06647954
        d_ds = -9.5568e-06
        I_prop = 0.000029
        u, v, w = state[0:3]
        p, q, _ = state[3:6]
        V_c = -(w - self.disturbance_z)
        if (w - self.disturbance_z) < 0:
            T = K_TS * (speed ** 2) - K_TV * (w - self.disturbance_z) * speed
            ratio = k_q1 * V_c + k_q0
        else:
            T = K_TS * (speed ** 2)
            ratio = k_q0
        V_i = -(w - self.disturbance_z) / (2 * (1 - ratio)) + np.sqrt(
            ((w - self.disturbance_z) / (2 * (1 - ratio))) ** 2 + T / (
                    2 * den * Sarea * (1 - ratio))) - V_c
        Amplitude = np.sqrt(
            (u - self.disturbance_x) ** 2 + (v - self.disturbance_y) ** 2 + (w - self.disturbance_z) ** 2)  # 来流速度
        Coupling = Amplitude / (Amplitude + V_i)
        tmp = np.sqrt((u - self.disturbance_x) ** 2 + (w - self.disturbance_z) ** 2)
        temp = np.sqrt((v - self.disturbance_y) ** 2 + (w - self.disturbance_z) ** 2)
        Coupling_x = tmp / (tmp + V_i)
        Coupling_y = temp / (temp + V_i)
        beta = np.arctan((v - self.disturbance_y) / (u - self.disturbance_x + eps))
        alpha = self.calculate_alpha(Amplitude, V_c)
        r_sm = f1(alpha)
        k_rs = f2(alpha)
        if (u - self.disturbance_x) < 0:
            Attenuation1 = Constrain(1 - k_rs * Coupling_x, 1 - r_sm, 1 + r_sm)
            Attenuation3 = Constrain(1 + k_rs * Coupling_x, 1 - r_sm, 1 + r_sm)
        else:
            Attenuation1 = Constrain(1 + k_rs * Coupling_x, 1 - r_sm, 1 + r_sm)
            Attenuation3 = Constrain(1 - k_rs * Coupling_x, 1 - r_sm, 1 + r_sm)
        if (v - self.disturbance_y) < 0:
            Attenuation2 = Constrain(1 - k_rs * Coupling_y, 1 - r_sm, 1 + r_sm)
            Attenuation4 = Constrain(1 + k_rs * Coupling_y, 1 - r_sm, 1 + r_sm)
        else:
            Attenuation2 = Constrain(1 + k_rs * Coupling_y, 1 - r_sm, 1 + r_sm)
            Attenuation4 = Constrain(1 - k_rs * Coupling_y, 1 - r_sm, 1 + r_sm)
        k_cs = ((V_c + V_i) ** 2) * np.array([[0, -Attenuation2 * d_cs, 0, Attenuation4 * d_cs],
                                              [Attenuation1 * d_cs, 0, -Attenuation3 * d_cs, 0],
                                              [0, 0, 0, 0]])
        D_cs = ((V_c + V_i) ** 2) * np.array([[-Attenuation1 * d_cs * l_1, 0, Attenuation3 * d_cs * l_1, 0],
                                              [0, -Attenuation2 * d_cs * l_1, 0, Attenuation4 * d_cs * l_1],
                                              [Attenuation1 * d_cs * l_2, Attenuation2 * d_cs * l_2,
                                               Attenuation3 * d_cs * l_2, Attenuation4 * d_cs * l_2]])
        r_m = f3(alpha)
        kya = f4(alpha)
        kas = f5(alpha)
        kac = f6(alpha)
        ya = kya * Coupling
        F_as = ya * (Amplitude ** 2) * kas  # 升力
        F_ac = ya * (Amplitude ** 2) * kac  # 阻力
        F_p = np.array([F_as * C(beta), F_as * S(beta), -F_ac])
        F_m = -r_m * den * Sarea * (V_c + V_i) * np.array([(u - self.disturbance_x), (v - self.disturbance_y), 0])
        F_cs = np.matmul(k_cs, (c + np.array([-c_b, c_b, c_b, -c_b])))  # 舵面力 驱动力
        F_T = np.array([0.0, 0.0, -T])  # 螺旋桨拉力 驱动力
        F = F_T + F_cs + F_p + F_m + self.get_body_gravity()  # 最终合力
        M_prop = np.array([0, 0, d_MS * (speed ** 2)])  # 螺旋桨力矩 驱动力
        M_cs = np.matmul(D_cs, c)  # 舵面力矩 驱动力
        M_ds = np.array([0, 0, (V_c + V_i) * speed * d_ds])
        M_gyro = I_prop * speed * np.array([-q, p, 0])
        M = M_prop + M_cs + M_ds + M_gyro  # 最终合力矩
        return F, M

    def update_state(self, state):
        self.vb = np.clip(state[0:3], [-50, -50, -40], [50, 50, 40])  # 速度限制在-50~50之间
        self.angular_v = np.clip(state[3:6], -2 * np.pi, 2 * np.pi)
        self.pose = np.array(
            [state[6], state[7], state[8], angle_correct(state[9]), angle_correct(state[10]), angle_correct(state[11])])
        return np.concatenate((self.vb, self.angular_v, self.pose))

    def stateDerivative(self, x, action):  # 六自由度方程
        Force, Moment = self.Calculate_Force(action[0], action[1:], x)
        Fx, Fy, Fz = Force
        Mx, My, Mz = Moment
        ub = x[0]
        vb = x[1]
        wb = x[2]
        p = x[3]
        q = x[4]
        r = x[5]
        # x,y,z是6，7，8
        phi = x[9]
        theta = x[10]
        psi = x[11]
        xdot = np.zeros(12)
        xdot[0] = Fx / self.mass + r * vb - q * wb  # = udot
        xdot[1] = Fy / self.mass - r * ub + p * wb  # = vdot
        xdot[2] = Fz / self.mass + q * ub - p * vb  # = wdot
        xdot[3] = 1 / self.Ix * (Mx + (self.Iy - self.Iz) * q * r)  # = pdot
        xdot[4] = 1 / self.Iy * (My + (self.Iz - self.Ix) * p * r)  # = qdot
        xdot[5] = 1 / self.Iz * (Mz + (self.Ix - self.Iy) * p * q)  # = rdot
        xdot[9], xdot[10], xdot[11] = np.matmul(euler_angular_v(phi, theta, psi), np.array([p, q, r]))
        xdot[6], xdot[7], xdot[8] = np.matmul(body_to_earth_frame(phi, theta, psi), np.array([ub, vb, wb]))
        xdot[8] = -xdot[8]  # z轴表示方便
        return xdot

    def step(self, action):
        x = np.concatenate((self.vb, self.angular_v, self.pose))
        K1 = self.stateDerivative(x, action)
        K2 = self.stateDerivative(x + K1 * self.dt / 2, action)
        K3 = self.stateDerivative(x + K2 * self.dt / 2, action)
        K4 = self.stateDerivative(x + K3 * self.dt, action)
        x_next = x + 1 / 6 * (K1 + 2 * K2 + 2 * K3 + K4) * self.dt
        # 更新状态
        x_next = self.update_state(x_next)
        self.time += self.dt
        self.already_landing = self.check_landing_success()
        self.already_crash = self.check_crash()
        reward = self.calculate_reward()
        if self.time > self.runtime or self.already_landing or self.already_crash:
            self.done = True
        else:
            self.done = False
        if (x_next < self.low_state).any():
            self.done = True
        if (x_next > self.high_state).any():
            self.done = True
        self.statebuffer.append(x_next)
        self.stepid += 1
        return x_next, reward, self.done, None
        # 根据x_next计算reward
