import random
import numpy as np


def gen_start():
    Xwidth = 1000  # cm
    Vmax = 10  # cm/s
    Amax = 1  # cm/s^2
    Smax = 0.1  # cm/s^3
    Wmax = 0.5  # rad/s
    Batemax = 1  # rad/s^2
    # 初始位置
    x0 = np.random.rand(3)
    x0 = Xwidth * (x0 - 0.5)
    # 初速度
    v0 = np.random.rand(3)
    v0 = Vmax * (x0 - 0.5)
    # 初加速度
    a0 = np.random.rand(3)
    a0 = Amax * (x0 - 0.5)
    # 初惯动度
    s0 = np.random.rand(3)
    s0 = Smax * (x0 - 0.5)
    # 初始yaw
    yaw = np.random.rand(1)
    # 初角速度
    w = np.random.rand(3)
    w = Wmax * (x0 - 0.5)
    # 初角加
    bate = np.random.rand(1)

    state = np.random.rand(15)
    state = np.hstack((state, np.array([1])))
    print(state)
    state.resize(16, 1)
    H = np.diag([Xwidth, Xwidth, Xwidth, Vmax, Vmax, Vmax, Amax, Amax, Amax, Smax, Smax, Smax, 1, Wmax, 1])
    H = np.hstack((H, np.array([[-0.5*Xwidth], [-0.5*Xwidth], [-0.5*Xwidth],
                                [-0.5*Vmax], [-0.5*Vmax], [-0.5*Vmax],
                                [-0.5*Amax], [-0.5*Amax], [-0.5*Amax],
                                [-0.5*Smax], [-0.5*Smax], [-0.5*Smax],
                                [0], [-0.5*Wmax], [0]])))
    print(H)
    state = H @ state
    return state


print(gen_start())
