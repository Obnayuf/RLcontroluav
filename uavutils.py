import numpy as np
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
import time
from tensorboardX import SummaryWriter

class Recording(object):
    def __init__(self):
        self.reporter = SummaryWriter("../logs")
    def add(self,name,loss,step):
        self.reporter.add_scalar(name,loss,step)

def actionmapping(action, action_max, action_min):#从（-1，1）折射成期望动作
    K = (action_max - action_min) / 2.
    b = (action_max + action_min) / 2.
    return action * K + b


def statemapping(state, state_max, state_min):#归一到（0，1）
    K = 1. / (state_max - state_min)
    b = -state_min / (state_max - state_min)
    return state * K + b


def save_figure(total_time, stepid, state_buffer,episode,reward):
    label = ['Ub', 'Vb', 'Wb', 'p', 'q', 'r', 'Xe', 'Ye', 'Ze', 'PITCH', 'ROLL', 'YAW']
    t = np.linspace(0, total_time + 0.02, stepid)
    data = list(map(list, zip(*state_buffer)))
    plt.subplots(constrained_layout=True)
    fig, ax = plt.subplots(4, 3)
    for i in range(12):
        plt.subplot(4, 3, i + 1)
        plt.plot(t, data[i])
        plt.title(label[i])
    plt.tight_layout()
    plt.savefig('{}episode_reward_{}.jpg'.format(episode,reward), bbox_inches='tight')




