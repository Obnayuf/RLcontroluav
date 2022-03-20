import numpy as np
from collections import namedtuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from statistics import mean
import physics_sim
import uavutils
from tensorboardX import SummaryWriter
torch.manual_seed(0)
np.random.seed(0)
Transition = namedtuple('Transition', ['s', 'a', 'a_log_p', 'r', 's_'])
gamma=0.99
beta = 0.01

class ActorNet(nn.Module):

    def __init__(self):
        super(ActorNet, self).__init__()
        self.fc = nn.Linear(12, 256)
        self.fc1 = nn.Linear(256,128)
        self.mu_head = nn.Linear(128, 5)
        self.sigma_head = nn.Linear(128, 5)

    def forward(self, x):
        x = F.relu(self.fc(x))
        x = F.relu(self.fc1(x))
        mu = torch.tanh(self.mu_head(x))
        sigma = F.sigmoid(self.sigma_head(x))
        return mu, sigma

class CriticNet(nn.Module):

    def __init__(self):
        super(CriticNet, self).__init__()
        self.fc = nn.Linear(12, 256)
        self.fc1 = nn.Linear(256, 128)
        self.v_head = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.fc(x))
        x = F.relu(self.fc1(x))
        state_value = self.v_head(x)
        return state_value

class PPO():

    clip_param = 0.2
    max_grad_norm = 0.5
    ppo_epoch = 30
    buffer_capacity, batch_size = 10000, 32

    def __init__(self):
        self.training_step = 0
        self.anet = ActorNet().float()
        self.cnet = CriticNet().float()
        self.buffer = []
        self.counter = 0
        self.reporter = SummaryWriter()
        self.optimizer_a = optim.Adam(self.anet.parameters(), lr=1e-4)
        self.optimizer_c = optim.Adam(self.cnet.parameters(), lr=3e-4)

    def select_action(self, state):
        action_list=[]
        action_log_prob_list = []
        state = torch.from_numpy(state).float()#.unsqueeze(0)#增加一个维度，方便
        with torch.no_grad():
            (mu, sigma) = self.anet(state)
        for mu_,sigma_ in zip(mu,sigma):
            dist = Normal(mu_,sigma_)
            action = dist.sample()
            action_log_prob = dist.log_prob(action)
            action = action.clamp(-1.0,1.0)
            action_list.append(action.item())
            action_log_prob_list.append(action_log_prob.item())
        return np.array(action_list),np.mean(action_log_prob_list)

    def store(self, transition):
        self.buffer.append(transition)
        self.counter += 1
        return self.counter % self.buffer_capacity == 0

    def update(self):
        self.training_step += 1
        s = torch.tensor([t.s for t in self.buffer], dtype=torch.float)
        a = torch.tensor([t.a for t in self.buffer], dtype=torch.float)
        r = torch.tensor([t.r for t in self.buffer], dtype=torch.float).view(-1, 1)
        s_ = torch.tensor([t.s_ for t in self.buffer], dtype=torch.float)
        old_action_log_probs = torch.tensor(
            [t.a_log_p for t in self.buffer], dtype=torch.float)

        with torch.no_grad():
            target_v = r + gamma * self.cnet(s_)

        adv = (target_v - self.cnet(s)).detach()

        for _ in range(self.ppo_epoch):
            for index in BatchSampler(
                    SubsetRandomSampler(range(self.buffer_capacity)), self.batch_size, False):#从容量1000中随机抽取32个step去训练，一共32次，32*32=1024

                prob_ac = []
                (mu, sigma) = self.anet(s[index])
                dist = Normal(mu,sigma)
                entropy = dist.entropy().mean(axis=1)#哪一维度希望为1哪一个维度是1
                action_log_prob = dist.log_prob(a[index]).mean(axis=1)
                ratio = torch.exp(action_log_prob - (old_action_log_probs[index]))
                surr1 = ratio * adv[index]
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param,
                                    1.0 + self.clip_param) * adv[index]
                entro = entropy.mean()
                action_loss = -torch.min(surr1, surr2).mean()#index是一个1*32的数组，因此我们每从1000个步长里抽取32个，进行一次更新，更新32次
                total_loss = action_loss-beta*entro
                self.optimizer_a.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(self.anet.parameters(), self.max_grad_norm)#防止梯度爆炸
                self.optimizer_a.step()

                value_loss = F.mse_loss(self.cnet(s[index]), target_v[index])
                self.optimizer_c.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.cnet.parameters(), self.max_grad_norm)
                self.optimizer_c.step()
                self.reporter.add_scalar('value_loss',value_loss.item(),global_step=self.training_step)
                self.reporter.add_scalar('actor_loss',action_loss.item(),global_step=self.training_step)
                self.reporter.add_scalar('entropy', entro.item(),global_step=self.training_step)
                self.reporter.add_scalar('total_loss', total_loss.item(),global_step=self.training_step)
        del self.buffer[:]#清空buffer中的数据


def main():

    env = physics_sim.PhysicsSim()
    agent = PPO()
    score = 0
    for i_ep in range(100000):
        state = env.reset()
        done = False
        standard_reward=0
        while not done:
            action, action_log_prob = agent.select_action(uavutils.statemapping(state,env.high_state,env.low_state))
            action=uavutils.actionmapping(action,action_max=env.max_action,action_min=env.min_action)#动作映射
            state_, reward, done, _ = env.step(action)
            if agent.store(Transition(state, action, action_log_prob, reward, uavutils.statemapping(state_,env.high_state,env.low_state))):
                agent.update()
            score += reward
            standard_reward += reward
            state = state_
        agent.reporter.add_scalar('reward',standard_reward, i_ep)
        agent.reporter.add_scalar('max_step',env.stepid, i_ep)
        if env.already_landing:
            uavutils.save_figure(env.time,env.stepid,env.statebuffer,i_ep,standard_reward)
            torch.save({'epoch': i_ep + 1, 'Critic_dict': agent.cnet.state_dict(),'Actor_dict': agent.anet.state_dict(),
                        'optimizer_a': agent.optimizer_a.state_dict(),'optimizer_c': agent.optimizer_c.state_dict()},
                       'EP{}reward{}'.format(i_ep,standard_reward)+'.pth.tar')

        if i_ep % 20 == 0:
            print('Ep {}\t average score: {:.2f}\t'.format(i_ep, score/20))
            #uavutils.save_figure(env.time, env.stepid, env.statebuffer, i_ep, standard_reward)
            score = 0.0




if __name__ == '__main__':
    main()