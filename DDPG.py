import random
import collections
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import physics_sim
torch.manual_seed(0)
np.random.seed(0)
import uavutils
# Hyperparameters
lr_mu = 0.0001  # actor
lr_q = 0.001  # critic
gamma = 0.99
batch_size = 64
buffer_limit = 100000
tau = 0.001  # for target network soft update


class ReplayBuffer():
    def __init__(self):
        self.buffer = collections.deque(maxlen=buffer_limit)

    def put(self, transition):
        self.buffer.append(transition)

    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []

        for transition in mini_batch:
            s, a, r, s_prime, done = transition
            s_lst.append(s)
            a_lst.append(a)
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask = 0.0 if done else 1.0
            done_mask_lst.append([done_mask])

        return torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst, dtype=torch.float), \
               torch.tensor(r_lst, dtype=torch.float), torch.tensor(s_prime_lst, dtype=torch.float), \
               torch.tensor(done_mask_lst, dtype=torch.float)

    def size(self):
        return len(self.buffer)
    def pop(self):
        self.buffer.popleft()


class MuNet(nn.Module):  # Actor
    def __init__(self):
        super(MuNet, self).__init__()
        self.fc1 = nn.Linear(12, 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc_mu = nn.Linear(300, 5)  # 5个动作两个隐藏层

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = torch.tanh(self.fc_mu(x))  # 限制在[-1,1]
        return mu

    def initialize(self):  # 将最后一层初始化权重
        nn.init.uniform_(self.fc1.weight.data, -1./np.sqrt(12), 1./np.sqrt(12))
        nn.init.uniform_(self.fc1.bias.data,-1./np.sqrt(12), 1./np.sqrt(12))
        nn.init.uniform_(self.fc2.weight.data, -1./20., 1./20.)
        nn.init.uniform_(self.fc2.bias.data,-1./20., 1./20.)
        nn.init.uniform_(self.fc_mu.weight.data, -0.003, 0.003)
        nn.init.uniform_(self.fc_mu.bias.data, -0.003, 0.003)


class QNet(nn.Module):
    def __init__(self):
        super(QNet, self).__init__()
        self.fc_s = nn.Linear(12, 200)
        self.fc_a = nn.Linear(5, 200)
        self.fc_q = nn.Linear(400, 300)
        self.fc_out = nn.Linear(300, 1)

    def forward(self, x, a):
        h1 = F.relu(self.fc_s(x))
        h2 = F.relu(self.fc_a(a))
        cat = torch.cat([h1, h2], dim=1)
        q = F.relu(self.fc_q(cat))
        q = self.fc_out(q)
        return q

    def initialize(self):  # 将最后一层初始化权重
        nn.init.uniform_(self.fc_s.weight.data, -1./np.sqrt(12), 1./np.sqrt(12))
        nn.init.uniform_(self.fc_s.bias.data,-1./np.sqrt(12), 1./np.sqrt(12))
        nn.init.uniform_(self.fc_a.weight.data, -1./np.sqrt(5), 1./np.sqrt(5))
        nn.init.uniform_(self.fc_a.bias.data,-1./np.sqrt(5), 1./np.sqrt(5))
        nn.init.uniform_(self.fc_q.weight.data, -1./20., 1./20.)
        nn.init.uniform_(self.fc_q.bias.data,-1./20., 1./20.)
        nn.init.uniform_(self.fc_out.weight.data, -0.003, 0.003)
        nn.init.uniform_(self.fc_out.bias.data, -0.003, 0.003)


class OrnsteinUhlenbeckNoise:  # dt是我们的步长
    def __init__(self, mu):
        self.theta, self.dt, self.sigma = 0.15, 0.02, 0.2
        self.mu = mu
        self.x_prev = np.zeros_like(self.mu)

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x


def train(mu, mu_target, q, q_target, memory, q_optimizer, mu_optimizer,reporter,timestep):
    s, a, r, s_prime, done_mask = memory.sample(batch_size)
    target = r + gamma * q_target(s_prime, mu_target(s_prime)) * done_mask
    q_loss = F.mse_loss(q(s, a), target.detach())
    q_optimizer.zero_grad()
    q_loss.backward()
    q_optimizer.step()

    mu_loss = -q(s, mu(s)).mean()  # That's all for the policy loss.
    mu_optimizer.zero_grad()
    mu_loss.backward()
    mu_optimizer.step()
    reporter.add("qloss",q_loss,timestep)
    reporter.add("actor_loss", mu_loss, timestep)

def soft_update(net, net_target):
    for param_target, param in zip(net_target.parameters(), net.parameters()):
        param_target.data.copy_(param_target.data * (1.0 - tau) + param.data * tau)


def main():
    env = physics_sim.PhysicsSim()
    memory = ReplayBuffer()

    q, q_target = QNet(), QNet()
    q.initialize()
    q_target.load_state_dict(q.state_dict())
    mu, mu_target = MuNet(), MuNet()
    mu.initialize()
    mu_target.load_state_dict(mu.state_dict())

    score = 0.0
    print_interval = 20
    reporter = uavutils.Recording()
    mu_optimizer = optim.Adam(mu.parameters(), lr=lr_mu)
    q_optimizer = optim.Adam(q.parameters(), lr=lr_q, weight_decay=0.02)
    ou_noise = OrnsteinUhlenbeckNoise(mu=np.zeros(5))

    for n_epi in range(20000):
        s = env.reset()
        done = False
        reward = 0
        while not done:
            s = uavutils.statemapping(s, env.high_state, env.low_state)
            a = mu(torch.from_numpy(s).float())
            a = a.detach().numpy() + ou_noise()
            a = uavutils.actionmapping(a, env.max_action, env.min_action)
            s_prime, r, done, info = env.step(a)
            if memory.size()== buffer_limit:
                memory.pop()
            memory.put((s, a, r / 100.0, s_prime, done))#对于奖励进行放缩
            score += r
            reward += r
            s = s_prime
        reporter.add("reward",reward,n_epi)
        reporter.add("maxstep",env.stepid,n_epi)
        if memory.size() > 2000:
            for i in range(10):#训练十次
                train(mu, mu_target, q, q_target, memory, q_optimizer, mu_optimizer,reporter,n_epi+i)
                soft_update(mu, mu_target)
                soft_update(q, q_target)
        if env.already_landing:
            uavutils.save_figure(env.time, env.stepid, env.statebuffer, n_epi, reward)
            torch.save(
                {'epoch': n_epi + 1, 'Critic_dict': q.state_dict(), 'Actor_dict': mu.state_dict(),
                 'optimizer_a': mu.optimizer_a.state_dict(), 'optimizer_c': q.optimizer_c.state_dict()},
                'EP{}reward{}'.format(n_epi, reward) + '.pth.tar')
        if n_epi % print_interval == 0 and n_epi != 0:
            print("# of episode :{}, avg score : {:.1f}".format(n_epi, score / print_interval))
            score = 0.0



if __name__ == '__main__':
    main()
