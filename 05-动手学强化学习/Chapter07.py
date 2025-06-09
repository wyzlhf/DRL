import collections
import random
from typing import Tuple, Dict, List
import warnings
import rl_utils
import gym
import torch
import torch.nn.functional as F
import numpy as np

from gym import Env
from numpy import array
from torch import device
from matplotlib import pyplot as plt
from torch import nn, optim, Tensor
from tqdm import tqdm

warnings.filterwarnings("ignore")


# class ReplayBuffer(object):
#     def __init__(self, capacity: int):
#         self.buffer = collections.deque(maxlen=capacity)
#
#     def add(self, state, action, reward, next_state, done):
#         self.buffer.append((state, action, reward, next_state, done))
#
#     def sample(self, batch_size: int):
#         transitions = random.sample(self.buffer, batch_size)
#         state, action, reward, next_state, done = zip(*transitions)
#         return np.array(state), action, reward, np.array(next_state), done
#
#     def size(self) -> int:
#         return len(self.buffer)
#
#
# class Qnet(nn.Module):
#     def __init__(self, state_dim, hidden_dim, action_dim):
#         super().__init__()
#         self.fc1 = nn.Linear(state_dim, hidden_dim)
#         self.fc2 = nn.Linear(hidden_dim, action_dim)
#
#     def forward(self, x):
#         x = F.relu(self.fc1(x))
#         return self.fc2(x)
#
#
# class DQN(object):
#     def __init__(self, state_dim, hidden_dim, action_dim, learning_rate, gamma, epsilon, target_update, device):
#         self.action_dim = action_dim
#         self.q_net = Qnet(state_dim, hidden_dim, self.action_dim).to(device)
#         self.target_q_net = Qnet(state_dim, hidden_dim, self.action_dim).to(device)
#         self.optimizer = optim.Adam(self.q_net.parameters(), lr=learning_rate)
#         self.gamma = gamma
#         self.epsilon = epsilon
#         self.target_update = target_update
#         self.count = 0
#         self.device = device
#
#     def take_action(self, state):
#         if random.random() < self.epsilon:
#             action = np.random.randint(self.action_dim)
#         else:
#             state = torch.tensor([state], dtype=torch.float).to(self.device)
#             action = self.q_net(state).argmax().item()
#         return action
#
#     def update(self, transition_dict):
#         states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
#         actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(self.device)
#         rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
#         next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
#         dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)
#
#         q_values = self.q_net(states).gather(1, actions)
#         max_next_q_values = self.target_q_net(next_states).max(1)[0].view(-1, 1)
#         q_targets = rewards + self.gamma * max_next_q_values * (1 - dones)
#         dqn_loss = torch.mean(F.mse_loss(q_values, q_targets))
#         self.optimizer.zero_grad()
#         dqn_loss.backward()
#         self.optimizer.step()
#
#         if self.count % self.target_update == 0:
#             self.target_q_net.load_state_dict(self.q_net.state_dict())
#         self.count += 1
# class DQN:
#     ''' DQN算法 '''
#     def __init__(self, state_dim, hidden_dim, action_dim, learning_rate, gamma,
#                  epsilon, target_update, device):
#         self.action_dim = action_dim
#         self.q_net = Qnet(state_dim, hidden_dim,
#                           self.action_dim).to(device)  # Q网络
#         # 目标网络
#         self.target_q_net = Qnet(state_dim, hidden_dim,
#                                  self.action_dim).to(device)
#         # 使用Adam优化器
#         self.optimizer = torch.optim.Adam(self.q_net.parameters(),
#                                           lr=learning_rate)
#         self.gamma = gamma  # 折扣因子
#         self.epsilon = epsilon  # epsilon-贪婪策略
#         self.target_update = target_update  # 目标网络更新频率
#         self.count = 0  # 计数器,记录更新次数
#         self.device = device
#
#     def take_action(self, state):  # epsilon-贪婪策略采取动作
#         if np.random.random() < self.epsilon:
#             action = np.random.randint(self.action_dim)
#         else:
#             state = torch.tensor([state], dtype=torch.float).to(self.device)
#             action = self.q_net(state).argmax().item()
#         return action
#
#     def update(self, transition_dict):
#         states = torch.tensor(transition_dict['states'],
#                               dtype=torch.float).to(self.device)
#         actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(
#             self.device)
#         rewards = torch.tensor(transition_dict['rewards'],
#                                dtype=torch.float).view(-1, 1).to(self.device)
#         next_states = torch.tensor(transition_dict['next_states'],
#                                    dtype=torch.float).to(self.device)
#         dones = torch.tensor(transition_dict['dones'],
#                              dtype=torch.float).view(-1, 1).to(self.device)
#
#         q_values = self.q_net(states).gather(1, actions)  # Q值
#         # 下个状态的最大Q值
#         max_next_q_values = self.target_q_net(next_states).max(1)[0].view(
#             -1, 1)
#         q_targets = rewards + self.gamma * max_next_q_values * (1 - dones
#                                                                 )  # TD误差目标
#         dqn_loss = torch.mean(F.mse_loss(q_values, q_targets))  # 均方误差损失函数
#         self.optimizer.zero_grad()  # PyTorch中默认梯度会累积,这里需要显式将梯度置为0
#         dqn_loss.backward()  # 反向传播更新参数
#         self.optimizer.step()
#
#         if self.count % self.target_update == 0:
#             self.target_q_net.load_state_dict(
#                 self.q_net.state_dict())  # 更新目标网络
#         self.count += 1
# class ConvolutionalQnet(nn.Module):
#     def __init__(self,action_dim,in_channels=4):
#         super().__init__()
#         self.conv1 = nn.Conv2d(in_channels,32,kernel_size=8,stride=4)
#         self.conv2 = nn.Conv2d(32,64,kernel_size=4,stride=2)
#         self.conv3 = nn.Conv2d(64,64,kernel_size=3,stride=1)
#         self.fc4 = nn.Linear(7*7*64,512)
#         self.head = nn.Linear(512, action_dim)
#     def forward(self, x):
#         x=x/255
#         x = F.relu(self.conv1(x))
#         x = F.relu(self.conv2(x))
#         x = F.relu(self.conv3(x))
#         x=F.relu(self.fc4(x))
#         return self.head(x)
class ReplayBuffer(object):
    def __init__(self, capacity: int) -> None:
        self.buffer = collections.deque(maxlen=capacity)  # 此处就是创建了一个指定长度的队列（双向列表）

    def add(self, state: int, action: int, reward: float, next_state: int, done: bool) -> None:
        self.buffer.append((state, action, reward, next_state, done))
        # 该方法就是将s，a，r，s'，done作为一个元组（可以称为一个buffer）从后面添加到双向列表（队列）中

    def sample(self, batch_size: int) -> Tuple[array, int, float, array, bool]:
        transitions = random.sample(self.buffer, batch_size)
        # 此处就是在初始化的buffer队列中抽取batch_size个（s，a，r，s'，done）元组，而且不管是什么iterable，sample返回的就是list
        '''
        buffers = collections.deque(maxlen=10)
        buffer_list = [(1.1, 1.2, 1.3, 1.4, 1.5), (2.1, 2.2, 2.3, 2.4, 2.5), (3.1, 3.2, 3.3, 3.4, 3.5),
                       (4.1, 4.2, 4.3, 4.4, 4.5), (5.1, 5.2, 5.3, 5.4, 5.5), (6.1, 6.2, 6.3, 6.4, 6.5),
                       (7.1, 7.2, 7.3, 7.4, 7.5), (8.1, 8.2, 8.3, 8.4, 8.5), (9.1, 9.2, 9.3, 9.4, 9.5)]
        for item in buffer_list:
            buffers.append(item)
        transitions = random.sample(buffers, 3)
        # print(transitions)
        # print(type(transitions))
        a, b, c, d, e = zip(*transitions)
        print(a, b, c, d, e)#(5.1, 7.1, 3.1) (5.2, 7.2, 3.2) (5.3, 7.3, 3.3) (5.4, 7.4, 3.4) (5.5, 7.5, 3.5)
        '''
        state, action, reward, next_state, done = zip(*transitions)  # 如上面代码所示，返回来的就是batch_size个state、action等的元组
        return np.array(state), action, reward, np.array(next_state), done  # 此处一点不明白的是，两个state为什么要array化

    def size(self):
        return len(self.buffer)


class Qnet(nn.Module):  # 这整个类就是创建了一个很简单的网络
    def __init__(self, state_dim, hidden_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class DQN(object):
    def __init__(self, state_dim: int, hidden_dim: int, action_dim: int, learning_rate: float, gamma: float,
                 epsilon: float, target_update: int, device: device):
        self.action_dim = action_dim
        self.q_net = Qnet(state_dim, hidden_dim, action_dim).to(device)  # 实例化一个Qnet
        self.target_q_net = Qnet(state_dim, hidden_dim, action_dim).to(device)  # 再实例化一个Qnet作为目标网络，上面那个作为运行的网络
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=learning_rate)  # 因为q_net是实际运行的，所以优化器的参数是q_net的
        self.gamma = gamma
        self.epsilon = epsilon
        self.target_update = target_update
        self.count = 0
        self.device = device

    def take_action(self, state):  # 这整个方法就是在给定state下选取一个action
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.action_dim)  # 在行动空间中取一个索引
        else:
            state = torch.tensor([state], dtype=torch.float).to(self.device)
            action = self.q_net(state).argmax().item()
            # 此处需要和QLearning区别，DQN是负责根据state生成策略的。如7.3节所说
            # “我们还可以只将状态s输入到神经网络中， 使其同时输出每一个动作的Q值。 ”
            # 从QNet看，输出维度为action_dim，就是每个动作的值函数。那么这里的action就是所得的动作
        return action  # 一直到这里，还都明白

    def update(self, transition_dict: Dict) -> None:  # 从名称看应该是更新target网络的
        states: Tensor = torch.tensor(transition_dict['states'], dtype=torch.float).to(
            self.device)  # 此处 Dict是从PlayBuffer中返回的那个？但是那里返回的是元组啊，此处阙疑，或者是整理过
        # ###################♠♠♠♠♠打断点看一下###################
        actions: Tensor = torch.tensor(transition_dict['actions']).view(-1, 1).to(self.device)
        print(f'actions:{actions}')
        print(f'actions shape:{actions.shape}')
        rewards: Tensor = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(
            self.device)  # 这里需要tensor转一下数据类型可以理解，因为rewa本来就是float类型的，但是states为什么也要转？action没转也可以理解，那个本来就是正数类型
        next_states: Tensor = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        dones: Tensor = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(
            self.device)  # 还有一点就是为什么有的需要view一下，有的不需要？就是两个状态不需要，看一下后面代码吧。另外一点，ReplayBuffer里面的buffer是哪里添加的，至今没有着落，这里直接开始用了！

        # 上面准备了数据，严格来讲什么都没有做呢
        # ###################♠♠♠♠♠打断点看一下###################
        print(f'self.q_net(states):{self.q_net(states)}')
        print(f'self.q_net(states) shape:{(self.q_net(states)).shape}')
        q_values: Tensor = self.q_net(states).gather(1, actions)
        print(f'q_values:{self.q_net(states)}')
        print(f'q_values shape:{(self.q_net(states)).shape}')
        # q_net输出的是一个batch_size*action_dim的向量，actions是batch_size维的向量，向量的每个值是动作。从源码表述看，这里的结果是Q 值，依然不是很清晰，需要留意。
        # ###################♠♠♠♠♠打断点看一下###################
        max_next_q_values: Tensor = self.q_net(next_states).max(1)[0].view(-1, 1)  # 这里是那个迭代更新式子中的max那块。
        q_targets: Tensor = rewards + self.gamma * max_next_q_values * (1 - dones)  # 整个的TD误差目标
        qdn_loss = torch.mean(F.mse_loss(q_values, q_targets))  # 损失函数
        self.optimizer.zero_grad()  # 从这里开始就开始反向传播了。后面都是固定程序了。从这里看，只有q_values、max_next_q_values如何获得的不甚明了
        qdn_loss.backward()
        self.optimizer.step()
        # 下面是选择时机更新target网络了
        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.count += 1


if __name__ == '__main__':
    # lr = 2e-3
    # num_episodes = 500
    # hidden_dim = 128
    # gamma = 0.98
    # epsilon = 0.01
    # target_update = 10
    # buffer_size = 10000
    # minimal_size = 500
    # batch_size = 64
    # device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    #
    # env_name = 'CartPole-v0'
    # env = gym.make(env_name)
    # random.seed(0)
    # env.seed(0)
    # torch.manual_seed(0)
    # replay_buffer = ReplayBuffer(buffer_size)
    # state_dim = env.observation_space.shape[0]
    # action_dim = env.action_space.n
    # agent = DQN(state_dim, hidden_dim, action_dim, lr, gamma, epsilon, target_update, device)
    #
    # return_list = []
    # for i in range(10):
    #     with tqdm(total=int(num_episodes / 10), desc=f'Iteration {i}') as pbar:
    #         for i_episode in range(int(num_episodes / 10)):
    #             episode_return = 0
    #             state = env.reset()
    #             done = False
    #             while not done:
    #                 action = agent.take_action(state)
    #                 next_state, reward, done, _ = env.step(action)
    #                 replay_buffer.add(state, action, reward, next_state, done)
    #                 state = next_state
    #                 episode_return += reward
    #                 if replay_buffer.size() > minimal_size:
    #                     b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size=batch_size)
    #                     transition_dict = {'states': b_s, 'actions': b_a, 'next_states': b_ns, 'rewards': b_r,
    #                                        'dones': b_d}
    #                     agent.update(transition_dict)
    #                 return_list.append(episode_return)
    #                 if (i_episode + 1) % 10 == 0:
    #                     pbar.set_postfix({'episode': f'{num_episodes / 10 * i + i_episode + 1}',
    #                                       'return': f'{np.mean(return_list[-10:]):.3f}'})
    #                 pbar.update(1)
    # episodes_list = list(range(len(return_list)))
    # plt.plot(episodes_list, return_list)
    # plt.xlabel('Episodes')
    # plt.ylabel('Returns')
    # plt.title('DQN on {}'.format(env_name))
    # plt.show()
    #
    # mv_return = rl_utils.moving_average(return_list, 9)
    # plt.plot(episodes_list, mv_return)
    # plt.xlabel('Episodes')
    # plt.ylabel('Returns')
    # plt.title('DQN on {}'.format(env_name))
    # plt.show()
    # ################################################################
    # lr = 2e-3
    # num_episodes = 500
    # hidden_dim = 128
    # gamma = 0.98
    # epsilon = 0.01
    # target_update = 10
    # buffer_size = 10000
    # minimal_size = 500
    # batch_size = 64
    # device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
    #     "cpu")
    #
    # env_name = 'CartPole-v0'
    # env = gym.make(env_name)
    # random.seed(0)
    # np.random.seed(0)
    # env.seed(0)
    # torch.manual_seed(0)
    # replay_buffer = ReplayBuffer(buffer_size)
    # state_dim = env.observation_space.shape[0]
    # action_dim = env.action_space.n
    # agent = DQN(state_dim, hidden_dim, action_dim, lr, gamma, epsilon,
    #             target_update, device)
    #
    # return_list = []
    # for i in range(10):
    #     with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
    #         for i_episode in range(int(num_episodes / 10)):
    #             episode_return = 0
    #             state = env.reset()
    #             done = False
    #             while not done:
    #                 action = agent.take_action(state)
    #                 next_state, reward, done, _ = env.step(action)
    #                 replay_buffer.add(state, action, reward, next_state, done)
    #                 state = next_state
    #                 episode_return += reward
    #                 # 当buffer数据的数量超过一定值后,才进行Q网络训练
    #                 if replay_buffer.size() > minimal_size:
    #                     b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
    #                     transition_dict = {
    #                         'states': b_s,
    #                         'actions': b_a,
    #                         'next_states': b_ns,
    #                         'rewards': b_r,
    #                         'dones': b_d
    #                     }
    #                     agent.update(transition_dict)
    #             return_list.append(episode_return)
    #             if (i_episode + 1) % 10 == 0:
    #                 pbar.set_postfix({
    #                     'episode':
    #                         '%d' % (num_episodes / 10 * i + i_episode + 1),
    #                     'return':
    #                         '%.3f' % np.mean(return_list[-10:])
    #                 })
    #             pbar.update(1)
    # episodes_list = list(range(len(return_list)))
    # plt.plot(episodes_list, return_list)
    # plt.xlabel('Episodes')
    # plt.ylabel('Returns')
    # plt.title('DQN on {}'.format(env_name))
    # plt.show()
    #
    # mv_return = rl_utils.moving_average(return_list, 9)
    # plt.plot(episodes_list, mv_return)
    # plt.xlabel('Episodes')
    # plt.ylabel('Returns')
    # plt.title('DQN on {}'.format(env_name))
    # plt.show()

    ######################################
    lr: float = 2e-3
    num_episodes: int = 500
    hidden_dim: int = 128
    gamma: float = 0.98
    epsilon: float = 0.01
    target_update: int = 10
    buffer_size: int = 10000
    minimal_size: int = 500  # 这是什么东西？
    batch_size: int = 64
    device: device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # 上面是准备了所有的超参数，下面开始实例化环境并进行训练
    env_name: str = 'CartPole-v0'
    env: Env = gym.make(env_name)
    # env: Env = gym.make(env_name, render_mode="human")
    random.seed(0)
    np.random.seed(0)
    env.seed(0)
    torch.manual_seed(0)
    replay_buffer: ReplayBuffer = ReplayBuffer(buffer_size)  # ReplayBuffer这个类初始化就是弄了一个数据结构
    state_dim: int = env.observation_space.shape[0]
    # print(state_dim)#4
    action_dim: int = env.action_space.n
    # print(action_dim)  # 2
    agent: DQN = DQN(state_dim, hidden_dim, action_dim, lr, gamma, epsilon, target_update, device)  # 实例化一个智能体
    # 下面开始训练过程
    # ###################♠♠♠♠♠打断点看一下###################
    return_list: List = []  # 这是干啥的？
    for i in range(10):
        with tqdm(total=int(num_episodes / 10), desc=f'Iteration {i}') as pbar:
            for i_episode in range(int(num_episodes / 10)):
                episode_return = 0
                state = env.reset()  # 此处开始
                done = False
                while not done:
                    action = agent.take_action(state)
                    next_state, reward, done, _ = env.step(action)
                    replay_buffer.add(state, action, reward, next_state, done)  # 前面关于没有地方向ReplayBuffer中添加的疑问，在这里
                    state = next_state
                    episode_return += reward
                    if replay_buffer.size() > minimal_size:
                        b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(
                            batch_size)  # 就是只要满了就抽出batch_size，也就是本例中的64个来，b是batch的意思
                        transition_dict = {
                            'states': b_s,
                            'actions': b_a,
                            'rewards': b_r,
                            'next_states': b_ns,
                            'dones': b_d
                        }  # 前面关于update方法中Dict哪来的疑问，在这里
                        agent.update(transition_dict)
                        # 以下就是进度条处理了，上面除了那个gather，基本都明白了。
                return_list.append(episode_return)
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({'episode': f'{(num_episodes / 10 * i + i_episode + 1)}',
                                      'return': f'{np.mean(return_list[-10:]):.3f}'})
                pbar.update(1)
