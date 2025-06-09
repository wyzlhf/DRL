from typing import List, Union

import matplotlib.pyplot as plt
import numpy as np
from numpy import array
from tqdm import tqdm


class CliffWalkingEnv(object):
    def __init__(self, ncol: int, nrow: int):
        self.ncol: int = ncol
        self.nrow: int = nrow
        self.x: int = 0
        self.y: int = self.nrow - 1

    def step(self, action: int):
        change: List[List[int]] = [[0, -1], [0, 1], [-1, 0], [1, 0]]
        self.x: int = min(self.ncol - 1, max(0, self.x + change[action][0]))
        self.y: int = min(self.nrow - 1, max(0, self.y + change[action][1]))
        next_state: int = self.y * self.ncol + self.x
        reward: int = -1
        done: bool = False
        if self.y == self.nrow - 1 and self.x > 0:
            done: bool = True
            if self.x != self.ncol - 1:
                reward: int = -100
        return next_state, reward, done

    def reset(self):
        self.x: int = 0
        self.y: int = self.nrow - 1
        return self.y * self.ncol + self.x  # 就是返回所在状态


class Sarsa(object):
    def __init__(self, ncol: int, nrow: int, epsilon: float, alpha: float, gamma: float, n_action: int = 4) -> None:
        self.Q_table: array = np.zeros([nrow * ncol, n_action])
        self.n_action: int = n_action
        self.alpha: float = alpha
        self.gamma: float = gamma
        self.epsilon: float = epsilon

    def take_action(self, state: int) -> int:
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.n_action)
        else:
            action = np.argmax(self.Q_table[state])
        return action

    def best_action(self, state: int) -> List[int]:
        Q_max: int = np.max(self.Q_table[state])
        a = [0 for _ in range(self.n_action)]
        for i in range(self.n_action):
            if self.Q_table[state, i] == Q_max:
                a[i] = 1
        return a

    def update(self, s0, a0, r, s1, a1) -> None:
        td_error = r + self.gamma * self.Q_table[s1, a1] - self.Q_table[s0, a0]
        self.Q_table[s0, a0] += self.alpha * td_error


class nstep_Sarsa(object):
    def __init__(self, n: int, ncol: int, nrow: int, epsilon: float, alpha: float, gamma: float,
                 n_action: int = 4) -> None:
        self.Q_table: array = np.zeros([nrow * ncol, n_action])
        self.n_action: int = n_action
        self.alpha: float = alpha
        self.gamma: float = gamma
        self.epsilon: float = epsilon
        self.n: int = n
        self.state_list: List = []
        self.action_list: List = []
        self.reward_list: List = []

    def take_action(self, state: int) -> int:
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.n_action)
        else:
            action = np.argmax(self.Q_table[state])
        return action

    def best_action(self, state: int) -> List[int]:
        Q_max: List = np.max(self.Q_table[state])
        a: List[int] = [0 for _ in range(self.n_action)]
        for i in range(self.n_action):
            if self.Q_table[state, i] == Q_max:
                a[i] = 1
        return a

    def update(self, s0: int, a0: int, r: float, s1: int, a1: int, done: bool) -> None:
        self.state_list.append(s0)
        self.action_list.append(a0)
        self.reward_list.append(r)
        if len(self.state_list) == self.n:
            G: float = self.Q_table[s1, a1]
            for i in reversed(range(self.n)):
                G: float = self.gamma * G + self.reward_list[i]
                if done and i > 0:
                    s: int = self.state_list[i]
                    a: int = self.action_list[i]
                    self.Q_table[s, a] += self.alpha * (G - self.Q_table[s, a])
            s: int = self.state_list.pop(0)
            a: int = self.action_list.pop(0)
            self.reward_list.pop(0)
            self.Q_table[s, a] += self.alpha * (G - self.Q_table[s, a])
        if done:
            self.state_list = []
            self.action_list = []
            self.reward_list = []


class QLearning(object):
    def __init__(self, ncol: int, nrow: int, epsilon: float, alpha: float, gamma: float, n_action: int = 4) -> None:
        self.Q_table: array = np.zeros([nrow * ncol, n_action])
        self.n_action: int = n_action
        self.alpha: float = alpha
        self.gamma: float = gamma
        self.epsilon: float = epsilon

    def take_action(self, state: int) -> int:
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.n_action)
        else:
            action = np.argmax(self.Q_table[state])
        return action

    def best_action(self, state: int) -> List[int]:
        Q_max: List = np.max(self.Q_table[state])
        a = [0 for _ in range(self.n_action)]
        for i in range(self.n_action):
            if self.Q_table[state, i] == Q_max:
                a[i] = 1
        return a

    def update(self, s0, a0, r, s1) -> None:
        td_error = r + self.gamma * self.Q_table[s1].max() - self.Q_table[s0, a0]
        self.Q_table[s0, a0] += self.alpha * td_error


def print_agent(agent: Union[Sarsa, nstep_Sarsa,QLearning], env: CliffWalkingEnv, action_meaning: List[str], disaster=[],
                end=[]) -> None:
    for i in range(env.nrow):
        for j in range(env.ncol):
            if (i * env.ncol + j) in disaster:
                print('****', end=' ')
            elif (i * env.ncol + j) in end:
                print('EEEE', end=' ')
            else:
                a = agent.best_action(i * env.ncol + j)
                pi_str = ''
                for k in range(len(action_meaning)):
                    pi_str += action_meaning[k] if a[k] > 0 else 'o'
                print(pi_str, end=' ')
        print()


if __name__ == '__main__':
    # ncol, nrow = 12, 4
    # env = CliffWalkingEnv(ncol, nrow)
    # np.random.seed(0)
    # epsilon = 0.1
    # alpha = 0.1
    # gamma = 0.9
    # agent = Sarsa(ncol, nrow, epsilon, alpha, gamma)
    # num_episodes = 500
    #
    # return_list = []
    # ==============================
    # for i in range(10):
    #     with tqdm(total=int(num_episodes / 10), desc=f'Iteration {i}') as pbar:
    #         for i_episode in range(int(num_episodes / 10)):
    #             episode_return = 0
    #             state = env.reset()
    #             action = agent.take_action(state)
    #             done = False
    #             while not done:
    #                 next_state, reward, done = env.step(action)
    #                 next_action = agent.take_action(next_state)
    #                 episode_return += reward
    #                 agent.update(state, action, reward, next_state, next_action)
    #                 state = next_state
    #                 action = next_action
    #             return_list.append(episode_return)
    #             if (i_episode + 1) % 10 == 0:
    #                 pbar.set_postfix({
    #                     'episode': f'{(num_episodes / 10 * i + i_episode + 1)}',
    #                     'return': f'{np.mean(return_list[-10:])}'
    #                 })
    #             pbar.update(1)
    # ==============================
    # episodes_list = list(range(len(return_list)))
    # plt.plot(episodes_list, return_list)
    # plt.xlabel('Episodes')
    # plt.ylabel('Returns')
    # plt.title('Sarsa on {}'.format('Cliff Walking'))
    # plt.show()
    #
    # action_meaning = ['^', 'v', '<', '>']
    # print('Sarsa 算法最终收敛得到的策略为： ')
    # print_agent(agent, env, action_meaning, list(range(37, 47)), [47])

    ######################################################
    # ncol, nrow = 12, 4
    # env = CliffWalkingEnv(ncol, nrow)
    # np.random.seed(0)
    # n_step = 5  # 5步Sarsa算法
    # alpha = 0.1
    # epsilon = 0.1
    # gamma = 0.9
    # agent = nstep_Sarsa(n_step, ncol, nrow, epsilon, alpha, gamma)
    # num_episodes = 500  # 智能体在环境中运行的序列的数量
    #
    # return_list = []  # 记录每一条序列的回报
    # for i in range(10):  # 显示10个进度条
    #     # tqdm的进度条功能
    #     with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
    #         for i_episode in range(int(num_episodes / 10)):  # 每个进度条的序列数
    #             episode_return = 0
    #             state = env.reset()
    #             action = agent.take_action(state)
    #             done = False
    #             while not done:
    #                 next_state, reward, done = env.step(action)
    #                 next_action = agent.take_action(next_state)
    #                 episode_return += reward  # 这里回报的计算不进行折扣因子衰减
    #                 agent.update(state, action, reward, next_state, next_action,
    #                              done)
    #                 state = next_state
    #                 action = next_action
    #             return_list.append(episode_return)
    #             if (i_episode + 1) % 10 == 0:  # 每10条序列打印一下这10条序列的平均回报
    #                 pbar.set_postfix({
    #                     'episode':
    #                         '%d' % (num_episodes / 10 * i + i_episode + 1),
    #                     'return':
    #                         '%.3f' % np.mean(return_list[-10:])
    #                 })
    #             pbar.update(1)
    #
    # episodes_list = list(range(len(return_list)))
    # plt.plot(episodes_list, return_list)
    # plt.xlabel('Episodes')
    # plt.ylabel('Returns')
    # plt.title('5-step Sarsa on {}'.format('Cliff Walking'))
    # plt.show()
    #
    # action_meaning = ['^', 'v', '<', '>']
    # print('5步Sarsa算法最终收敛得到的策略为：')
    # print_agent(agent, env, action_meaning, list(range(37, 47)), [47])


    ###########################################
    ncol, nrow = 12, 4
    env = CliffWalkingEnv(ncol, nrow)
    np.random.seed(0)
    epsilon = 0.1
    alpha = 0.1
    gamma = 0.9
    agent = QLearning(ncol, nrow, epsilon, alpha, gamma)
    num_episodes = 500  # 智能体在环境中运行的序列的数量

    return_list = []  # 记录每一条序列的回报
    for i in range(10):  # 显示10个进度条
        # tqdm的进度条功能
        with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes / 10)):  # 每个进度条的序列数
                episode_return = 0
                state = env.reset()
                done = False
                while not done:
                    action = agent.take_action(state)
                    next_state, reward, done = env.step(action)
                    episode_return += reward  # 这里回报的计算不进行折扣因子衰减
                    agent.update(state, action, reward, next_state)
                    state = next_state
                return_list.append(episode_return)
                if (i_episode + 1) % 10 == 0:  # 每10条序列打印一下这10条序列的平均回报
                    pbar.set_postfix({
                        'episode':
                            '%d' % (num_episodes / 10 * i + i_episode + 1),
                        'return':
                            '%.3f' % np.mean(return_list[-10:])
                    })
                pbar.update(1)

    episodes_list = list(range(len(return_list)))
    plt.plot(episodes_list, return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('Q-learning on {}'.format('Cliff Walking'))
    plt.show()

    action_meaning = ['^', 'v', '<', '>']
    print('Q-learning算法最终收敛得到的策略为：')
    print_agent(agent, env, action_meaning, list(range(37, 47)), [47])
