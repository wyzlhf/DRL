import random
import time
import nptyping
import numpy as np
from tqdm import tqdm
from numpy import array
from matplotlib import pyplot as plt
from numpy import ndarray as Array
from typing import List, Tuple, Dict


class CliffWalkingEnv(object):
    def __init__(self, ncol: int, nrow: int) -> None:
        self.ncol: int = ncol
        self.nrow: int = nrow
        self.x: int = 0
        self.y: int = self.nrow - 1

    def step(self, action: int) -> Tuple[int, int, bool]:
        change: List[List[int]] = [[0, -1], [0, 1], [-1, 0], [1, 0]]
        self.x: int = min(self.ncol - 1, max(0, self.x + change[action][0]))
        self.y: int = min(self.nrow - 1, max(0, self.y + change[action][1]))
        next_state: int = self.y * self.ncol + self.x
        reward: int = -1
        done: bool = False
        if self.y == self.nrow - 1 and self.x > 0:
            done = True
            if self.x != self.ncol - 1:
                reward: int = -100
        return next_state, reward, done

    def reset(self) -> int:
        self.x: int = 0
        self.y: int = self.nrow - 1
        return self.y * self.ncol + self.x  # 具体到本例就是返回到36


class DynaQ(object):
    def __init__(self, ncol: int, nrow: int, epsilon: float, alpha: float, gamma: float, n_planning: int,
                 n_action: int = 4) -> None:
        self.Q_table: Array[int, nrow * ncol, n_action] = np.zeros([nrow * ncol, n_action])
        # self.Q_table:nptyping.NDArray[int, shape=(nrow * ncol,n_action)] = np.zeros([nrow * ncol, n_action])
        self.n_action: int = n_action
        self.alpha: float = alpha
        self.gamma: float = gamma
        self.epsilon: float = epsilon
        self.n_planning: int = n_planning  # 执行 Q-planning 的次数, 对应 1 次 Q-learning
        self.model: Dict = dict()  # 环境模型

    def take_action(self, state: int) -> int:
        if np.random.random() < self.epsilon:
            action: int = np.random.randint(self.n_action)
        else:
            action: int = np.argmax(self.Q_table[state])
        return action

    def q_learning(self, s0: int, a0: int, r: float, s1: int) -> None:
        td_error: float = r + self.gamma * self.Q_table[s1].max() - self.Q_table[s0, a0]
        self.Q_table[s0, a0] += self.alpha * td_error

    def update(self, s0, a0, r, s1) -> None:
        self.q_learning(s0, a0, r, s1)
        self.model[(s0, a0)] = (r, s1)
        for _ in range(self.n_planning):
            (s, a), (r, s_) = random.choice(list(self.model.items()))
            self.q_learning(s, a, r, s_)


def DynaQ_CliffWalking(n_planning):
    ncol = 12
    nrow = 4
    env = CliffWalkingEnv(ncol, nrow)
    epsilon = 0.01
    alpha = 0.1
    gamma = 0.9
    agent = DynaQ(ncol, nrow, epsilon, alpha, gamma, n_planning)
    num_episodes = 300  # 智能体在环境中运行多少条序列

    return_list = []  # 记录每一条序列的回报
    for i in range(10):  # 显示10个进度条
        # tqdm的进度条功能
        with tqdm(total=int(num_episodes / 10),
                  desc='Iteration %d' % i) as pbar:
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
    return return_list


if __name__ == '__main__':
    np.random.seed(0)
    random.seed(0)
    n_planning_list = [0, 2, 20]
    for n_planning in n_planning_list:
        print('Q-planning步数为：%d' % n_planning)
        time.sleep(0.5)
        return_list = DynaQ_CliffWalking(n_planning)
        episodes_list = list(range(len(return_list)))
        plt.plot(episodes_list,
                 return_list,
                 label=str(n_planning) + ' planning steps')
    plt.legend()
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('Dyna-Q on {}'.format('Cliff Walking'))
    plt.show()
