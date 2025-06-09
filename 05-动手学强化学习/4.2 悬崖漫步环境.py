import copy
from typing import List, Tuple, Union

class CliffWalkEnv(object):
    def __init__(self, ncol=12, nrow=4):
        self.ncol = ncol
        self.nrow = nrow
        # 转移矩阵 P[state][action] = [(p, next_state, reward, done)]包含下一个状态和奖励
        self.P = self.createP()

    '''
    以下几点：
    1、这个环境类就是用行标、列标去表示一个按照序号排列的状态；
    2、转移矩阵 P[state][action] = [(p, next_state, reward, done)]的表示法是和GYM对齐的
    '''

    def createP(self):
        P = [[[] for _ in range(4)] for _ in range(self.nrow * self.ncol)]
        change = [
            [0, -1],  # up
            [0, 1],  # down
            [-1, 0],  # left
            [1, 0]  # right
        ]
        for i in range(self.nrow):
            for j in range(self.ncol):
                for a in range(4):
                    if i == self.nrow - 1 and j > 0:
                        P[i * self.ncol + j][a] = [(1, i * self.ncol + j, 0, True)]
                        continue
                    next_x = min(self.ncol - 1, max(0, j + change[a][0]))
                    next_y = min(self.nrow - 1, max(0, i + change[a][1]))
                    next_state = next_y * self.ncol + next_x
                    reward = -1
                    done = False
                    # 下一个位置在悬崖或者终点
                    if next_y == self.nrow - 1 and next_x > 0:
                        done = True
                        if next_x != self.ncol - 1:
                            reward = -100
                    P[i * self.ncol + j][a] = [(1, next_state, reward, done)]
        return P


class PolicyIteration(object):
    def __init__(self, env: CliffWalkEnv, theta: float, gamma: float) -> None:
        self.env = env
        self.v: List = [0] * self.env.ncol * self.env.nrow
        self.pi: List[List[float]] = [[0.25, 0.25, 0.25, 0.25] for _ in range(self.env.ncol * self.env.nrow)]
        self.theta = theta  # 策略评估收敛阈值
        self.gamma = gamma  # 回报折扣因子

    def policy_evaluation(self) -> None:
        cnt: int = 1
        while 1:
            max_diff: float = 0
            new_v: List[float] = [0] * self.env.ncol * self.env.nrow
            for s in range(self.env.ncol * self.env.nrow):
                qsa_list: List[float] = []  # 开始计算状态 s 下的所有 Q(s,a)价值
                for a in range(4):
                    qsa: float = 0
                    for res in self.env.P[s][a]:
                        # next_p: Tuple[float, int, float, int] = res
                        # (p, next_state, r, done) = next_p
                        p, next_state, r, done = res
                        qsa += p * (r + self.gamma * self.v[next_state] * (1 - done))
                    qsa_list.append(self.pi[s][a] * qsa)
                new_v[s] = sum(qsa_list)
                max_diff = max(max_diff, abs(new_v[s] - self.v[s]))
            self.v = new_v
            if max_diff < self.theta:
                break
            cnt += 1
        print(f'策略评估进行{cnt}轮后完成')

    def policy_improvement(self) -> List:
        for s in range(self.env.ncol * self.env.nrow):
            qsa_list: List = []
            for a in range(4):
                qsa: float = 0
                for res in self.env.P[s][a]:
                    p, next_state, r, done = res
                    qsa += p * (r + self.gamma * self.v[next_state] * (1 - done))
                qsa_list.append(qsa)
            maxq: float = max(qsa_list)
            cntq = qsa_list.count(maxq)
            self.pi[s] = [1 / cntq if q == maxq else 0 for q in qsa_list]
        print('策略提升完成')
        return self.pi

    def policy_iteration(self) -> None:
        while 1:
            self.policy_evaluation()
            old_Pi = copy.deepcopy(self.pi)
            new_Pi = self.policy_improvement()
            if old_Pi == new_Pi:
                break


class ValueIteration(object):
    def __init__(self, env: CliffWalkEnv, theta: float, gamma: float) -> None:
        self.env = env
        self.v = [0] * self.env.ncol * self.env.nrow
        self.theta = theta
        self.gamma = gamma
        self.pi = [None for i in range(self.env.ncol * self.env.nrow)]

    def value_iteration(self) -> None:
        cnt: int = 0
        while 1:
            max_diff: float = 0
            new_v: List[float] = [0] * self.env.ncol * self.env.nrow
            for s in range(self.env.ncol * self.env.nrow):
                qsa_list: List[float] = []
                for a in range(4):
                    qsa: float = 0
                    for res in self.env.P[s][a]:
                        p, next_state, r, done = res
                        qsa += p * (r + self.gamma * self.v[next_state] * (1 - done))
                    qsa_list.append(qsa)  # 这一行和下一行代码是价值迭代和策略迭代的主要区别——————此处需要仔细参详
                new_v[s] = max(qsa_list)
                max_diff = max(max_diff, abs(new_v[s] - self.v[s]))
            self.v = new_v
            if max_diff < self.theta:
                break
            cnt += 1
        print(f'价值迭代一共进行{cnt}轮')
        self.get_policy()

    def get_policy(self):
        for s in range(self.env.ncol * self.env.nrow):
            qsa_list: List[float] = []
            for a in range(4):
                qsa: float = 0
                for res in self.env.P[s][a]:
                    p, next_state, r, done = res
                    qsa += r + p * self.gamma * self.v[next_state] * (1 - done)
                qsa_list.append(qsa)
            maxq: float = max(qsa_list)
            cntq = qsa_list.count(maxq)
            self.pi[s] = [1 / cntq if q == maxq else 0 for q in qsa_list]


def print_agent(agent: Union[PolicyIteration,ValueIteration], action_meaning: List[str], disaster: List = [],
                end: List = []) -> None:
    print('状态价值：')
    for i in range(agent.env.nrow):
        for j in range(agent.env.ncol):
            print('%6.6s' % ('%.3f' % agent.v[i * agent.env.ncol + j]), end=' ')
        print()
    print('策略：')
    for i in range(agent.env.nrow):
        for j in range(agent.env.ncol):
            if (i * agent.env.ncol + j) in disaster:
                print('****', end=' ')
            elif (i * agent.env.ncol + j) in end:
                print('EEEE', end=' ')
            else:
                a = agent.pi[i * agent.env.ncol + j]
                pi_str = ''
                for k in range(len(action_meaning)):
                    pi_str += action_meaning[k] if a[k] > 0 else 'o'
                print(pi_str, end=' ')
        print()


if __name__ == '__main__':
    # env = CliffWalkEnv()
    # action_meaning = ['^', 'v', '<', '>']
    # theta = 0.001
    # gamma = 0.9
    # agent = PolicyIteration(env, theta, gamma)
    # agent.policy_iteration()
    # print_agent(agent, action_meaning, list(range(37, 47)), [47])

    env = CliffWalkEnv()
    action_meaning = ['^', 'v', '<', '>']
    theta = 0.001
    gamma = 0.9
    agent = ValueIteration(env, theta, gamma)
    agent.value_iteration()
    print_agent(agent, action_meaning, list(range(37, 47)), [47])
