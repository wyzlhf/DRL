import numpy as np

# np.random.seed(0)
# P = [
#     [0.9, 0.1, 0.0, 0.0, 0.0, 0.0],
#     [0.5, 0.0, 0.5, 0.0, 0.0, 0.0],
#     [0.0, 0.0, 0.0, 0.6, 0.0, 0.4],
#     [0.0, 0.0, 0.0, 0.0, 0.3, 0.7],
#     [0.0, 0.2, 0.3, 0.5, 0.0, 0.0],
#     [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
# ]
# P = np.array(P)
rewards = [-1, -2, -2, 10, 1, 0]
gamma = 0.5


def compute_returns(start_index: int, chain: list, gamma: float):
    G: float = 0
    # print(len(chain)) #==4
    for i in reversed(range(start_index, len(chain))):
        G = gamma * G + rewards[chain[i] - 1]
        # print(i)
        # print(f'状态：{chain[i]}')
        # print()
    return G


chain = [1, 2, 3, 6]
start_index = 0
G = compute_returns(start_index, chain, gamma)


# print(f'根据本序列计算得到回报为：{G}')
def compute(P, rewards, gamma, states_num):
    ''' 利用贝尔曼方程的矩阵形式计算解析解,states_num 是 MRP 的状态数 '''
    rewards = np.array(rewards).reshape((-1, 1))  # 将 rewards 写成列向量形式
    value = np.dot(np.linalg.inv(np.eye(states_num, states_num) - gamma * P), rewards)
    return value


# 3.4.4节
S = ["s1", "s2", "s3", "s4", "s5"]  # 状态集合
A = ["保持 s1", "前往 s1", "前往 s2", "前往 s3", "前往 s4", "前往 s5", "概率前往"]  # 动作集合
# 状态转移函数
P = {
    "s1-保持 s1-s1": 1.0, "s1-前往 s2-s2": 1.0,
    "s2-前往 s1-s1": 1.0, "s2-前往 s3-s3": 1.0,
    "s3-前往 s4-s4": 1.0, "s3-前往 s5-s5": 1.0,
    "s4-前往 s5-s5": 1.0, "s4-概率前往-s2": 0.2,
    "s4-概率前往-s3": 0.4, "s4-概率前往-s4": 0.4,
}
# 奖励函数
R = {
    "s1-保持 s1": -1, "s1-前往 s2": 0,
    "s2-前往 s1": -1, "s2-前往 s3": -2,
    "s3-前往 s4": -2, "s3-前往 s5": 0,
    "s4-前往 s5": 10, "s4-概率前往": 1,
}
gamma = 0.5  # 折扣因子
MDP = (S, A, P, R, gamma)
# 策略 1,随机策略
Pi_1 = {
    "s1-保持 s1": 0.5, "s1-前往 s2": 0.5,
    "s2-前往 s1": 0.5, "s2-前往 s3": 0.5,
    "s3-前往 s4": 0.5, "s3-前往 s5": 0.5,
    "s4-前往 s5": 0.5, "s4-概率前往": 0.5,
}
# 策略 2
Pi_2 = {
    "s1-保持 s1": 0.6, "s1-前往 s2": 0.4,
    "s2-前往 s1": 0.3, "s2-前往 s3": 0.7,
    "s3-前往 s4": 0.5, "s3-前往 s5": 0.5,
    "s4-前往 s5": 0.1, "s4-概率前往": 0.9,
}


def join(str1: str, str2: str):
    return str1 + '-' + str2


gamma = 0.5
# 转化后的 MRP 的状态转移矩阵
P_from_mdp_to_mrp = [
    [0.5, 0.5, 0.0, 0.0, 0.0],
    [0.5, 0.0, 0.5, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.5, 0.5],
    [0.0, 0.1, 0.2, 0.2, 0.5],
    [0.0, 0.0, 0.0, 0.0, 1.0],
]
P_from_mdp_to_mrp = np.array(P_from_mdp_to_mrp)
R_from_mdp_to_mrp = [-0.5, -1.5, -1.0, 5.5, 0]


# V = compute_returns()
# 3.5 蒙特卡洛方法

def sample(MDP, Pi, timestep_max, number):
    S, A, P, R, gamma = MDP
    episodes = []
    for _ in range(number):
        episode = []
        timestep = 0
        s = S[np.random.randint(4)]
        while s != "s5" and timestep <= timestep_max:
            timestep += 1
            rand, temp = np.random.rand(), 0
            for a_opt in A:
                temp += Pi.get(join(s, a_opt), 0)
                if temp > rand:
                    a = a_opt
                    r = R.get(join(s, a), 0)
                    break
            rand, temp = np.random.rand(), 0
            for s_opt in S:
                temp += P.get(join(join(s, a), s_opt), 0)
                if temp > rand:
                    s_next = s_opt
                    break
            episode.append((s, a, r, s_next))
            s = s_next
        episodes.append(episode)
    return episodes
def MC(episodes,V,N,gamma):
    for episode in episodes:
        G=0
        for i in range(len(episode)-1,-1,-1):
            (s, a, r, s_next) = episode[i]
            G=r+gamma*G
            N[s]=N[s]+1
            V[s]=V[s]+(G-V[s])/N[s]


if __name__ == '__main__':
    # episodes = sample(MDP, Pi_1, 20, 5)
    # print(f'第一条序列\n {episodes[0]}')
    # print(f'第二条序列\n {episodes[1]}')
    # print(f'第五条序列\n {episodes[4]}')

    timestep_max = 20
    # 采样 1000 次,可以自行修改
    episodes = sample(MDP, Pi_1, timestep_max, 1000)
    gamma = 0.5
    V = {"s1": 0, "s2": 0, "s3": 0, "s4": 0, "s5": 0}
    N = {"s1": 0, "s2": 0, "s3": 0, "s4": 0, "s5": 0}
    MC(episodes, V, N, gamma)
    print("使用蒙特卡洛方法计算 MDP 的状态价值为\n", V)
