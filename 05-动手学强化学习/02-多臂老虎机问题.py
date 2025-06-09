from typing import List

import numpy as np
import matplotlib.pyplot as plt
from numpy import array, signedinteger


class BernoulliBandit(object):
    def __init__(self, K: int) -> None:
        self.probs: array = np.random.uniform(size=K)
        self.best_idx: signedinteger = np.argmax(self.probs)
        self.best_prob: float = self.probs[self.best_idx]
        self.K: int = K

    def step(self, k) -> int:
        if np.random.rand() < self.probs[k]:
            return 1
        else:
            return 0


class Solver(object):
    def __init__(self, bandit: BernoulliBandit) -> None:
        self.bandit: BernoulliBandit = bandit
        self.counts: array = np.zeros(self.bandit.K)
        self.regret: float = 0.
        self.actions: list = []
        self.regrets: list = []

    def update_regret(self, k: int) -> None:
        self.regret += self.bandit.best_prob - self.bandit.probs[k]
        self.regrets.append(self.regret)

    def run_one_step(self) -> int:
        raise NotImplementedError

    def run(self, num_steps: int) -> None:
        for _ in range(num_steps):
            k = self.run_one_step()
            self.counts[k] += 1
            self.actions.append(k)
            self.update_regret(k)


class EpsilonGreedy(Solver):
    def __init__(self, bandit: BernoulliBandit, epsilon: float = 0.01, init_prob: float = 1.0):
        super(EpsilonGreedy, self).__init__(bandit)
        self.epsilon: float = epsilon
        self.estimates: array = np.array([init_prob] * self.bandit.K)

    def run_one_step(self) -> int:
        if np.random.random() < self.epsilon:
            k: int = np.random.randint(0, self.bandit.K)
        else:
            k: int = np.argmax(self.estimates)
        r: int = self.bandit.step(k)
        self.estimates[k] += 1. / (self.counts[k] + 1) * (r - self.estimates[k])
        return k


def plot_results(solvers: List[Solver], solver_names: List[str]):
    for idx, solver in enumerate(solvers):
        time_list: range = range(len(solver.regrets))
        plt.plot(time_list, solver.regrets, label=solver_names[idx])
    plt.xlabel('Time steps')
    plt.ylabel('Cumulative regrets')
    plt.title(f'{solvers[0].bandit.K}-armed bandit')
    plt.legend()
    plt.show()


class DecayingEpsilonGreedy(Solver):
    def __init__(self, bandit: BernoulliBandit, init_prob: float = 1.0) -> None:
        super(DecayingEpsilonGreedy, self).__init__(bandit)
        self.estimates: array = np.array([init_prob] * self.bandit.K)
        self.total_count: float = 0

    def run_one_step(self) -> int:
        self.total_count += 1
        if np.random.random() < 1 / self.total_count:
            k = np.random.randint(0, self.bandit.K)
        else:
            k = np.argmax(self.estimates)
        r = self.bandit.step(k)
        self.estimates[k] += 1. / (self.counts[k] + 1) * (r - self.estimates[k])
        return k


if __name__ == '__main__':
    K = 10
    bandit_10_arm = BernoulliBandit(K)

    np.random.seed(1)
    epsilon_greedy_solver = EpsilonGreedy(bandit_10_arm, epsilon=0.01)
    epsilon_greedy_solver.run(5000)
    print('epsilon-贪婪算法的累积懊悔为：', epsilon_greedy_solver.regret)
    plot_results([epsilon_greedy_solver], ["EpsilonGreedy"])
