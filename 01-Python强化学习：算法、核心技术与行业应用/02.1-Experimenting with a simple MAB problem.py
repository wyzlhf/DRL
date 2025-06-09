import numpy as np


class GaussianBandit(object):
    def __init__(self, mean: float = 0, stdev: float = 1) -> None:
        self.mean = mean
        self.stdev = stdev

    def pull_lever(self) -> float:
        reward: float = np.random.normal(self.mean, self.stdev)
        return reward


class GaussianBanditGame(object):
    def __init__(self, bandits: list) -> None:
        self.bandits = bandits
        np.random.shuffle(bandits)
        self.reset_game()

    def play(self, choice: int) -> float:
        reward: float = self.bandits[choice - 1].pull_lever()
        self.rewards.append(reward)
        self.total_reward += reward
        self.n_played += 1
        return reward

    def user_play(self) -> None:
        self.reset_game()
        print('游戏开始。请输入0结束游戏。')
        while True:
            print(f'\n----第{self.n_played}轮游戏：')
            choice: int = int(input(f'请从序号1到{len(self.bandits)}选择一台机器：'))
            if choice in range(1, len(self.bandits) + 1):
                reward: float = self.play(choice)
                print(f'第{choice}台机器给出奖励{reward}')
                avg_rew: float = self.total_reward / self.n_played
                print(f'到目前为止，您所获得的平均奖励为{avg_rew}')
            else:
                break
        print('游戏结束')
        if self.n_played > 0:
            print(f'您获得的总奖励为：{self.total_reward}；您一共玩{self.n_played}场游戏。')
            avg_rew = self.total_reward / self.n_played
            print(f'平均奖励为：{avg_rew}')

    def reset_game(self):
        self.rewards: list = []
        self.total_reward: float = 0
        self.n_played: float = 0
if __name__ == '__main__':
    slotA=GaussianBandit(mean=5, stdev=3)
    slotB=GaussianBandit(mean=6, stdev=2)
    slotC=GaussianBandit(mean=1, stdev=5)
    game=GaussianBanditGame(bandits=[slotA, slotB, slotC])
    game.user_play()