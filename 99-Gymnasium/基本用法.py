import gymnasium as gym

env = gym.make("MountainCarContinuous-v0", render_mode="human")
observation, info = env.reset()

episode_over = False
while not episode_over:
    action = env.action_space.sample()  # agent policy that uses the observation and info
    observation, reward, terminated, truncated, info = env.step(action)
    print(reward)

    episode_over = terminated or truncated

env.close()