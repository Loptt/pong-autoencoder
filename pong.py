import gym
import numpy as np
import numpy as np
import matplotlib.pyplot as plt


def rgb2gray(rgb):
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray


env = gym.make('Pong-v0')
env.reset()


for _ in range(1000):
    env.render()
    o, r, d, i = env.step(env.action_space.sample())
    gray = rgb2gray(o)

plt.imshow(gray, cmap='gray', vmin=0, vmax=255)
plt.show()
env.close()
