import gym
import numpy as np
from PIL import Image
import sys

env = gym.make('Pong-v0')
env.reset()

done = False
i = 0
start = 0

if len(sys.argv) < 3:
    print("Usage: collect_pong <games> <start_point>")
    exit()

try:
    games = int(sys.argv[1])
    start = int(sys.argv[2])
    i = start
except:
    print("Please provide a valid number for games and start point.")
    exit()

for _ in range(games):
    count = 0
    while not done:
        o, r, done, info = env.step(env.action_space.sample())
        count += 1
        # Ignore first 25 frames of the game, since the games starts after this amount.
        if count < 25:
            continue
        img = Image.fromarray(o)
        img.save("images/pong_" + str(i) + ".png")
        i += 1
    done = False
    env.reset()

print("Saved {} images.".format(i-start))
print("Total images: {}".format(i))
env.close()
