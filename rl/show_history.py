import matplotlib.pyplot as plt
import pickle
import numpy as np

history = pickle.load(open('history.p', 'rb'))

running_average = []
window = 200
history = np.array(history[:17000])

print(np.max(history[:, 1]))

for i in range(window//2, len(history)-(window//2)):
    avg = 0
    for j in range(i - window//2, i + window//2):
        avg += history[j][1]

    avg /= window
    running_average.append(avg)

plt.plot(range(len(running_average)), running_average, 'b')
plt.xlabel('Games')
plt.ylabel('Average Score')
plt.title("Pong Agent Training on Raw Space")
plt.legend()
plt.show()
