import matplotlib.pyplot as plt
import pickle

history = pickle.load(open('history_latent.p', 'rb'))

running_average = []
window = 100

for i in range(window//2, len(history)-(window//2)):
    avg = 0
    for j in range(i - window//2, i + window//2):
        avg += history[j][1]

    avg /= window
    running_average.append(avg)

plt.plot(range(len(running_average)), running_average, 'b')
plt.show()
