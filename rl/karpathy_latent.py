""" Trains an agent with (stochastic) Policy Gradients on Pong. Uses OpenAI Gym. """
import numpy as np
import pickle as pickle
import gym
import tensorflow as tf
import time

from transforms.transform_pong_ballless import conversion_pipe as transform_paddle
from transforms.transform_pong_paddleless_big import conversion_pipe as transform_ball
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import array_to_img
import matplotlib.pyplot as plt

plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111)

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)


ball_model = load_model('../models/vae_big_paddleless')
paddles_model = load_model('../models/vae_ballless')

latent_size = ball_model.encoder.output_shape[2][1] + \
    paddles_model.encoder.output_shape[2][1]

# hyperparameters
H = 200  # number of hidden layer neurons
batch_size = 10  # every how many episodes to do a param update?
learning_rate = 1e-4
gamma = 0.99  # discount factor for reward
decay_rate = 0.99  # decay factor for RMSProp leaky sum of grad^2
resume = True  # resume from previous checkpoint?
render = True
history = []
episode_number = 0

# model initialization
D = latent_size * 2  # input dimensionality: 80x80 grid
if resume:
    model = pickle.load(open('save_latent.p', 'rb'))
    history = pickle.load(open('history_latent.p', 'rb'))
    episode_number = len(history)
else:
    model = {}
    model['W1'] = np.random.randn(H, D) / np.sqrt(D)  # "Xavier" initialization
    model['W2'] = np.random.randn(H) / np.sqrt(H)

line1, = ax.plot([], [], 'b')

# update buffers that add up gradients over a batch
grad_buffer = {k: np.zeros_like(v) for k, v in model.items()}
rmsprop_cache = {k: np.zeros_like(v)
                 for k, v in model.items()}  # rmsprop memory


def show_history(history, line, fig):
    running_average = []
    window = 100

    for i in range(window//2, len(history)-(window//2)):
        avg = 0
        for j in range(i - window//2, i + window//2):
            avg += history[j][1]

        avg /= window
        running_average.append(avg)

    line.set_ydata(running_average)
    line.set_xdata(range(len(running_average)))
    fig.canvas.draw()
    fig.canvas.flush_events()


def sigmoid(x):
    # sigmoid "squashing" function to interval [0,1]
    return 1.0 / (1.0 + np.exp(-x))


def prepro(I):
    """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
    I = I[35:195]  # crop
    I = I[::2, ::2, 0]  # downsample by factor of 2
    I[I == 144] = 0  # erase background (background type 1)
    I[I == 109] = 0  # erase background (background type 2)
    I[I != 0] = 1  # everything else (paddles, ball) just set to 1
    return I.astype(np.float).ravel()


def discount_rewards(r):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size)):
        if r[t] != 0:
            # reset the sum, since this was a game boundary (pong specific!)
            running_add = 0
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r


def policy_forward(x):
    h = np.dot(model['W1'], x)
    h[h < 0] = 0  # ReLU nonlinearity
    logp = np.dot(model['W2'], h)
    p = sigmoid(logp)
    return p, h  # return probability of taking action 2, and hidden state


def policy_backward(eph, epdlogp):
    """ backward pass. (eph is array of intermediate hidden states) """
    dW2 = np.dot(eph.T, epdlogp).ravel()
    dh = np.outer(epdlogp, model['W2'])
    dh[eph <= 0] = 0  # backpro prelu
    dW1 = np.dot(dh.T, epx)
    return {'W1': dW1, 'W2': dW2}


env = gym.make("Pong-v0")
observation = env.reset()
prev_x = None  # used in computing the difference frame
prev_latent = None
xs, hs, dlogps, drs = [], [], [], []
running_reward = None
reward_sum = 0

prev_time = time.time()
current_time = time.time()

while True:
    if render:
        env.render()

    '''
    prev_time = current_time
    current_time = time.time()

    print("Frame time:", current_time - prev_time)

    # preprocess the observation, set input to network to be difference image
    cur_x = prepro(observation)
    x = cur_x - prev_x if prev_x is not None else np.zeros(D)
    prev_x = cur_x
    '''
    ball_image = transform_ball(array_to_img(observation))
    paddles_image = transform_paddle(array_to_img(observation))

    latent_ball = ball_model.encoder.call(
        tf.constant([img_to_array(ball_image)]))[2]

    latent_paddles = paddles_model.encoder.call(
        tf.constant([img_to_array(paddles_image)]))[2]

    latent = np.concatenate(
        (np.reshape(latent_ball.numpy(), latent_ball.numpy().shape[1]), np.reshape(latent_paddles.numpy(), latent_paddles.numpy().shape[1])), axis=0)

    prev_latent = prev_latent if prev_latent is not None else np.zeros(
        latent_size)

    x = np.concatenate(
        (latent, prev_latent), axis=0)

    prev_latent = latent

    # forward the policy network and sample an action from the returned probability
    aprob, h = policy_forward(x)
    action = 2 if np.random.uniform() < aprob else 3  # roll the dice!

    # record various intermediates (needed later for backprop)
    xs.append(x)  # observation
    hs.append(h)  # hidden state
    y = 1 if action == 2 else 0  # a "fake label"
    # grad that encourages the action that was taken to be taken (see http://cs231n.github.io/neural-networks-2/#losses if confused)
    dlogps.append(y - aprob)

    # step the environment and get new measurements
    observation, reward, done, info = env.step(action)
    reward_sum += reward

    # record reward (has to be done after we call step() to get reward for previous action)
    drs.append(reward)

    if done:  # an episode finished
        episode_number += 1
        show_history(history, line1, fig)

        # stack together all inputs, hidden states, action gradients, and rewards for this episode
        epx = np.vstack(xs)
        eph = np.vstack(hs)
        epdlogp = np.vstack(dlogps)
        epr = np.vstack(drs)
        xs, hs, dlogps, drs = [], [], [], []  # reset array memory

        # compute the discounted reward backwards through time
        discounted_epr = discount_rewards(epr)
        # standardize the rewards to be unit normal (helps control the gradient estimator variance)
        discounted_epr -= np.mean(discounted_epr)
        discounted_epr /= np.std(discounted_epr)

        # modulate the gradient with advantage (PG magic happens right here.)
        epdlogp *= discounted_epr
        grad = policy_backward(eph, epdlogp)
        for k in model:
            grad_buffer[k] += grad[k]  # accumulate grad over batch

        # perform rmsprop parameter update every batch_size episodes
        if episode_number % batch_size == 0:
            for k, v in model.items():
                g = grad_buffer[k]  # gradient
                rmsprop_cache[k] = decay_rate * \
                    rmsprop_cache[k] + (1 - decay_rate) * g**2
                model[k] += learning_rate * g / \
                    (np.sqrt(rmsprop_cache[k]) + 1e-5)
                # reset batch gradient buffer
                grad_buffer[k] = np.zeros_like(v)

        # boring book-keeping
        running_reward = reward_sum if running_reward is None else running_reward * \
            0.99 + reward_sum * 0.01
        print('resetting env. episode reward total was %f. running mean: %f' %
              (reward_sum, running_reward))
        history.append((episode_number, reward_sum))
        if episode_number % 10 == 0:
            pickle.dump(model, open('save_latent.p', 'wb'))
            pickle.dump(history, open('history_latent.p', 'wb'))
        reward_sum = 0
        observation = env.reset()  # reset env
        prev_x = None
        prev_latent = None

    if reward != 0:  # Pong has either +1 or -1 reward exactly when game ends.
        print('ep %d: game finished, reward: %f' %
              (episode_number, reward) + ('' if reward == -1 else ' !!!!!!!!'))
