import gym
import numpy as np
import sys
from itertools import count
from collections import namedtuple
from config import generate_configs
from models import get_models
from logbook import StreamHandler, Logger

import torch
import cv2

import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical


SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])


def to_cuda(data, use_cuda=True):
    input_ = data.float()
    if use_cuda:
        input_ = input_.cuda()
    return input_


def set_seed(seed, use_cuda=True):
    import random
    import numpy as np

    if seed == 0:
        seed = random.randint(1, 9999999)

    torch.manual_seed(seed)
    if use_cuda:
        torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    return seed


def select_action(policy_model, value_model, state, use_cuda=True):

    state = torch.from_numpy(state).float()
    state = to_cuda(state, use_cuda)

    probs = policy_model(state)
    state_value = value_model(state)
    m = Categorical(probs)
    action = m.sample()
    policy_model.saved_actions.append(SavedAction(m.log_prob(action), state_value))
    return action.item()


def finish_episode(policy_model, opt1, opt2, gamma, use_cuda=True):
    R = 0
    saved_actions = policy_model.saved_actions
    eps = np.finfo(np.float32).eps.item()
    policy_losses = []
    value_losses = []
    rewards = []
    import pdb;pdb.set_trace()
    for r in policy_model.rewards[::-1]:
        R = r + gamma * R
        rewards.insert(0, R)
    rewards = to_cuda(torch.tensor(rewards), use_cuda)
    rewards = (rewards - rewards.mean()) / (rewards.std() + eps)
    for (log_prob, value), rr in zip(saved_actions, rewards):
        reward = rr - value.item()
        policy_losses.append(-log_prob * reward)
        value_losses.append(F.smooth_l1_loss(value, to_cuda(torch.tensor([rr]), use_cuda)))
    opt1.zero_grad()
    opt2.zero_grad()
    loss1 = torch.stack(policy_losses).sum()
    loss2 = torch.stack(value_losses).sum()
    loss1.backward(); loss2.backward()
    opt1.step(); opt2.step()
    del policy_model.rewards[:]
    del policy_model.saved_actions[:]


def eval_game(env, policy_estimator, value_estimator,path, epp, use_cuda=True):


    state = env.reset()
    policy_estimator.set_eval_mode()
    value_estimator.set_eval_mode()
    ep_number = 0
    while True:
        action = select_action(policy_estimator, value_estimator, state)
        state, reward, done, _ = env.step(action)
        img = env.render(mode='rgb_array')
        img_path = path + "/img" + str(epp) + "_" + str(ep_number) + ".jpg"
        cv2.imwrite(img_path, img)
        if done:
            break
        ep_number += 1

    del policy_estimator.rewards[:]
    del policy_estimator.saved_actions[:]
    policy_estimator.set_train_mode()
    value_estimator.set_train_mode()


def run(args):

    cfg, run_id, path = args

    # -- Set seed
    #cfg.agent.seed = set_seed(cfg.general.seed)

    env = gym.make(cfg.env.name)
    env.seed(cfg.agent.seed)
    torch.manual_seed(cfg.agent.seed)

    obs_dim = env.observation_space.shape[0]
    n_acts = env.action_space.n

    estimator_mc, value_function_mc = get_models(cfg.agent)

    if not cfg.agent.cuda:
        policy_estimator = estimator_mc[0](obs_dim, n_acts, False)
        value_estimator = value_function_mc[0](obs_dim, 1, False)
    else:
        policy_estimator = estimator_mc[0](obs_dim, n_acts)
        value_estimator = value_function_mc[0](obs_dim, 1)

    policy_optimizer = optim.Adam(policy_estimator.parameters(), lr=cfg.agent.lr)
    value_optimizer = optim.Adam(policy_estimator.parameters(), lr=cfg.agent.lr)

    running_reward = 0
    for i_episode in range(cfg.episodes):
        state = env.reset()
        ep_length = 0
        while True:
            action = select_action(policy_estimator, value_estimator, state, cfg.agent.cuda)
            state, reward, done, _ = env.step(action)
            if cfg.render:
                env.render()
            policy_estimator.rewards.append(reward)
            if done:
                break
            ep_length += 1

        print('Episode {}\tLast length: {:5d}\tAverage length: {:.2f}'.format(
            i_episode, ep_length, running_reward))

        running_reward = running_reward * 0.99 + sum(policy_estimator.rewards) * 0.01
        finish_episode(policy_estimator,
                       policy_optimizer,
                       value_optimizer,
                       cfg.agent.gamma,
                       cfg.agent.cuda)

        if i_episode % cfg.checkpoints.save_freq == 0:
            # save networks
            with open(path + "/policy_network" + str(i_episode), "wb") as f:
                torch.save(policy_estimator.state_dict(), f)

            with open(path + "/value_network" + str(i_episode), "wb") as f:
                torch.save(value_estimator.state_dict(), f)

            print("Saved networks and parameters after " +
                  str(i_episode) + "\n")
            eval_game(env, policy_estimator, value_estimator, path, i_episode)


if __name__ == '__main__':

    log = Logger('train')
    StreamHandler(sys.stdout).push_application()
    log.info("[MODE] Eval agent only")
    log.warn('Logbook is too awesome for most applications')

    # -- Parse config file & generate
    arg_list = generate_configs()
    log.info("Starting...")

    run(arg_list[0])
