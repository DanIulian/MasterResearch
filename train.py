import time

from envs import make_vec_envs
from utils.storage import  RolloutStorage
from algos.ppo_agent import  PPOAgent
from utils.agent_utils import save_reward_plot, to_cuda

import numpy as np
import sys

from config import generate_configs
from logbook import StreamHandler, Logger
from models import get_models

import torch
import cv2

log = Logger('train')

def set_seed(seed, use_cuda=True):
    import random
    import numpy as np

    if seed == 0:
        seed = random.randint(1, 9999999)

    torch.manual_seed(seed)
    if use_cuda:
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

    return seed


def eval_learning_process(agent, cfg, device, show=False):

    nn_model = agent.model
    mean_reward = 0.0
    for i in range(5):

        eval_env = make_vec_envs(cfg.env.name,
                            cfg.general.seed,
                            1,
                            cfg.agent.gamma,
                            cfg.env.add_timestep,
                            device, cfg.env.nr_frames)

        obs = eval_env.reset()
        eval_masks = torch.zeros(1, 1, device=device)
        ep_reward = 0.0
        nr_steps = 0
        actions = []
        final_done = 0
        while final_done < 5:
            if show:
                img  = obs.cpu().numpy()[0, 3, :, :]
                img = cv2.resize(img, (480, 480), interpolation=cv2.INTER_AREA).astype(np.uint8)
                cv2.imshow("game", img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            obs = to_cuda(obs, cfg.general.cuda)
            action = nn_model.get_best_action(obs)
            actions.append(action.item())

            # Obser reward and next obs
            obs, reward, done, _ = eval_env.step(action)

            ep_reward += reward
            nr_steps += 1
            if done[0]:
                final_done += done[0]

        eval_env.close()
        print("Episode length {}, {}".format(nr_steps, ep_reward))
        mean_reward += ep_reward

    cv2.destroyAllWindows()

    return mean_reward / 5.0


def run(args):

    cfg, path = args

    # -- Set seed
    cfg.agent.seed = set_seed(cfg.general.seed)

    num_updates = int(cfg.training.num_env_steps) \
                  // cfg.training.num_steps // cfg.training.num_processes

    torch.set_num_threads(1)
    if torch.cuda.is_available() and cfg.general.cuda:
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    envs = make_vec_envs(cfg.env.name,
                         cfg.general.seed,
                         cfg.training.num_processes,
                         cfg.agent.gamma,
                         cfg.env.add_timestep, device, cfg.env.nr_frames)

    rollouts = RolloutStorage(cfg.training.num_steps,
                              cfg.training.num_processes,
                              envs.observation_space.shape)

    if cfg.algo == 'a2c':
        pass
    elif cfg.algo == 'ppo':
        agent = PPOAgent(cfg, path)

    model_class = get_models(cfg.agent)
    nn_model = model_class[0](
        envs.action_space.n,
        cfg.agent.input_space,
        cfg.agent.use_initialization
    )
    agent.add_model(nn_model)
    agent.add_optimizer()

    agent.to(device)
    agent.set_train_mode()

    obs = envs.reset()
    rollouts.obs_buf[0].copy_(obs)
    rollouts.to(device)

    training_rewards = []
    trained_steps = []
    for j in range(num_updates):

        if cfg.general.use_linear_lr_decay:
            agent.update_linear_schedule(j, num_updates)

        if cfg.algo == 'ppo' and cfg.general.use_linear_clip_decay:
            agent.clip_param = cfg.agent.clip_param * (1 - j / float(num_updates))
        for step in range(cfg.training.num_steps):
            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob = nn_model.act(rollouts.obs_buf[step],
                                                              rollouts.masks[step])

            # Obser reward and next obs
            obs, reward, done, infos = envs.step(action)


            # If done then clean the history of observations.
            masks = torch.FloatTensor([[0.0] if done_ else [1.0]
                                       for done_ in done])
            action = action.unsqueeze(dim=1)
            action_log_prob = action_log_prob.unsqueeze(dim=1)
            rollouts.insert(obs, action, action_log_prob, value, reward, masks)


        with torch.no_grad():
            next_value = nn_model.get_value(rollouts.obs_buf[-1]).detach()

        rollouts.compute_returns(cfg.agent.gamma, cfg.agent.lam, next_value)

        value_loss, action_loss, dist_entropy = agent.update(rollouts) # use KL divergence for early stop

        rollouts.after_update()

        total_num_steps = (j + 1) * cfg.training.num_processes * cfg.training.num_steps
        # save for every interval-th episode or for the last epoch
        if j % cfg.save_interval == 0 or j == num_updates - 1:
            agent.to_cpu()
            agent.save()
            agent.to(device)


        log.info("Updates {}, num_steps {} value_loss {}, action_loss {}, dist_entropy {}".format(
            j, total_num_steps,
            value_loss,
            action_loss,
            dist_entropy
        ))

        if j % cfg.eval_interval == 0:
            log.info("Save agent progress after {}".format(total_num_steps))
            agent.set_eval_mode()
            if j > 1500:
                rew = eval_learning_process(agent, cfg, device, True)
            else:
                rew = eval_learning_process(agent, cfg, device)
            training_rewards.append(rew)
            trained_steps.append(total_num_steps)
            save_reward_plot(trained_steps, training_rewards, cfg.env.name, path)
            agent.set_train_mode()


if __name__ == "__main__":

    StreamHandler(sys.stdout).push_application()
    log.info("[MODE] Train agent")
    log.warn('Logbook is too awesome for most applications')

    # -- Parse config file & generate
    arg_list = generate_configs()
    log.info("Starting...")
    run(arg_list[0])
