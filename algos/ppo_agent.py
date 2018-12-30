import torch
from torch.optim.lr_scheduler import StepLR
from logbook import Logger
import gym

from torch.distributions import Categorical
from .base_agent import BasePolicyRLAgent
from models import get_models

import torch.nn as nn
from utils.agent_utils import  to_cuda

NAMESPACE = 'ppo_agent'  # ++ Identifier name for logging
log = Logger(NAMESPACE)

DATA_SAVE_PREFIX = "agent_data"



class PPOAgent(BasePolicyRLAgent):
    def __init__(self, cfg):
        super(PPOAgent, self).__init__(cfg)

        # -- Get necessary variables from cfg
        self.cfg = cfg
        self.gamma = cfg.agent.gamma
        self.lam = cfg.agent.lam
        self.target_kl  = cfg.agent.target_kl
        self.clip_param = cfg.agent.clip_param
        self.ppo_epoch = cfg.agent.ppo_epoch
        self.num_mini_batch = cfg.agent.num_mini_batch
        self.value_loss_coef = cfg.agent.value_loss_coef
        self.entropy_coef = cfg.agent.entropy_coef

        self.max_grad_norm = cfg.agent.max_grad_norm
        self.use_clipped_value_loss = cfg.agent.use_clipped_value_loss

        # -- Make the environment
        self.env = gym.make(cfg.environment)
        self.obs_dim = self.env.observation_space.shape
        self.act_dim = self.env.action_space.shape

        # -- Initialize model
        model_class = get_models(cfg.model)

        self.model = model_class[0](
            self.env.action_space.n,
            self.cfg.agent.input_size,
            cfg.agent.use_initialization)

        self._models.append(
            self.model)  # -- Add models & optimizers to base for saving

        # ++ After adding model you can set the agent to cuda mode
        # ++ Parent class already makes some adjustments. E.g. turns model to cuda mode
        if self._use_cuda:
            self.cuda()

        # -- Initialize optimizers  - USE MULTIPLE OPTIMIZERS ???
        self.optimizer = self.get_optim(cfg.train.algorithm,
                                        cfg.train.algorithm_args, self.model)

        self._optimzers.append(
            self.optimizer ) # -- Add models & optimizers to base for saving

        #self.scheduler = StepLR(self.optimizer, cfg.train.step_size,
        #                        cfg.train.decay)

        # ++ E.g. to add variable name to be saved at checkpoints
        #self._save_data.append("scheduler")

        super(PPOAgent, self).__end_init__()

    def _session_init(self):
        if self._is_train:
            for opt in self._optimizers:
                opt.zero_grad()

    def _encode_recent_frames(self, obs_history):
        pass

    def _select_action(self, observation):

        obs_history = self.cfg.agent.obs_history
        if obs_history > 1:
            obs = self._encode_recent_frames(obs_history)
        else:
            obs = torch.from_numpy(observation).float()
            obs = to_cuda(obs, self._use_cuda)

        with torch.no_grad():
            logits, probs_pi, value = self.model(obs)

            m = Categorical(probs_pi)
            action = m.sample()
        return action, value, m.log_prob(action), logits, probs_pi

    def _train(self):
        """
        Considering a dataloader (loaded from config.)
        Implement the training loop.
        :return training loss metric & other information
        """
        optimizer = self.optimizer
        use_cuda = self._use_cuda
        model = self.model
        criterion = self._get_criterion
        env = self.env
        steps_per_epoch = self.cfg.agent.epoch_steps
        epochs = self.cfg.agent.nr_epochs
        max_ep_len = self.cfg.agent.max_ep_len

        obs, rew, done, ep_ret, ep_len = env.reset(), 0, False, 0, 0
        for epoch in range(epochs):

            # Main loop: collect experience in env and update/log each epoch
            for t in range(steps_per_epoch):

                act, val, logp_t, logits, probs_pi = self._select_action(obs)

                # save and log
                self._store(obs, act, rew, val, logp_t)

                obs, rew, done, _ = env.step(act.item())
                ep_ret += rew
                ep_len += 1

                if done:
                    print(
                        "Episode ran for {} steps and the agent received total reward {}".format(ep_len, ep_ret))

                terminal = done or (ep_len == max_ep_len)
                if terminal or (t == steps_per_epoch - 1):
                    if not (terminal):
                        print('Warning: trajectory cut off by epoch at %d steps.' % ep_len)
                    # if trajectory didn't reach terminal state, bootstrap value target
                    if done:
                        last_val = rew
                    else:
                        _, last_val, _, _, _ = self._select_action(obs)

                    self._finish_path(last_val)

                    obs, rew, done, ep_ret, ep_len = env.reset(), 0, False, 0, 0

            # Save model
            if (epoch % self._save_freq == 0) or (epoch == epochs - 1):
                self.save(prefix=DATA_SAVE_PREFIX + "_{}".format(epoch))

            # Perform PPO update!
            self._update()


    def _update(self, rollouts):

        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0
        criterion = self._get_criterion

        for e in range(self.ppo_epoch):
                data_generator = rollouts.feed_forward_generator(self.num_mini_batch)

            for sample in data_generator:

                obs_batch, actions_batch, value_batch_old, \
                return_batch, masks_batch, log_probs_old_batch, adv_batch = sample


                # Reshape to do in a single forward pass for all steps
                values, action_log_probs, dist_entropy, _ = self.model.evaluate_actions(
                    obs_batch, masks_batch, actions_batch)

                t_loss, a_loss, v_loss, e_loss = \
                    criterion(action_log_probs,
                              log_probs_old_batch,
                              adv_batch,
                              values,
                              value_batch_old,
                              return_batch)

                value_loss_epoch += v_loss
                action_loss_epoch += a_loss
                dist_entropy_epoch += e_loss

                #compute estimate of KL divergence and stop updating if too big
                approx_kl = (log_probs_old_batch - action_log_probs).mean()
                if approx_kl > 1.5 * self.target_kl:
                    break

        num_updates = self.ppo_epoch * self.num_mini_batch

        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates

        return value_loss_epoch, action_loss_epoch, dist_entropy_epoch

    def _get_criterion(self,
                       act_log_probs,
                       act_log_probs_old,
                       adv_batch,
                       val_batch_old,
                       val_batch_new,
                       ret_batch):


        #first term of objective function is the surrogate action objective
        ratio = torch.exp(act_log_probs - act_log_probs_old) # pi(a|s) / pi_old(a|s)
        term1 = ratio * adv_batch
        term2 = torch.clamp(ratio, 1.0 - self.clip_param,
                            1.0 + self.clip_param) * adv_batch

        action_loss = -torch.min(term1, term2).mean()

        #second term objective function is the loss value function
        if self.use_clipped_value_loss:
            value_pred_clipped = torch.clamp(
                val_batch_new, val_batch_old - self.clip_param,
                val_batch_old + self.clip_param)

            value_losses = torch.pow((val_batch_new - ret_batch), 2)
            value_losses_clipped = torch.pow((value_pred_clipped - ret_batch), 2)

            value_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean() # max because you minimze the total
        else:
            value_loss = 0.5 * torch.pow((val_batch_new - ret_batch), 2).mean()

        #third term is the entropy loss
        entropy_loss = -act_log_probs.mean()

        total_loss = action_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_loss
        total_loss.backward()

        nn.utils.clip_grad_norm_(self.model.parameters(),
                                 self.max_grad_norm)

        for opt in self._optimizers:
            opt.zero_grad()
            opt.step()

        return total_loss.item(), action_loss.item(), value_loss.item(), entropy_loss.item()


    def _save(self, save_data, path):
        """
        Called when saving agent state. Agent already saves variables defined in the list
        self._save_data and other default options.
        :param save_data: Pre-loaded dictionary with saved data. Append here other data
        :param path: Path to folder where other custom data can be saved
        :return: should return default save_data dictionary to be saved
        """
        save_data['scheduler_state'] = self.scheduler.state_dict()
        save_data['train_epoch'] = self._train_epoch
        save_data['loss_value_train'] = self.loss_values_train
        save_data['loss_value_test'] = self.loss_values_test

        return save_data

    def _resume(self, agent_check_point_path, saved_data):
        """
        Custom resume scripts should pe implemented here
        :param agent_check_point_path: Path of the checkpoint resumed
        :param saved_data: loaded checkpoint data (dictionary of variables)
        """
        self.scheduler.load_state_dict(saved_data['scheduler_state'])
        self.scheduler.optimizer = self.optimizer
        self.model = self._models[0]
        self.optimizer = self._optimizers[0]
        self._train_epoch = saved_data['train_epoch']
        #self.loss_values_train = saved_data['loss_value_train']
        #self.loss_values_test = saved_data['loss_value_test']
        if not self._use_cuda:
            self.model.cpu()
