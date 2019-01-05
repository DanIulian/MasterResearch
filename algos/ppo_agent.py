import torch
from torch.optim.lr_scheduler import StepLR
from logbook import Logger
from .base_agent import BasePolicyRLAgent
from models import get_models
import torch.nn as nn

NAMESPACE = 'ppo_agent'  # ++ Identifier name for logging
log = Logger(NAMESPACE)

DATA_SAVE_PREFIX = "agent_data"


class PPOAgent(BasePolicyRLAgent):
    def __init__(self, cfg, save_path):
        super(PPOAgent, self).__init__(cfg, save_path)

        # -- Get necessary variables from cfg
        self.cfg = cfg

        # -- Get agent configs
        self.gamma = cfg.agent.gamma                        #discount factor
        self.lam = cfg.agent.lam                            #gae lambda factor
        self.target_kl = cfg.agent.target_kl               #KL max diff between policies
        self.clip_param = cfg.agent.clip_param              #clipping parameter of Objective function - epsilon
        self.ppo_epoch = cfg.training.ppo_epoch
        self.num_mini_batch = cfg.training.num_mini_batch
        self.value_loss_coef = cfg.agent.value_loss_coef
        self.entropy_coef = cfg.agent.entropy_coef

        self.max_grad_norm = cfg.training.max_grad_norm        #clip the gradients of the model
        self.use_clipped_value_loss = cfg.general.use_clipped_value_loss  #clipped value loss

        self.initial_lr = cfg.training.optimizer_args.lr

        super(PPOAgent, self).__end_init__()

    def add_model(self, model):
        self.model = model
        self._models.append(model)

    def add_optimizer(self):
        self.optimizer = self.get_optim(self.cfg.training.optimizer,
                                        self.cfg.training.optimizer_args, self.model)
        self._optimizers.append(self.optimizer)

    def update(self, rollouts):

        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0
        criterion = self._get_criterion
        kl_exit = False

        for e in range(self.ppo_epoch):
            data_generator = rollouts.feed_forward_generator(self.num_mini_batch)

            for sample in data_generator:

                obs_batch, actions_batch, value_batch_old, \
                return_batch, masks_batch, action_log_probs_old, adv_batch = sample

                # Reshape to do in a single forward pass for all steps
                values, action_log_probs, entropy = self.model.evaluate_actions(
                    obs_batch, masks_batch, actions_batch)

                t_loss, a_loss, v_loss, e_loss = \
                    criterion(action_log_probs,
                              action_log_probs_old,
                              adv_batch,
                              values,
                              value_batch_old,
                              return_batch,
                              entropy)

                value_loss_epoch += v_loss
                action_loss_epoch += a_loss
                dist_entropy_epoch += e_loss
                #compute estimate of KL divergence and stop updating if too big
                #approx_kl = (action_log_probs_old - action_log_probs).mean()
                #if approx_kl > 1.5 * self.target_kl:
                #    log.info('KL divergence between the old and new plicy exceeded limit, %s STOP UPDATING',
                #             approx_kl)
                #    kl_exit = True
                #    break

            #if kl_exit:
            #    break

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
                       ret_batch,
                       entropy_loss):

        #first term of objective function is the surrogate action objective
        ratio = torch.exp(act_log_probs - act_log_probs_old) # pi(a|s) / pi_old(a|s)
        term1 = ratio * adv_batch
        term2 = torch.clamp(ratio, 1.0 - self.clip_param,
                            1.0 + self.clip_param) * adv_batch

        action_loss = -torch.min(term1, term2).mean()

        #second term objective function is the loss value function
        if self.use_clipped_value_loss:
            value_pred_clipped = val_batch_old + \
                                 (val_batch_new - val_batch_old).clamp(
                                     -self.clip_param, self.clip_param)

            value_losses = torch.pow((val_batch_new - ret_batch), 2)
            value_losses_clipped = torch.pow((value_pred_clipped - ret_batch), 2)

            value_loss = torch.max(value_losses, value_losses_clipped).mean() # max because you minimze the total
        else:
            value_loss = torch.pow((val_batch_new - ret_batch), 2).mean()

        total_loss = action_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_loss

        self.optimizer.zero_grad()

        total_loss.backward()


        nn.utils.clip_grad_norm_(self.model.parameters(),
                                 self.max_grad_norm)

        self.optimizer.step()

        return total_loss.item(), action_loss.item(), value_loss.item(), entropy_loss.item()

    def update_linear_schedule(self, epoch, total_num_epochs):
        lr = self.initial_lr - (self.initial_lr * (epoch / float(total_num_epochs)))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def _save(self, save_data, path):

        # Save model
        """
        Called when saving agent state. Agent already saves variables defined in the list
        self._save_data and other default options.
        :param save_data: Pre-loaded dictionary with saved data. Append here other data
        :param path: Path to folder where other custom data can be saved
        :return: should return default save_data dictionary to be saved
        """
        pass

    def _resume(self, agent_check_point_path, saved_data, device):
        """
        Custom resume scripts should pe implemented here
        :param agent_check_point_path: Path of the checkpoint resumed
        :param saved_data: loaded checkpoint data (dictionary of variables)
        """
        self.model = self._models[0]
        self.optimizer = self._optimizers[0]
        self._train_epoch = saved_data['train_epoch']
        self.to(device)
