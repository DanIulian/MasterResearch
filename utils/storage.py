import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler


def _flatten_helper(T, N, _tensor):
    return _tensor.view(T * N, *_tensor.size()[2:])


class RolloutStorage(object):

    def __init__(self, num_steps, num_processes, obs_shape):

        self.obs_buf = torch.zeros(num_steps + 1, num_processes, *obs_shape)
        self.rew_buf = torch.zeros(num_steps, num_processes, 1)
        self.val_buf = torch.zeros(num_steps + 1, num_processes, 1)
        self.ret_buf = torch.zeros(num_steps + 1, num_processes, 1)
        self.adv_buf = torch.zeros(num_steps + 1, num_processes, 1)
        self.action_log_probs = torch.zeros(num_steps, num_processes, 1)
        self.act_buf = torch.zeros(num_steps, num_processes, 1).long()
        self.masks = torch.ones(num_steps + 1, num_processes, 1)

        self.num_steps = num_steps
        self.step = 0

    def to(self, device):

        self.obs_buf = self.obs_buf.to(device)
        self.rew_buf = self.rew_buf.to(device)
        self.val_buf = self.val_buf.to(device)
        self.ret_buf = self.ret_buf.to(device)
        self.action_log_probs = self.action_log_probs.to(device)
        self.act_buf = self.act_buf.to(device)
        self.masks = self.masks.to(device)
        self.adv_buf = self.adv_buf.to(device)

    def insert(self, obs, actions, action_log_probs, value_preds, rewards, masks):

        self.obs_buf[self.step + 1].copy_(obs)
        self.act_buf[self.step].copy_(actions)
        self.action_log_probs[self.step].copy_(action_log_probs)
        self.val_buf[self.step].copy_(value_preds)
        self.rew_buf[self.step].copy_(rewards)
        self.masks[self.step + 1].copy_(masks)
        self.step = (self.step + 1) % self.num_steps

    def after_update(self):
        self.obs_buf[0].copy_(self.obs_buf[-1])
        self.masks[0].copy_(self.masks[-1])

    def compute_returns(self, gamma, lam, next_value=0, eps=1e-5):
        self.val_buf[-1] = next_value
        gae = 0
        for step in reversed(range(self.rew_buf.size(0))):
            delta = self.rew_buf[step] + \
                    gamma * self.val_buf[step + 1] * self.masks[step + 1] - \
                    self.val_buf[step]
            gae = delta + gamma * lam * self.masks[step + 1] * gae
            self.adv_buf[step] = gae
            self.ret_buf[step] = gae + self.val_buf[step]
        self.adv_buf = (self.adv_buf - self.adv_buf.mean()) / (self.adv_buf.std() + eps)

    def feed_forward_generator(self, num_mini_batch):

        num_steps, num_processes = self.rew_buf.size()[0:2]
        batch_size = num_processes * num_steps

        assert batch_size >= num_mini_batch, (
            "PPO requires the number of processes ({}) "
            "* number of steps ({}) = {} "
            "to be greater than or equal to the number of PPO mini batches ({})."
            "".format(num_processes, num_steps, num_processes * num_steps, num_mini_batch))

        mini_batch_size = batch_size // num_mini_batch
        sampler = BatchSampler(SubsetRandomSampler(range(batch_size)), mini_batch_size, drop_last=False)

        for indices in sampler:
            obs_batch = self.obs_buf[:-1].view(-1, *self.obs_buf.size()[2:])[indices]
            actions_batch = self.act_buf.view(-1, self.act_buf.size(-1))[indices]
            value_preds_batch = self.val_buf[:-1].view(-1, 1)[indices]
            return_batch = self.ret_buf[:-1].view(-1, 1)[indices]
            adv_batch = self.adv_buf[:-1].view(-1, 1)[indices]
            masks_batch = self.masks[:-1].view(-1, 1)[indices]
            log_probs_old_batch = self.action_log_probs.view(-1, 1)[indices]

            yield obs_batch, actions_batch, value_preds_batch, \
                  return_batch, masks_batch, log_probs_old_batch, adv_batch

