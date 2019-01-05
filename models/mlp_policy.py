import torch.nn as nn
import torch.nn.functional as F
from utils.agent_utils import init_mlp
from torch.distributions import Categorical

def get_models():
    return [MLPPolicy]


class MLPPolicy(nn.Module):
    """Policy approximator implemented from PPO paper for discrete_actions"""

    def __init__(self, nr_actions, input_size, orthogonal_init=True):
        super(MLPPolicy, self).__init__()

        self.o_init = orthogonal_init
        self.fc1 = nn.Sequential(
            init_mlp(
                nn.Linear(in_features=input_size, out_features=64),
                self.o_init
            ),
            nn.Tanh()
        )

        self.fc2 = nn.Sequential(
            init_mlp(
                nn.Linear(in_features=64, out_features=64),
                self.o_init
            ),
            nn.Tanh()
        )

        self.policy_head = init_mlp(
            nn.Linear(in_features=64, out_features=nr_actions),
            self.o_init)
        self.value_head = init_mlp(
            nn.Linear(in_features=64, out_features=1),
            self.o_init)

    def forward(self, x):

        x = self.fc1(x)
        x = self.fc2(x)

        logits = self.policy_head(x)
        value = self.value_head(x)

        return logits, F.softmax(logits, dim=-1), value

    def set_eval_mode(self):
        """ Set agent to evaluation mode"""
        self.eval()

    def set_train_mode(self):
        """ Set agent to training mode """
        self.train()

    def evaluate_actions(self, inputs, masks, actions):

        logits, prob_pi, values = self.forward(inputs)
        dist_log_prob = F.log_softmax(logits, dim=1)
        action_log_probs = dist_log_prob.gather(1, actions.view(-1, 1))
        return values, action_log_probs

    def act(self, inputs, masks, deterministic=False):
        logits, probs_pi, values = self.forward(inputs)

        m = Categorical(probs_pi)
        actions = m.sample()
        return values, actions, m.log_prob(actions)

    def get_best_action(self, inputs):
        logits, probs_pi, values = self.forward(inputs)
        return probs_pi.argmax()

    def get_value(self, inputs):
        _, _, values = self.forward(inputs)

        return values

