import torch.nn as nn
import torch.nn.functional as F
from utils.agent_utils import init_cnn
from utils.agent_utils import init_fc
from torch.distributions import Categorical
import cv2


def get_models():
    return [A3cCNNPolicy]


class A3cCNNPolicy(nn.Module):
    """Policy approximator implemented from A3C paper for descrete_actions"""

    def __init__(self, nr_actions, input_size=4, orthogonal_init=True):
        super(A3cCNNPolicy, self).__init__()

        self.o_init = orthogonal_init
        self.conv_block1 = nn.Sequential(
            init_cnn(
                nn.Conv2d(
                    in_channels=input_size,
                    out_channels=16,
                    kernel_size=8,
                    stride=4,
                    bias=True
                ),
                self.o_init
            ),
            nn.ReLU()
        )
        self.conv_block2 = nn.Sequential(
            init_cnn(
                nn.Conv2d(
                    in_channels=16,
                    out_channels=32,
                    kernel_size=4,
                    stride=2,
                    bias=True
                ),
                self.o_init
            ),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            init_cnn(
                nn.Linear(
                    in_features=2592,
                    out_features=256
                ),
                self.o_init
            ),
            nn.ReLU()
        )
        self.policy_head = init_fc(
            nn.Linear(in_features=256, out_features=nr_actions),
            self.o_init)
        self.value_head = init_fc(
            nn.Linear(in_features=256, out_features=1),
            self.o_init)

    def forward(self, x):
        x = self.conv_block1(x / 255.0)
        x = self.conv_block2(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

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
        dist_prob = F.softmax(logits, dim=1)
        action_log_probs = dist_log_prob.gather(1, actions.view(-1, 1))

        entropy = -(action_log_probs * dist_prob.gather(1, actions.view(-1, 1))).mean()
        return values, action_log_probs, entropy

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

