import torch.nn as nn
import torch.nn.functional as F
from utils.agent_utils import init_cnn, init_fc
from torch.distributions import Categorical


def get_models():
    return [AtariCNNPolicy]


class AtariCNNPolicy(nn.Module):
    """Policy approximator implemented from DQN Nature paper for discrete_actions"""

    def __init__(self, nr_actions, input_size=4, orthogonal_init=True):
        super(AtariCNNPolicy, self).__init__()

        self.o_init = orthogonal_init
        self.conv_block1 = nn.Sequential(
            init_cnn(
                nn.Conv2d(
                    in_channels=input_size,
                    out_channels=32,
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
                    in_channels=32,
                    out_channels=64,
                    kernel_size=4,
                    stride=2,
                    bias=True
                ),
                self.o_init
            ),
            nn.ReLU()
        )

        self.conv_block3 = nn.Sequential(
            init_cnn(
                nn.Conv2d(
                    in_channels=64,
                    out_channels=64,
                    kernel_size=3,
                    stride=1,
                    bias=True
                ),
                self.o_init
            ),
            nn.ReLU()
        )

        self.fc = nn.Sequential(
            init_cnn(
                nn.Linear(
                    in_features=3136,
                    out_features=512
                ),
                self.o_init
            ),
            nn.ReLU()
        )
        self.policy_head = init_fc(
            nn.Linear(in_features=512, out_features=nr_actions),
            self.o_init)
        self.value_head = init_fc(
            nn.Linear(in_features=512, out_features=1),
            self.o_init)

    def forward(self, x):

        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
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
        import pdb; pdb.set_trace()

        #dist = self.dist(actor_features)

        #action_log_probs = dist.log_probs(action)
        #dist_entropy = dist.entropy().mean()

        #return value, action_log_probs, dist_entropy, rnn_hxs


    def act(self, inputs, masks, deterministic=False):
        logits, probs_pi, values = self.model(inputs)

        m = Categorical(probs_pi)
        actions = m.sample()
        return values, actions, m.log_prob(actions)

    def get_value(self, inputs):
        _, _, values = self.forward(inputs)

        return values

