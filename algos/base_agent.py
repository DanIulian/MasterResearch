from __future__ import print_function
import os
import torch
from logbook import Logger


NAMESPACE = 'learning_agent'
log = Logger(NAMESPACE)

MODELS_KEY = "models"
OPTIMIZERS_KEY = "optimizers"
TRAIN_EPOCH_KEY = "train_epoch"
DATA_SAVE_PREFIX = "agent_data"


class BasePolicyRLAgent():
    def __init__(self, cfg, save_path):

        self._save_path = save_path
        self._resume_path = cfg.resume

        # -- models and optimizers should be loaded after initialization in this list
        self._models = []
        self._optimizers = []

        self._is_train = True
        self._eval_agent = False

        # Dictionary generator for saving data
        self._save_data = [
            "_models", "_optimizers",
        ]

    def __end_init__(self):
        if self._resume_path:
            self.resume(self._resume_path)

    def session_init(self):
        if self._is_train:
            for opt in self._optimizers:
                opt.zero_grad()

    def set_eval_mode(self):
        """ Set agent to evaluation mode"""
        if self._is_train:
            for m in self._models:
                m.set_eval_mode()
        self._is_train = False

    def set_train_mode(self):
        """ Set agent to training mode """
        if not self._is_train:
            for m in self._models:
                m.set_train_mode()
        self._is_train = True

    def to(self, device):
        for model in self._models:
            model.to(device)

    def to_cpu(self):
        for model in self._models:
            model.cpu()

    @property
    def is_training(self):
        return self._is_train

    def save(self, prefix="agent_data_"):
        save_data = {key: self.__dict__[key] for key in self._save_data}
        save_data = self._save(save_data, self._save_path)
        torch.save(save_data, os.path.join(self._save_path, prefix))

    def _save(self, save_data, path):
        return save_data

    def resume(self, agent_check_point_path):

        log.info("Resuming agent from {}".format(agent_check_point_path))
        data = torch.load(agent_check_point_path)

        # Resume save data
        for key in self._save_data:
            self.__dict__[key] = data[key]

        self._resume(agent_check_point_path, data)

    def _resume(self, agent_check_point_path, saved_data):
        pass

    @staticmethod
    def get_optim(algorithm, algorithm_args, model):
        _optimizer = getattr(torch.optim, algorithm)
        optim_args = vars(algorithm_args)
        return _optimizer(model.parameters(), **optim_args)
