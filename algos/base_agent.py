from __future__ import print_function
import os
import torch
from logbook import Logger
import subprocess
import numpy as np
from tensorboardX import SummaryWriter


NAMESPACE = 'learning_agent'
log = Logger(NAMESPACE)

MODELS_KEY = "models"
OPTIMIZERS_KEY = "optimizers"
TRAIN_EPOCH_KEY = "train_epoch"
DATA_SAVE_PREFIX = "agent_data"


class BasePolicyRLAgent():
    def __init__(self, cfg):
        super(BasePolicyRLAgent, self).__init__(cfg.name, verbose=cfg.verbose)

        self._use_cuda = cfg.use_cuda
        self._save_path = cfg.common.save_path
        self._save_freq = cfg.save_freq
        self._last_ep_saved_best = -np.inf
        self._resume_path = cfg.resume


        # -- models and optimizers should be loaded after initialization in this list
        self._models = []
        self._optimizers = []

        self._is_train = True
        self._eval_agent = False
        self._train_epoch = 0


        # Dictionary generator for saving data
        self._save_data = [
            "_models", "_optimizers", "_train_epoch",
        ]

        #self._writer = SummaryWriter(log_dir=self._save_path, comment="tensorboard")
        #self._start_tensorboard()

        #

    def __end_init__(self):
        if self._resume_path:
            self.resume(self._resume_path)

    def session_init(self):
        return self._session_init()

    def _session_init(self):
        raise NotImplemented

    def run_step(self, measurements, sensor_data, directions, target):
        return self._run_step(measurements, sensor_data, directions, target)

    def train(self, dataloader):
        log.info("--> Train")
        self.set_train_mode()
        self.session_init()
        self._train_epoch += 1
        loss, other = self._train(dataloader)

        return loss, other

    def _train(self, dataloader):
        raise NotImplemented

    def test(self, dataloader):
        log.info("--> Test")
        train_epoch = self._train_epoch

        self.set_eval_mode()
        self.session_init()

        score, is_best, other = self._test(dataloader)

        return score, is_best, other

    def _test(self, dataloader):
        raise NotImplemented

    def set_eval_mode(self):
        """ Set agent to evaluation mode"""
        if self._is_train:
            for m in self._models:
                m.set_eval_mode()
            self._set_eval_mode()
        self._is_train = False

    def _set_eval_mode(self):
        pass

    def set_train_mode(self):
        """ Set agent to training mode """
        if not self._is_train:
            for m in self._models:
                m.set_train_mode()
            self._set_train_mode()
        self._is_train = True

    def _set_train_mode(self):
        pass

    @property
    def is_training(self):
        return self._is_train

    def cuda(self):
        """ Set agent to run on CUDA """
        models = self._models
        for m in models:
            m.cuda()

    def cpu(self):
        models = self._models
        for m in models:
            m.cpu()

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

    def _start_tensorboard(self, kill_other=True):
        if kill_other:
            os.system("killall -9 tensorboard")

        save_path = self._save_path
        logdir = "--logdir" + "=" + save_path
        port = "--port" + "=" + str(8008)
        subprocess.Popen(["tensorboard", logdir, port])

    def _end_tensorboard(self):
        os.sysyem("killall -2 tensorboard")

