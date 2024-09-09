from flgo.algorithm.fedbase import BasicServer, BasicClient
import copy
import torch
import random
from torch.utils.data import DataLoader
from flgo.utils import fmodule
from torch.utils.data.dataset import Subset
import numpy as np
from flgo.utils import fmodule
class Server(BasicServer):
    def initialize(self, *args, **kwargs):
        self.init_algo_para({'mu':1.0,'p':2, 's':1.0, 'eta':0.1 ,'alpha':0.2,'beta':1.0})
        self.h = self.model.zeros_like()
    def aggregate(self, models):
        if len(models) == 0: return self.model
        nan_exists = [m.has_nan() for m in models]
        if any(nan_exists):
            if all(nan_exists): raise ValueError("All the received local models have parameters of nan value.")
            self.gv.logger.info(
                'Warning("There exists nan-value in local models, which will be automatically removed from the aggregatino list.")')
            new_models = []
            received_clients = []
            for ni, mi, cid in zip(nan_exists, models, self.received_clients):
                if ni: continue
                new_models.append(mi)
                received_clients.append(cid)
            self.received_clients = received_clients
            models = new_models
        self.h = self.h - self.alpha * 1.0 / self.num_clients * (fmodule._model_sum(models) - len(models) * self.model)
        new_model = fmodule._model_average(models) - 1.0 / self.alpha * self.h
        return new_model

class Client(BasicClient):
    def initialize(self, *args, **kwargs):
        self.model = None
        self.weights = None
        self.start_phase = True
        self.num_pre_loss = 10
        self.threshold = 0.1
    def pack(self, model, *args, **kwargs):
        return {
            "model": copy.deepcopy(model)  
        }
    def unpack(self, svr_pkg):
        global_model = svr_pkg['model']
        # initialize local model to the global model if local model is None,
        # and deactivate ALA at the 1st communication iteration by recoginizing the first round automatically
        if self.model is None:
            self.model = copy.deepcopy(global_model)
            return self.model

        # load the global encoder into local model
        params_global = list(global_model.parameters())
        params_local = list(self.model.parameters())
        for param, param_g in zip(params_local, params_global):
            param.data = param_g.data.clone()

        # temp local model only for weight learning
        local_model_temp = copy.deepcopy(self.model)
        params_local_temp = list(local_model_temp.parameters())


        # adaptively aggregate local model and global model by the weight into local temp model's head
        if self.weights is None: self.weights = [torch.ones_like(param.data).to(self.device) for param in params_local]
        for param_t, param, param_g, weight in zip(params_local_temp, params_local, params_global, self.weights):
            param_t.data = param + (param_g - param) * weight
        # weight learning
        # randomly sample partial local training data
        #rand_num = int(self.s * len(self.train_data))
        #rand_idx = random.randint(0, len(self.train_data) - rand_num)
        data_loader = DataLoader(self.train_data, self.batch_size, drop_last=True)
        losses = []  # record losses
        cnt = 0  # weight training iteration counter
        # train local aggregation weights (line 8-14)
        while True:
            for batch_data in data_loader:
                loss = self.calculator.compute_loss(local_model_temp, batch_data)['loss']
                loss.backward()
                # update weight in this batch
                for param_t, param, param_g, weight in zip(params_local_temp, params_local, params_global, self.weights):
                    weight.data = torch.clamp(
                        weight - self.eta * (param_t.grad * (param_g - param)), 0, 1)

                # update temp local model in this batch
                for param_t, param, param_g, weight in zip(params_local_temp, params_local,
                                                           params_global, self.weights):
                    param_t.data = param + (param_g - param) * weight
            losses.append(loss.item())
            cnt += 1
            # only train one epoch in the subsequent iterations
            if not self.start_phase: break
            # train the weight until convergence
            if len(losses) > self.num_pre_loss and np.std(losses[-self.num_pre_loss:]) < self.threshold:
                self.gv.logger.info('Client:{}\tStd:{}\tALA epochs:{}'.format(self.id, np.std(losses[-self.num_pre_loss:]), cnt))
                break
        self.start_phase = False

        for param, param_t in zip(params_local, params_local_temp):
            param.data = param_t.data.clone()
        return global_model
    @fmodule.with_multi_gpus
    def train(self, model):
        model.train()
        # global parameters
        src_model_global = copy.deepcopy(model)
        src_model_global.freeze_grad()
        if self.model is None: self.model = copy.deepcopy(model)
        self.model = self.model.to(self.device)
        self.model.train()
        # global solver
        optimizer_global = self.calculator.get_optimizer(model, lr=self.learning_rate, weight_decay=self.weight_decay, momentum=self.momentum)
        optimizer_local = self.calculator.get_optimizer(self.model, lr = self.learning_rate, weight_decay=self.weight_decay, momentum=self.momentum)
        for iter in range(self.num_steps):
            batch_data = self.get_batch_data()
            # local solver
            self.model.zero_grad()
            loss_local = self.calculator.compute_loss(self.model, batch_data)['loss']
            loss_proximal_local = 0
            for pm, ps in zip(self.model.parameters(), src_model_global.parameters()):
                loss_proximal_local += torch.sum(torch.pow(pm - ps, 2))
            loss_local = loss_local + 0.5 * self.mu * loss_proximal_local
            loss_local.backward()
            optimizer_local.step()

            src_model_local = copy.deepcopy(self.model)
            src_model_local.freeze_grad()

            # global solver
            model.zero_grad()
            loss_proximal_global = 0
            for pm, ps in zip(src_model_local.parameters(), src_model_global.parameters()):
                loss_proximal_global += torch.sum(torch.pow(pm - ps, 2))
            loss_global = self.calculator.compute_loss(model, batch_data)['loss']
            loss_global = loss_global + 0.5 * self.mu * loss_proximal_global
            loss_global.backward()
            optimizer_global.step()

        #self.model = self.model.to('cpu')
        return