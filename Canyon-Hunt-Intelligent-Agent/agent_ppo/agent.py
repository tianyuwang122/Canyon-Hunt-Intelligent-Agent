#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Agent class for Gorge Chase PPO.
峡谷追猎 PPO Agent 主类。
"""

import torch

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

import numpy as np
from kaiwudrl.interface.agent import BaseAgent

from agent_ppo.algorithm.algorithm import Algorithm
from agent_ppo.conf.conf import Config
from agent_ppo.feature.definition import ActData, ObsData
from agent_ppo.feature.preprocessor import Preprocessor
from agent_ppo.model.model import Model


class Agent(BaseAgent):
    def __init__(self, agent_type="player", device=None, logger=None, monitor=None):
        import torch
        import os
        
        # 核心逻辑：利用环境变量判断当前所属容器/微服务角色
        # KaiwuDRL 生成环境通常会注入 'KAIWU_ROLE' 或者 'svr_name' 之类的环境变量
        current_role = os.environ.get("KAIWU_ROLE", os.environ.get("svr_name", "unknown"))
        
        # 兜底：如果连跑的是 local 训练还是容器环境都不知道，
        # 则根据是否检测到配置了 GPU 环境参数来进行判定。
        # 只有在非常明确是 learner 节点的情况下，才强行进行拦截。
        if current_role == "learner" or "learner" in str(device).lower():
            if not torch.cuda.is_available():
                raise RuntimeError("【GPU 训练限制拦截】发现 learner 训练节点试图在 CPU 上启动！系统拒绝，请检查你的 RTX 4060 及 CUDA 容器挂载配置！")
            device = "cuda"  # Learner 强锁 GPU
        else:
            # 兼容 aisrv, actor 以及本地调试环境 (默认放行)
            pass

        torch.manual_seed(0)
        self.device = device
        self.model = Model(device).to(self.device)
        self.optimizer = torch.optim.Adam(
            params=self.model.parameters(),
            lr=Config.INIT_LEARNING_RATE_START,
            betas=(0.9, 0.999),
            eps=1e-8,
        )
        self.algorithm = Algorithm(self.model, self.optimizer, self.device, logger, monitor)
        self.preprocessor = Preprocessor()
        self.last_action = -1
        self.logger = logger
        self.monitor = monitor
        super().__init__(agent_type, device, logger, monitor)

    def reset(self, env_obs=None):
        """Reset per-episode state.

        每局开始时重置状态。
        """
        self.preprocessor.reset()
        self.last_action = -1

    def observation_process(self, env_obs):
        """Convert raw env_obs to ObsData and remain_info.

        将原始观测转换为 ObsData 和 remain_info。
        """
        feature, legal_action, reward = self.preprocessor.feature_process(env_obs, self.last_action)
        obs_data = ObsData(
            feature=list(feature),
            legal_action=legal_action,
        )
        remain_info = {"reward": reward}
        return obs_data, remain_info

    def predict(self, list_obs_data):
        """Stochastic inference for training (exploration).

        训练时随机采样动作（探索）。
        """
        feature = list_obs_data[0].feature
        legal_action = list_obs_data[0].legal_action

        logits, value, prob = self._run_model(feature, legal_action)

        action = self._legal_sample(prob, use_max=False)
        d_action = self._legal_sample(prob, use_max=True)

        return [
            ActData(
                action=[action],
                d_action=[d_action],
                prob=list(prob),
                value=value,
            )
        ]

    def exploit(self, env_obs):
        """Greedy inference for evaluation.

        评估时贪心选择动作（利用）。
        """
        obs_data, _ = self.observation_process(env_obs)
        act_data = self.predict([obs_data])
        return self.action_process(act_data[0], is_stochastic=False)

    def learn(self, list_sample_data):
        """Train the model.

        训练模型。
        """
        return self.algorithm.learn(list_sample_data)

    def save_model(self, path=None, id="1"):
        """Save model checkpoint.

        保存模型检查点。
        """
        model_file_path = f"{path}/model.ckpt-{str(id)}.pkl"
        state_dict_cpu = {k: v.clone().cpu() for k, v in self.model.state_dict().items()}
        torch.save(state_dict_cpu, model_file_path)
        self.logger.info(f"save model {model_file_path} successfully")

    def load_model(self, path=None, id="1"):
        """Load model checkpoint.

        加载模型检查点。
        """
        model_file_path = f"{path}/model.ckpt-{str(id)}.pkl"
        self.model.load_state_dict(torch.load(model_file_path, map_location=self.device))
        self.logger.info(f"load model {model_file_path} successfully")

    def action_process(self, act_data, is_stochastic=True):
        """Unpack ActData to int action and update last_action.

        解包 ActData 为 int 动作并记录 last_action。
        """
        action = act_data.action if is_stochastic else act_data.d_action
        self.last_action = int(action[0])
        return int(action[0])

    def _run_model(self, feature, legal_action):
        """Run model inference, return logits, value, prob.

        执行模型推理，返回 logits、value 和动作概率。
        """
        self.model.set_eval_mode()
        obs_tensor = torch.tensor(np.array([feature]), dtype=torch.float32).to(self.device)

        with torch.no_grad():
            logits, value = self.model(obs_tensor, inference=True)

        logits_np = logits.cpu().numpy()[0]
        value_np = value.cpu().numpy()[0]

        # Legal action masked softmax / 合法动作掩码 softmax
        legal_action_np = np.array(legal_action, dtype=np.float32)
        prob = self._legal_soft_max(logits_np, legal_action_np)

        return logits_np, value_np, prob

    def _legal_soft_max(self, input_hidden, legal_action):
        """Softmax with legal action masking (numpy).

        合法动作掩码下的 softmax（numpy 版）。
        """
        _w, _e = 1e20, 1e-5
        tmp = input_hidden - _w * (1.0 - legal_action)
        tmp_max = np.max(tmp, keepdims=True)
        tmp = np.clip(tmp - tmp_max, -_w, 1)
        tmp = (np.exp(tmp) + _e) * legal_action
        return tmp / (np.sum(tmp, keepdims=True) * 1.00001)

    def _legal_sample(self, probs, use_max=False):
        """Sample action from probability distribution.

        按概率分布采样动作，使用 float64 与 CDF 避免概率相加超 1.0 的问题。
        """
        if use_max:
            return int(np.argmax(probs))
        pvals = np.array(probs, dtype=np.float64)
        pvals = np.clip(pvals, 0.0, 1.0)
        p_sum = np.sum(pvals)
        if p_sum > 0:
            pvals /= p_sum
        else:
            pvals = np.ones_like(pvals) / len(pvals)
        
        cdf = np.cumsum(pvals)
        r = np.random.rand()
        return int(np.searchsorted(cdf, r))
