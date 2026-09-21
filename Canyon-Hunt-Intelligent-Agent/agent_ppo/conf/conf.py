#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Configuration for Gorge Chase PPO.
峡谷追猎 PPO 配置。
"""


class Config:

    # Feature dimensions / 特征维度（共521维: 6 + 16 + 30 + 12 + 16 + 1764）
    FEATURES = [
        6,    # hero: 6
        16,   # monsters: 2 * 8 = 16
        30,   # treasures: 5 * 6 = 30
        12,   # buffs: 2 * 6 = 12
        16,   # legal_action: 16
        1764, # map_info: 4x21x21 = 1764
    ]
    FEATURE_SPLIT_SHAPE = FEATURES
    FEATURE_LEN = sum(FEATURE_SPLIT_SHAPE)
    DIM_OF_OBSERVATION = FEATURE_LEN

    # Action space / 动作空间：16个方向（8移动+8闪现）
    ACTION_NUM = 16

    # Value head / 价值头：单头生存奖励
    VALUE_NUM = 1

    # PPO hyperparameters / PPO 超参数
    GAMMA = 0.99
    LAMDA = 0.95
    INIT_LEARNING_RATE_START = 0.0001
    BETA_START = 0.03
    CLIP_PARAM = 0.2
    VF_COEF = 1.0
    GRAD_CLIP_RANGE = 0.5
