#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Feature preprocessor and reward design for Gorge Chase PPO.
峡谷追猎 PPO 特征预处理与奖励设计。
"""

import numpy as np

# Map size / 地图尺寸（128×128）
MAP_SIZE = 128.0
# Max monster speed / 最大怪物速度
MAX_MONSTER_SPEED = 5.0
# Max flash cooldown / 最大闪现冷却步数
MAX_FLASH_CD = 2000.0
# Max buff duration / buff最大持续时间
MAX_BUFF_DURATION = 50.0
# Maximum steps 
MAX_STEPS = 1000.0

def _norm(v, v_max, v_min=0.0):
    v = float(np.clip(v, v_min, v_max))
    return (v - v_min) / (v_max - v_min) if (v_max - v_min) > 1e-6 else 0.0

class Preprocessor:
    def __init__(self):
        self.reset()

    def reset(self):
        self.step_no = 0
        self.max_step = MAX_STEPS
        self.last_min_monster_dist_norm = 1.0
        self.last_treasure_count = 0
        self.last_buff_count = 0
        self.last_min_treasure_dist_norm = 1.0
        self.last_min_treasure_dist= 20.0
        self.last_m_min_dist=30
        self.last_flash_cd_val = 0
        self.last_buff_time_val = 0

        self.last_h_x = None
        self.last_h_z = None
        self.last_dx = None
        self.last_dz = None
        self.pos_history = []
        self.visited_counts = {}  # 新增：记录走过的网格（用于逃离死胡同）

    def feature_process(self, env_obs, last_action):
        observation = env_obs["observation"]
        frame_state = observation["frame_state"]
        env_info = observation["env_info"]
        map_info = observation["map_info"]
        legal_act_raw = observation.get("legal_action", observation.get("legal_act", []))

        # 【安全保险】如果从环境中读到当前步数为 0，说明是新的一局游戏开始，强制清空账本和相对状态！
        if observation.get("step_no", 0) == 0:
            self.visited_counts = {}
            self.pos_history = []
            self.last_h_x = None
            self.last_h_z = None
            self.last_dx = None
            self.last_dz = None
            self.last_min_treasure_dist = 999.0
            self.last_m_min_dist = 999.0
            self.last_min_treasure_dist_norm = 1.0
            self.last_min_monster_dist_norm = 1.0
            self.last_treasure_count = 0
            self.last_buff_count = 0
            self.last_flash_cd_val = 0

        self.step_no = observation["step_no"]
        self.max_step = env_info.get("max_step", MAX_STEPS)

        # ----------------------------------------------------
        # 1. Hero Features (6D)
        # ----------------------------------------------------
        hero = frame_state["heroes"]
        hero_pos = hero["pos"]
        h_x = hero_pos["x"]
        h_z = hero_pos["z"]
        
        tc_count = hero.get("treasure_collected_count", 0)
        # Handle dict or simple value for flash_cooldown
        flash_cd_val = hero.get("flash_cooldown", 0)
        flash_cd_val = flash_cd_val if isinstance(flash_cd_val, (int, float)) else 0
        
        buff_time_val = hero.get("buff_remaining_time", 0)
        buff_time_val = buff_time_val if isinstance(buff_time_val, (int, float)) else 0

        hero_feat = [
            _norm(h_x, MAP_SIZE),
            _norm(h_z, MAP_SIZE),
            _norm(flash_cd_val, MAX_FLASH_CD),
            _norm(buff_time_val, MAX_BUFF_DURATION),
            _norm(self.step_no, self.max_step),
            _norm(tc_count, 10.0)
        ]

        # ----------------------------------------------------
        # 2. Monster Features (16D)
        # ----------------------------------------------------
        monsters = frame_state.get("monsters", [])
        monsters_processed = []
        cur_min_dist_norm = 1.0
        m_min_dist = 20.0
        
        for m in monsters:
            is_in_view = float(m.get("is_in_view", 0))
            m_pos = m.get("pos", {"x":-1, "z":-1})
            
            if is_in_view and m_pos.get("x", -1) != -1:
                m_x = m_pos["x"]
                m_z = m_pos["z"]
                dx = m_x - h_x
                dz = m_z - h_z
                m_speed = m.get("speed", 1)
                m_dir = m.get("hero_relative_direction", -1)
                if m_dir == -1:
                    angle = np.arctan2(dz, dx)
                    m_dir = (int(np.round(-angle * 4 / np.pi)) % 8) + 1
                dist = np.sqrt(dx**2 + dz**2)
                d_norm = _norm(dist, MAP_SIZE * 1.41)
                feat = [1.0, _norm(m_x, MAP_SIZE), _norm(m_z, MAP_SIZE), 
                        _norm(dx, MAP_SIZE, -MAP_SIZE), _norm(dz, MAP_SIZE, -MAP_SIZE), 
                        d_norm, _norm(m_speed, MAX_MONSTER_SPEED), _norm(m_dir, 8.0)]
            else: # Out of view, use relative
                m_dir = m.get("hero_relative_direction", -1)
                dist = m.get("hero_l2_distance", 5) * 30.0 + 15.0
                
                if m_dir > 0 and m_dir <= 8:
                    dir_map = {
                        1: (1, 0),        # 东 (x正方向)
                        2: (1, -1),       # 东北
                        3: (0, -1),       # 北
                        4: (-1, -1),      # 西北
                        5: (-1, 0),       # 西
                        6: (-1, 1),       # 西南
                        7: (0, 1),        # 南
                        8: (1, 1)         # 东南
                    }
                    dir_vector = dir_map[m_dir]
                    norm_val = np.sqrt(dir_vector[0]**2 + dir_vector[1]**2)
                    dx = dist * (dir_vector[0] / norm_val)
                    dz = dist * (dir_vector[1] / norm_val)
                    m_x = h_x + dx
                    m_z = h_z + dz
                else:
                    m_x, m_z, dx, dz = 0.0, 0.0, 0.0, 0.0
                    
                d_norm = _norm(dist, MAP_SIZE * 1.41)
                feat = [0.0, _norm(m_x, MAP_SIZE), _norm(m_z, MAP_SIZE), 
                        _norm(dx, MAP_SIZE, -MAP_SIZE), _norm(dz, MAP_SIZE, -MAP_SIZE), 
                        d_norm, 0.0, _norm(m_dir, 8.0)]
                        
            monsters_processed.append((dist, d_norm, feat))
            
        # 按距离从近到远排序，确保神经网络的输入位置不会因为引擎的列表顺序打乱而产生“特征跳跃”
        monsters_processed.sort(key=lambda x: x[0])
        
        monster_feats = []
        if monsters_processed:
            # 【重要修复】确保不论怪物在不在视野内，都能更新环境全局的最短怪物距离
            m_min_dist = monsters_processed[0][0]
            cur_min_dist_norm = monsters_processed[0][1]
            
        for i in range(2):
            if i < len(monsters_processed):
                monster_feats.extend(monsters_processed[i][2])
            else:
                monster_feats.extend([0.0]*8)

        # ----------------------------------------------------
        # 3 & 4. Organ Features (Treasures 30D, Buffs 12D)
        # ----------------------------------------------------
        organs = frame_state.get("organs", [])
        
        # Process dynamically
        treasures = []
        buffs = []
        cur_min_treasure_dist = 20.0  # 同样改为较大值，防止视野外的宝箱距离被截断在8.0导致误判
        for org in organs:
            if org.get("status", 1) != 1:
                continue
            t_type = org.get("sub_type", 0)
            is_in_view = float(org.get("is_in_view", 1))
            o_pos = org.get("pos", {"x": -1, "z": -1})
            
            dx, dz = 0.0, 0.0
            if o_pos.get("x", -1) != -1:
                o_x = o_pos["x"]
                o_z = o_pos["z"]
                dx = o_x - h_x
                dz = o_z - h_z
                dist = np.sqrt(dx**2 + dz**2)
                if t_type == 1:
                    cur_min_treasure_dist = min(cur_min_treasure_dist, dist)
            else:
                dist = org.get("hero_l2_distance", 5) * 30.0 + 15 # Approximate
                o_dir = org.get("hero_relative_direction", -1)
                
                # 【核心修复】严格按照官方文档的 1-8 枚举转换宝箱相对方位
                if o_dir > 0 and o_dir <= 8:
                    dir_map = {
                        1: (1, 0),        # 东 (x正方向)
                        2: (1, -1),       # 东北 (x正, z负)
                        3: (0, -1),       # 北 (z负方向)
                        4: (-1, -1),      # 西北 (x负, z负)
                        5: (-1, 0),       # 西 (x负方向)
                        6: (-1, 1),       # 西南 (x负, z正)
                        7: (0, 1),        # 南 (z正方向)
                        8: (1, 1)         # 东南 (x正, z正)
                    }
                    dir_vector = dir_map[o_dir]
                    norm_val = np.sqrt(dir_vector[0]**2 + dir_vector[1]**2)
                    dx = dist * (dir_vector[0] / norm_val)
                    dz = dist * (dir_vector[1] / norm_val)
                    
                    o_x = h_x + dx
                    o_z = h_z + dz
                    if t_type == 1:
                        cur_min_treasure_dist = min(cur_min_treasure_dist, dist)
                else:
                    o_x, o_z, dx, dz = 0.0, 0.0, 0.0, 0.0
                
            dist_norm = _norm(dist, MAP_SIZE * 1.41)
            feat = [1.0, _norm(o_x, MAP_SIZE), _norm(o_z, MAP_SIZE), 
                    _norm(dx, MAP_SIZE, -MAP_SIZE), _norm(dz, MAP_SIZE, -MAP_SIZE), dist_norm]
            
            # 将 dist_norm 放在元组第一位，是为了后续方便使用 sort(key=lambda x: x[0]) 依据距离对宝箱和 Buff 进行排序
            if t_type == 1:
                treasures.append((dist_norm, feat))
            elif t_type == 2:
                buffs.append((dist_norm, feat))

        # 依据距离从小到大排序 (最近的优先)
        treasures.sort(key=lambda x: x[0])
        buffs.sort(key=lambda x: x[0])

        treasure_feats = []
        cur_min_treasure_dist_norm = 1.0
        if treasures:
            cur_min_treasure_dist_norm = treasures[0][0]
            
        for i in range(5):
            if i < len(treasures):
                treasure_feats.extend(treasures[i][1])
            else:
                treasure_feats.extend([0.0]*6)
                
        buff_feats = []
        for i in range(2):
            if i < len(buffs):
                buff_feats.extend(buffs[i][1])
            else:
                buff_feats.extend([0.0]*6)

        # ----------------------------------------------------
        # 5. Legal Action Mask (16D)
        # ----------------------------------------------------
        legal_action = [1] * 16
        if isinstance(legal_act_raw, list) and legal_act_raw:
            if isinstance(legal_act_raw[0], bool):
                for j in range(min(16, len(legal_act_raw))):
                    legal_action[j] = int(legal_act_raw[j])
            else:
                valid_set = {int(a) for a in legal_act_raw if int(a) < 16}
                legal_action = [1 if j in valid_set else 0 for j in range(16)]

        if sum(legal_action) == 0:
            legal_action[0] = 1 # fallback
            
        legal_action_feat = list(map(float, legal_action))

        # ----------------------------------------------------
        # 6. Map Features (1764D) - 4x21x21 flattened
        # ----------------------------------------------------
        map_feat = np.zeros((4, 21, 21), dtype=np.float32)

        # 通道 0：地形图
        if map_info and len(map_info) == 21 and len(map_info[0]) == 21:
            for row in range(21):
                for col in range(21):
                    map_feat[0, row, col] = float(map_info[row][col] != 0)
        else:
            # Fallback mapping if size mismatch
            if map_info:
                center_r = len(map_info)//2
                center_c = len(map_info[0])//2
                for row_idx, row in enumerate(range(center_r - 10, center_r + 11)):
                    for col_idx, col in enumerate(range(center_c - 10, center_c + 11)):
                        if 0 <= row < len(map_info) and 0 <= col < len(map_info[0]):
                            map_feat[0, row_idx, col_idx] = float(map_info[row][col] != 0)

        # 通道 1：敌情图
        for m in monsters:
            m_pos = m.get("pos", {"x": -1, "z": -1})
            if m.get("is_in_view", 0) and m_pos["x"] != -1:
                dx = m_pos["x"] - h_x
                dz = m_pos["z"] - h_z
            else:
                dist = m.get("hero_l2_distance", 5) * 30.0 + 15.0 # 使用真实世界距离，后续在画图时会自动 clip 到地图边缘
                m_dir = m.get("hero_relative_direction", -1)
                
                # 【与宝箱同步修复】严格按照官方文档的 1-8 枚举转换怪物相对方位
                if m_dir > 0 and m_dir <= 8:
                    dir_map = {
                        1: (1, 0),        # 东
                        2: (1, -1),       # 东北
                        3: (0, -1),       # 北
                        4: (-1, -1),      # 西北
                        5: (-1, 0),       # 西
                        6: (-1, 1),       # 西南
                        7: (0, 1),        # 南
                        8: (1, 1)         # 东南
                    }
                    dir_vector = dir_map[m_dir]
                    norm_val = np.sqrt(dir_vector[0]**2 + dir_vector[1]**2)
                    dx = dist * (dir_vector[0] / norm_val)
                    dz = dist * (dir_vector[1] / norm_val)
                else: 
                    continue
                    
            col = int(np.clip(10 + round(dx), 0, 20))
            row = int(np.clip(10 + round(dz), 0, 20))
            map_feat[1, row, col] = 1.0

        # 通道 2：资源图
        for org in organs:
            if org.get("status", 1) != 1:
                continue
            t_type = org.get("sub_type", 0)
            val = 1.0 if t_type == 1 else 0.5
            o_pos = org.get("pos", {"x": -1, "z": -1})
            
            dx, dz = 0.0, 0.0
            if org.get("is_in_view", 1) and o_pos["x"] != -1:
                dx = o_pos["x"] - h_x
                dz = o_pos["z"] - h_z
            else:
                dist = org.get("hero_l2_distance", 5) * 30.0 + 15.0 # 使用真实世界距离，后续在画图时会自动 clip 退退雷达边缘画出来
                o_dir = org.get("hero_relative_direction", -1)
                
                # 【核心修复】根据官方文档严格匹配相对方向 (1-8 映射到对应的角度或直角坐标)
                # 栅格地图左上(0,0)，右正(x东)，下正(z南)，即上侧为北，下侧为南
                # 1=东(x+, z=0), 2=东北(x+, z-), 3=北(x=0, z-), 4=西北(x-, z-), 
                # 5=西(x-, z=0), 6=西南(x-, z+), 7=南(x=0, z+), 8=东南(x+, z+)
                if o_dir > 0 and o_dir <= 8:
                    dir_map = {
                        1: (1, 0),        # 东
                        2: (1, -1),       # 东北
                        3: (0, -1),       # 北
                        4: (-1, -1),      # 西北
                        5: (-1, 0),       # 西
                        6: (-1, 1),       # 西南
                        7: (0, 1),        # 南
                        8: (1, 1)         # 东南
                    }
                    dir_vector = dir_map[o_dir]
                    # 将离散的方向向量归一化后，乘以距离得到真实的 (dx, dz)
                    norm_val = np.sqrt(dir_vector[0]**2 + dir_vector[1]**2)
                    dx = dist * (dir_vector[0] / norm_val)
                    dz = dist * (dir_vector[1] / norm_val)
                else:
                    continue
                    
            col = int(np.clip(10 + round(dx), 0, 20))
            row = int(np.clip(10 + round(dz), 0, 20))
            map_feat[2, row, col] = val

        # 通道 3：探索记忆图 (Memory Map) - 让智能体长出“眼睛”，能够看到哪些区域去过，哪些没去过
        for row in range(21):
            for col in range(21):
                map_dx = col - 10
                map_dz = row - 10
                abs_x, abs_z = h_x + map_dx, h_z + map_dz
                g_x, g_z = int(abs_x / 2.0), int(abs_z / 2.0)
                v_count = self.visited_counts.get((g_x, g_z), 0)
                # 没去过的黑色区域是 0.0，去过很多次的旧路颜色越亮（接近 1.0）
                map_feat[3, row, col] = float(min(v_count / 10.0, 1.0))
        
        map_feat[3, 10, 10] = -1.0 # 强化标注玩家中心当前点

        map_feat_flat = map_feat.flatten()

        feature = np.concatenate([
            np.array(hero_feat, dtype=np.float32),          # 6
            np.array(monster_feats, dtype=np.float32),      # 16
            np.array(treasure_feats, dtype=np.float32),     # 30
            np.array(buff_feats, dtype=np.float32),         # 12
            np.array(legal_action_feat, dtype=np.float32),  # 16
            map_feat_flat                                   # 1764
        ])

        # ----------------------------------------------------
        # Reward Function
        # ----------------------------------------------------
        reward=0.0 # 提高单步存活奖励，让“活着”成为最大的正反馈
        if m_min_dist<8.0:
            reward += 0.015 * m_min_dist# 怪物距离变化奖励，鼓励远离怪物（但不直接扣分，避免自杀
        else:
            reward += 0.12 # Reduce baseline survival
        # 1. Highest Priority: Penalize getting stuck
        if self.last_h_x is not None and self.last_h_z is not None:
            dist_moved = np.sqrt((h_x - self.last_h_x)**2 + (h_z - self.last_h_z)**2)
            if dist_moved < 0.05:  # If barely moved across 1 frame (stuck on wall)
                reward -= 1.0# Heavy penalty for wall stuck
                
            # 2. Prevent circling / pacing back and forth (少走回头路)
            elif hasattr(self, 'pos_history') and len(self.pos_history) == 15:
                # 比较当前位置与 15 步（约 1.5 秒）前的位置
                old_x, old_z = self.pos_history[0]
                dist_15_steps = np.sqrt((h_x - old_x)**2 + (h_z - old_z)**2)
                if dist_15_steps < 4:
                    reward -= 0.20 # 如果15步前和现在位置非常接近，说明可能在原地绕圈或来回走，稍微扣分，但这也会被 S 型曲线绕开
            cos_sim_2 = 0.0
            # 3. 直线惯性奖励（防止无目标原地转圈/走折线）
            if getattr(self, 'last_dx', None) is not None and getattr(self, 'last_dz', None) is not None:
                # 只有在雷达没有响（没有发现<8的宝箱）时，才强制它保持直线探索
                if cur_min_treasure_dist > 8.0:
                    dx_curr = h_x - self.last_h_x
                    dz_curr = h_z - self.last_h_z
                    norm_c = np.sqrt(dx_curr**2 + dz_curr**2) + 1e-8
                    norm_l = np.sqrt(self.last_dx**2 + self.last_dz**2) + 1e-8
                    cos_sim_2 = (dx_curr * self.last_dx + dz_curr * self.last_dz) / (norm_c * norm_l)
                    
                    if cos_sim_2 > 0.4:
                        reward += 0.02  # 保持大体直线，微小奖励（促使其探索新区域）
                    elif cos_sim_2 < -0.2:
                        reward -= 0.15  # 掉头或者拐大弯，惩罚其做无用功
                
                # 视野外距离过远时，由于地图有死胡同，宝箱指向可能被墙体遮挡频繁变化。
                # 放弃“跟着最近宝箱走”，强制“沿着原方向一条路走到黑”，打破无脑横跳的死循环。
                if cur_min_treasure_dist > 8.0 and m_min_dist > 8.0:
                    if tc_count == self.last_treasure_count and self.last_h_x is not None:
                        # 鼓励沿着先前的惯性方向跑长线，不轻易回头（和上方的模块叠加，总直行加分0.07，掉头扣分0.15）
                        if cos_sim_2 > 0.4:
                            reward += 0.05  # 坚决保持一条道走到底加分
                        elif cos_sim_2 < -0.2:
                            reward -= 0.15 # 无故回头或者乱拐大弯扣分
            
            # 产生实质性位移时，再更新“惯性朝向”，避免卡墙微小位移带偏方向
            if dist_moved >= 0.05:
                self.last_dx = h_x - self.last_h_x
                self.last_dz = h_z - self.last_h_z
              
        


            # 4. 基于全局访问记录的“新路径探索”与“旧路径惩罚”机制 (Count-based Exploration Bonus)
            # 把地图划分为 2x2 的网格（精度可调）
            grid_x, grid_z = int(h_x / 2.0), int(h_z / 2.0)
            visit_count = self.visited_counts.get((grid_x, grid_z), 0)
            
            if visit_count == 0:
                # 鼓励探索：第一次踏入一个全新的地块，给一笔小小的“开荒”奖励
                reward += 0.05
            else:
                # 惩罚重走：稍微扣除一点分，防止无限白嫖存活分，但切忌扣分过度大于存活基础分
                # 否则到了游戏后期无新路可走时，四处都是巨额负分，模型会绝望自杀（主动撞墙或撞怪）
                reward -= 0.02 * min(visit_count, 5) # Limit penalty to max -0.1

            self.visited_counts[(grid_x, grid_z)] = visit_count + 1

        # 注意：彻底删除了“由于怪物靠近而每步扣分”的逻辑。
        # 因为如果因为怪物靠近而一直扣分，模型会发现“如果怪兽比自己跑得快，活得越久扣分越多，不如马上撞死清零”。
        # 现在，只要活着就能+1.0，怪兽逼近不会扣分，网络为了贪生存分，自然会主动拼命活得越久越好。
            
        # Treasure collection sparse reward - HIGHER VALUE to encourage collection
        if tc_count > self.last_treasure_count:
            reward += 10.0 # 保持固定巨额，停止二次方暴增（10.0 * tc_count 会引起后半局价值爆炸） # 每多一个宝箱，奖励翻倍，鼓励多收集宝箱（同时也让模型更愿意冒险去追逐宝箱，因为单个宝箱的奖励已经非常丰厚了）
            
        # Buff collection sparse reward
        cur_buff_count = env_info.get("collected_buff", 0)
        if cur_buff_count > self.last_buff_count:
            reward += 15.0
            
        # 检测是否交了闪现（冷却时间猛增说明使用了闪现）
        used_flash = (flash_cd_val > self.last_flash_cd_val + 10)
        
        # 当智能体使用闪现技能成功穿过障碍物时给巨额奖励
        if used_flash and self.last_h_x is not None and self.last_h_z is not None:
            if map_info and len(map_info) > 0 and len(map_info[0]) > 0:
                center_r = len(map_info) // 2
                center_c = len(map_info[0]) // 2
                dx = h_x - self.last_h_x
                dz = h_z - self.last_h_z
                
                # 沿虚线轨迹采样点，检测是否有障碍物 (在map_info中 0=障碍物, 1=可通行)
                passed_obstacle = False
                for t in [0.125, 0.25,0.5, 0.625,0.75, 0.875]:
                    # current position (h_x, h_z) is at center_c, center_r
                    # previous position relative to current is -dx, -dz
                    # z 轴向下对应由于是以英雄为中心的ego_map，通常 row=z, col=x
                    p_row = center_r - dz * t
                    p_col = center_c - dx * t
                    r_idx = int(round(p_row))
                    c_idx = int(round(p_col))
                    
                    if 0 <= r_idx < len(map_info) and 0 <= c_idx < len(map_info[0]):
                        if map_info[r_idx][c_idx] == 0:  # 0 为障碍物
                            passed_obstacle = True
                            break
                            
                if passed_obstacle and self.last_m_min_dist > 8.0: # 只有在没有怪物威胁的情况下穿墙才奖励，避免智能体为了刷穿墙奖励而故意撞怪自杀
                    reward += 1.0  # 穿墙成功，给予适当奖励（调低，防止故意刷分）
                else:
                    # 没穿墙的情况：触发“背水一战”（穿怪闪现）判定
                    flash_dist = np.sqrt(dx**2 + dz**2)
                    if self.last_m_min_dist < 4.0 and flash_dist > 2.0:
                        # 检查交闪现瞬间，智能体是不是被困在死胡同/原地区域
                        last_grid_x, last_grid_z = int(self.last_h_x / 2.0), int(self.last_h_z / 2.0)
                        reward += 1.0 # 首先给一个基础奖励，鼓励它尝试用闪现逃离（即使不成功也能得到这个奖励）
                        if m_min_dist > self.last_m_min_dist:
                            pass # 在平地空旷处依然不允许“碰瓷刷分”，不予奖励
                        else:
                                reward -= 2.0 # 没拉开距离反而变近，扣分
                    else:
                        reward -= 2.0  # 没怪的时候瞎交闪现，或者原地撞墙交闪现，扣分

        # 取消怪物靠近扣分，否则0.1存活分-0.15靠近分 = 负收益（导致自杀）
        # if m_min_dist<7 and m_min_dist<self.last_m_min_dist:
        #     reward -= 0.15
         # Encourage using speed buff
        
        # 恢复距离8.0限制：防止欧氏距离导航导致智能体为了靠近视野外的宝箱而隔墙撞死（局部最优）
        if cur_min_treasure_dist < 8.0:
            if tc_count == self.last_treasure_count and getattr(self, 'step_no', 0) > 0:  # 必须大于0，避免第一帧吃到2000分的初始化差值
                delta_t = self.last_min_treasure_dist - cur_min_treasure_dist
                # 限制单步接近宝箱的得分上限（避免传送等异常位移拿到暴发分数）
                delta_t = np.clip(delta_t, -2.0, 2.0)
                reward += 0.02 * delta_t # 极大削弱欧氏距离导航，防止智能体被非凸障碍物卡死


        self.last_min_monster_dist_norm = cur_min_dist_norm
        self.last_min_treasure_dist = cur_min_treasure_dist
        self.last_min_treasure_dist_norm = cur_min_treasure_dist_norm
        self.last_treasure_count = tc_count
        self.last_buff_count = cur_buff_count
        self.last_flash_cd_val = flash_cd_val
        self.last_h_x = h_x
        self.last_h_z = h_z
        self.last_m_min_dist=m_min_dist

        if not hasattr(self, 'pos_history'):
            self.pos_history = []
        self.pos_history.append((h_x, h_z))
        # Keep track of last 15 steps of history for anti-backtracking
        if len(self.pos_history) > 15:
            self.pos_history.pop(0)

        # ----------------------------------------------------
        # 死胡同检测机制 (Dead-end detection)
        # ----------------------------------------------------
        if map_info and len(map_info) > 0 and len(map_info[0]) > 0:
            center_r = len(map_info) // 2
            center_c = len(map_info[0]) // 2
            radius = 8 # 检测 17x17 范围的边界
            r_min = max(0, center_r - radius)
            r_max = min(len(map_info) - 1, center_r + radius)
            c_min = max(0, center_c - radius)
            c_max = min(len(map_info[0]) - 1, center_c + radius)
            
            perimeter = []
            for c in range(c_min, c_max + 1): perimeter.append(map_info[r_min][c])
            for r in range(r_min + 1, r_max): perimeter.append(map_info[r][c_max])
            for c in range(c_max, c_min - 1, -1): perimeter.append(map_info[r_max][c])
            for r in range(r_max - 1, r_min, -1): perimeter.append(map_info[r][c_min])
            
            if perimeter:
                exits = 0
                ones_count = 0
                for i in range(len(perimeter)):
                    prev = perimeter[i - 1]
                    curr = perimeter[i]
                    if curr != 0:
                        ones_count += 1
                    if prev == 0 and curr != 0:
                        exits += 1
                
                # 如果出口数量 <= 1，并且出口的宽度（可通行的栅格数）小于 10（说明是细长的开口或U型墙角），才是真正的死胡同
                # 避免把笔直的贴边大平墙（ones_count很大）误判为死胡同
                if exits <= 1 and ones_count < 10:
                    reward -= 0.3

        return feature, legal_action, [reward]


