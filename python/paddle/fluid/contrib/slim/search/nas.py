#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Neural Architecture Search.
"""
from __future__ import print_function

import time
import subprocess


class Nas(object):
    """Neural Architecture Search.
    """

    def __init__(self, controller, get_reward_command, verbose=False):
        self._controller = controller
        self._get_reward_command = get_reward_command
        self._verbose = verbose

    def search(self, max_iterations, init_var, init_reward):
        """Search.

        Args:
            max_iterations: max iterations.
            init_var: init var.
            init reward: init reward.

        Returns:
            tuple, a tuple of (var, reward)
                where reward is the maximum during searching.
        """
        var = init_var
        reward = init_reward
        var_max = var
        reward_max = reward
        if self._verbose:
            print('[INFO] {} iter 0 reward {} var {}'.format(
                time.ctime(), reward, var))
        for iteration in range(1, max_iterations + 1):
            var_new = self._controller.generate_new_var(var)
            reward_new = self.get_reward(str(var_new))
            if self._verbose:
                print('[INFO] {} iter {} reward_new {} var_new {}'.format(
                    time.ctime(), iteration, reward_new, var_new))
            if self._controller.check(reward_new, reward, iteration):
                reward = reward_new
                var = var_new[:]
            if reward_new > reward_max:
                reward_max = reward_new
                var_max = var
            if self._verbose:
                print('[INFO] {} iter {} reward {} var {}'.format(
                    time.ctime(), iteration, reward, var))
        return (var_max, reward_max)

    def get_reward(self, var):
        """Get reward.

        Args:
            var: a string that represents variable list.

        Returns:
            float, reward.
        """
        p_child = subprocess.Popen(
            self._get_reward_command(var),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE)
        out, err = p_child.communicate()
        out = out.strip().split('\n')[-1]
        try:
            reward = float(out.split()[0])
        except ValueError:
            print('[WARN] {} out {} err {}'.format(time.ctime(), out, err))
            reward = 0
        return reward
