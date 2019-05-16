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
"""Simulated annealing controller.
"""
import math
import numpy as np

from .controller import Controller


class SaController(Controller):
    """Simulated annealing controller.
    """

    def __init__(self, range_table, reduce_rate=0.85, init_temperature=1024):
        """Initialize.

        Args:
            range_table: list, variable range table.
            reduce_rate: float, reduce rate.
            init_temperature: float, init temperature.
        """
        super(SaController, self).__init__()
        self._range_table = range_table
        self._reduce_rate = reduce_rate
        self._init_temperature = init_temperature

    def check(self, reward_new, reward, iteration):
        """Check if the var should be updated using general policy.

        Args:
            reward_new: float, new reward.
            reward: float, reward.
            iteration: int, iteration.

        Returns:
            bool, a list of new variables.
        """
        if reward_new > reward:
            return True
        temperature = self._init_temperature * self._reduce_rate**iteration
        return np.random.random() <= math.exp(
            (reward_new - reward) / temperature)

    def generate_init_var(self):
        """Generate init var.

        Returns:
            list, a list of variables.
        """
        return [np.random.randint(_) for _ in self._range_table]

    def generate_new_var(self, var):
        """Generate new var.

        Args:
            var: list, a list of variables.

        Returns:
            list, a list of new variables.
        """
        var_new = var[:]
        index = int(len(self._range_table) * np.random.random())
        var_new[index] = int(self._range_table[index] * np.random.random())
        return var_new
