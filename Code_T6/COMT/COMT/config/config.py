#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   config.py
@Time    :   2025/01/20 01:32:26
@Author  :   biabuluo 
@Version :   1.0
@Desc    :   None
'''

import toml
import os

class Config:
    def __init__(self, config_path="config/config.toml"):
        self.config_path = config_path
        self.config = None
        self._load_config()

    def _load_config(self):
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Config file not found at {self.config_path}")
        self.config = toml.load(self.config_path)

    def get(self, section, key, default=None):
        """Retrieve a specific key from a section."""
        return self.config.get(section, {}).get(key, default)

    def get_section(self, section):
        """Retrieve a full section as a dictionary."""
        return self.config.get(section, {})

# Example usage
if __name__ == "__main__":
    config = Config('./config.toml')
    print(config.get_section("training"))
    print("Learning rate:", config.get("training", "learning_rate"))
