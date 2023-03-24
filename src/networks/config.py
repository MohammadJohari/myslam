# ESLAM is a A NeRF-based SLAM system.
# It utilizes Neural Radiance Fields (NeRF) to perform Simultaneous
# Localization and Mapping (SLAM) in real-time. This system uses neural
# rendering techniques to create a 3D map of an environment from a
# sequence of images and estimates the camera pose simultaneously.
#
# Apache License 2.0
#
# Copyright (c) 2023 ams-OSRAM AG
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from src.networks.decoders import Decoders

def get_model(cfg):
    c_dim = cfg['model']['c_dim']  # feature dimensions
    truncation = cfg['model']['truncation']
    learnable_beta = cfg['rendering']['learnable_beta']

    decoder = Decoders(c_dim=c_dim, truncation=truncation, learnable_beta=learnable_beta)

    return decoder
