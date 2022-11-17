# *****************************************************************
# This source code is only provided for the reviewing purpose of
# CVPR 2023. The source files should not be kept or used in any
# commercial or research products. Please delete all files after
# the reviewing period.
# *****************************************************************

from src.networks.decoders import Decoders

def get_model(cfg):
    c_dim = cfg['model']['c_dim']  # feature dimensions
    truncation = cfg['model']['truncation']
    decoder = Decoders(c_dim=c_dim, truncation=truncation)

    return decoder
