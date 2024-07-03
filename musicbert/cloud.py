# -*- coding: UTF-8 -*-
"""
-----------------------------------
@Author : Your Name
@Date : 2024/6/25
-----------------------------------
"""
import os
import torch.distributed.rpc as rpc

from model import DistMusicBertDecomposition, DistMusicBertOri
from config import args

# Initialize the RPC framework for the cloud worker
rpc.init_rpc("cloud", rank=1, world_size=2)
rpc.shutdown()
