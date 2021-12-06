from typing import Dict, Tuple

import torch.nn
import consts

from loss.semantic_loss import SemanticLoss
from loss.lpips import LPIPSWrapper
from loss.id_loss import IDLoss


def setup_generator_loss(opts) -> Tuple[Dict[str, torch.nn.Module], Dict[str, float]]:
    id_loss = IDLoss()
    lpips = LPIPSWrapper()
    semantic_loss = SemanticLoss(opts.semantic_architecture)

    lpips = lpips.cuda()
    id_loss = id_loss.cuda()
    criteria = {
        'loss_l2': torch.nn.MSELoss(),
        'semantic': semantic_loss,
        'ID': id_loss,
        'LPIPS': lpips
    }
    lambdas = {
        'loss_l2': opts.l2_lambda,
        'semantic': opts.semantic_lambda,
        'ID': opts.id_lambda,
        'LPIPS': opts.lpips_lambda
    }

    return criteria, lambdas
