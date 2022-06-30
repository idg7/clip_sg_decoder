from models.realNVP import RealNVP

import torch
import consts


def predict(mapper: RealNVP, x: torch.Tensor) -> torch.Tensor:
    u = mapper.base_dist.sample([int(x.shape[0])]).cuda(non_blocking=True)
    if not consts.PREDICT_WITH_RANDOM_Z:
        u = torch.zeros_like(u).cuda(non_blocking=True)

    if consts.TRUNCATE_LIMIT is not None:
        u = torch.clamp(u, min=-consts.TRUNCATE_LIMIT, max=consts.TRUNCATE_LIMIT)

    predicted_latent, _ = mapper.inverse(u, x)
    return predicted_latent
