"""Numerical checks using the normal CLI's epoch-zero smoke checkpoint and its refitted constants."""
import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import torch

from adaptive_scores import weights
from adaptive_runs import ROOT
import liu_model
from torch_model import load_model, build_model


def verify(path):
    torch.manual_seed(7)
    model, cfg, ck = load_model(path)
    residual = build_model(dict(cfg, net_type='residual'))
    residual.set_physics(model.liu_consts, ck['data_min'], ck['data_max'])
    xt = torch.rand(8, 20, 5)
    xs = torch.tensor([[1.17, .25]]).repeat(8, 1)
    raw = xt[:, -1].double() * (model.x_max-model.x_min) + model.x_min
    frame = pd.DataFrame(dict(vertspd=raw[:,2], airspeed=raw[:,4], aoa=raw[:,3],
                              payload=xs[:,1], density=xs[:,0])).T
    reference = liu_model.power(frame, *model.liu_consts.double().tolist())
    out, _ = model(xt, xs)
    assert out.shape == (8,1)
    watts, mult, terms = model.term_prediction(xt, xs)
    torch.testing.assert_close(mult, torch.ones_like(mult), atol=0, rtol=0)
    np.testing.assert_allclose(watts.detach().numpy().ravel(), reference, rtol=1e-5)
    torch.testing.assert_close(residual(xt,xs)[0],out)
    optim = torch.optim.Adam(model.parameters())
    (out - (out.detach() + .1)).square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert model.power_reg.weight.grad.abs().sum() > 0
    optim.step()
    with TemporaryDirectory() as tmp:
        torch.save(dict(ck,state_dict=model.state_dict()), Path(tmp)/'model.pt')
        restored,_,_ = load_model(Path(tmp)/'model.pt')
        for a,b in zip(model.term_prediction(xt,xs), restored.term_prediction(xt,xs)):
            torch.testing.assert_close(a,b,atol=0,rtol=0)
        torch.testing.assert_close(model(xt,xs)[0], restored(xt,xs)[0],atol=0,rtol=0)
    with torch.no_grad():
        model.power_reg.weight.zero_()
        model.power_reg.bias.copy_(torch.tensor([-100.,0.,100.]))
    watts,mult,terms = model.term_prediction(xt,xs)
    torch.testing.assert_close(mult,torch.tensor([.5,1.,2.]).expand_as(mult),atol=0,rtol=0)
    torch.testing.assert_close(((mult-1)*terms).sum(1), watts[:,0]-terms.sum(1), atol=1e-4,rtol=1e-5)
    t = np.array([0., 1., 8., 51., 60.])
    contribution = weights(t)*-20/3600
    np.testing.assert_allclose(contribution.sum(), -1/3)
    np.testing.assert_allclose(sum(contribution[idx].sum() for idx in [[0,1],[2],[3,4]]), -1/3)
    plain,_,_ = load_model(ROOT/'models_torch/s-TCN.pt')
    assert torch.isfinite(plain(xt,xs)[0]).all()
    print('Passed Liu identity, bounds, gradients, restore, legacy plain and signed Wh checks.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    verify(parser.parse_args().checkpoint)
