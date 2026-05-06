import sys; sys.argv = ['x', '-l']
  import torch
  from dae.launcher import *
  from dae.util import dae_app
  gpu = torch.device('cuda')
  dae = Launcher(1, device=gpu)
  dae.i(TerminateC(), TerminateM())
  print("smem=", dae.smem_size, flush=True)
  dae_app(dae)
  print("kernel returned", flush=True)
