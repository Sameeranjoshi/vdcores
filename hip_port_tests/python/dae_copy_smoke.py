import sys; sys.argv = ['x', '-b', '10']                                                                          
import torch
from dae.launcher import *                                                                                  
from dae.util import dae_app
                                                                                                            
gpu = torch.device('cuda')
num_sms, num_loads, load_bytes = 1, 1, 1024                                                                 
                                                                                                            
vec = torch.arange(num_sms*num_loads*(load_bytes//4),                                                       
                   dtype=torch.float32, device=gpu).reshape(num_sms, num_loads, load_bytes//4)              
out = torch.zeros_like(vec)                                                                                 
                     
dae = Launcher(num_sms, device=gpu)                                                                         
def f(sm):           
    return RepeatM.on(num_loads,                                                                            
        [TmaLoad1D(vec[sm,0]),  load_bytes],                                                                
        [TmaStore1D(out[sm,0]), load_bytes])                                                                
dae.i(Copy(num_loads, load_bytes), f, TerminateM(), TerminateC())                                           
                                                                                                            
total_bytes = num_sms * num_loads * load_bytes * 2  # read + write
dae_app(dae, total_bytes=total_bytes)                                                                                                
torch.cuda.synchronize()                                                                                    
print("equal:", torch.equal(vec, out), flush=True)                            
print("vec[:8]:", vec.flatten()[:8].tolist())                                                               
print("out[:8]:", out.flatten()[:8].tolist()) 
