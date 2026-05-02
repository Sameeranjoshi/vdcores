import torch, time                                                                                          
t = torch.tensor([1.])                                                                                    
t0 = time.time()                                         
t.to('cuda')                                                                                                
torch.cuda.synchronize()                                         
print(f'{time.time()-t0:.1f}s')
