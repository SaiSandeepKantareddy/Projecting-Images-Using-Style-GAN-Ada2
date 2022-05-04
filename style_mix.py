import dnnlib
import torch
import legacy
import numpy as np
import os
import cv2
network_pkl = '/home/jupyter-skanta2/Sandeep_Pytorch/results/00028--auto8/network-snapshot-022176.pkl' # PATH TO TRAINED NETWORK
# Pick sources and targets among YOUR DIRECTORIES OF PROJECTOR RESULTS
sources = ['/home/jupyter-skanta2/Sandeep_Pytorch/results_images_source/']
targets = ['/home/jupyter-skanta2/Sandeep_Pytorch/results_images/']
selects = list(set(sources+targets)) 

# tflib.init_tf()
# with dnnlib.util.open_url(network_pkl) as fp:
#     _G, _D, Gs = pickle.load(fp)
# Gs_syn_kwargs = {
#         'output_transform': dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True),
# }
device = torch.device('cuda')
with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore

simgs = {}
dlatents = {}
for x in selects:
    dlatents[x] = np.load(os.path.join(x,'projected_w.npz'))['w']
    timg = os.path.join(x,'proj.png')
    timg = cv2.imread(timg)
    simgs[x] = timg

mixed = {}
for s in sources:
    for t in targets:
        mix = np.copy(dlatents[s])
        mix[0][:6] = dlatents[t][0][:6] ## YOU CAN CHANGE WHICH LAYERS TO MIX
        miximg = G.synthesis(mix)
        mixed[(s,t)] = miximg[0]

# DISPLAY RESULT
r = len(targets)+1
c = len(sources)+1

plt.style.use('default')
fig, axs = plt.subplots(r,c, figsize=(5*c,5*r))

for i,s in enumerate(sources):
    fig0 = axs[0,1+i].imshow(simgs[s])
for j,t in enumerate(targets):
    fig0 = axs[1+j,0].imshow(simgs[t])
    
for i,s in enumerate(sources):
    for j,t in enumerate(targets):
        fig0 = axs[1+j,1+i].imshow(mixed[(s,t)])

plt.show()
plt.close(fig)