import copy
import os
from time import perf_counter

import click
import imageio
import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F

import dnnlib
import legacy

import gc
import cv2
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,5,6" 
gc.collect()

import pandas as pd
# df1=pd.read_csv('/home/jupyter-skanta2/Collaboratory/race_detection_data/Emory_CXR_test_censored.csv')
df1=pd.read_pickle('/home/jupyter-skanta2/Collaboratory/GAN_debias/mammo_split.pkl')

imgs=[]
labels=[]
# for i,j in zip(df1['hiti_path'],df1['Race']):
#     try:
#         imgs.append(PIL.Image.open('/home/jupyter-skanta2'+i).convert('RGB'))
#         labels.append(j)
#     except:
#         continue

for i in df1['test']['path'][0:].to_list():
    print(i)
    try:
        #imgs.append(PIL.Image.open('/home/jupyter-skanta2'+i).convert('RGB'))
        imgs.append(i)
    except:
        continue        

def project(
    G,
    target: torch.Tensor, # [C,H,W] and dynamic range [0,255], W & H must match G output resolution
    *,
    num_steps                  = 1000,
    w_avg_samples              = 10000,
    initial_learning_rate      = 0.1,
    initial_noise_factor       = 0.05,
    lr_rampdown_length         = 0.25,
    lr_rampup_length           = 0.05,
    noise_ramp_length          = 0.75,
    regularize_noise_weight    = 1e5,
    verbose                    = False,
    device: torch.device
):
    assert target.shape == (G.img_channels, G.img_resolution, G.img_resolution)

    def logprint(*args):
        if verbose:
            print(*args)

    G = copy.deepcopy(G).eval().requires_grad_(False).to(device) # type: ignore

    # Compute w stats.
    logprint(f'Computing W midpoint and stddev using {w_avg_samples} samples...')
    z_samples = np.random.RandomState(123).randn(w_avg_samples, G.z_dim)
    w_samples = G.mapping(torch.from_numpy(z_samples).to(device), None)  # [N, L, C]
    w_samples = w_samples[:, :1, :].cpu().numpy().astype(np.float32)       # [N, 1, C]
    w_avg = np.mean(w_samples, axis=0, keepdims=True)      # [1, 1, C]
    w_std = (np.sum((w_samples - w_avg) ** 2) / w_avg_samples) ** 0.5

    # Setup noise inputs.
    noise_bufs = { name: buf for (name, buf) in G.synthesis.named_buffers() if 'noise_const' in name }

    # Load VGG16 feature detector.
    url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt'
    with dnnlib.util.open_url(url) as f:
        vgg16 = torch.jit.load(f).eval().to(device)

    # Features for target image.
    target_images = target.unsqueeze(0).to(device).to(torch.float32)
    if target_images.shape[2] > 256:
        target_images = F.interpolate(target_images, size=(256, 256), mode='area')
    target_features = vgg16(target_images, resize_images=False, return_lpips=True)

    w_opt = torch.tensor(w_avg, dtype=torch.float32, device=device, requires_grad=True) # pylint: disable=not-callable
    w_out = torch.zeros([num_steps] + list(w_opt.shape[1:]), dtype=torch.float32, device=device)
    optimizer = torch.optim.Adam([w_opt] + list(noise_bufs.values()), betas=(0.9, 0.999), lr=initial_learning_rate)

    # Init noise.
    for buf in noise_bufs.values():
        buf[:] = torch.randn_like(buf)
        buf.requires_grad = True

    for step in range(num_steps):
        # Learning rate schedule.
        t = step / num_steps
        w_noise_scale = w_std * initial_noise_factor * max(0.0, 1.0 - t / noise_ramp_length) ** 2
        lr_ramp = min(1.0, (1.0 - t) / lr_rampdown_length)
        lr_ramp = 0.5 - 0.5 * np.cos(lr_ramp * np.pi)
        lr_ramp = lr_ramp * min(1.0, t / lr_rampup_length)
        lr = initial_learning_rate * lr_ramp
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Synth images from opt_w.
        w_noise = torch.randn_like(w_opt) * w_noise_scale
        ws = (w_opt + w_noise).repeat([1, G.mapping.num_ws, 1])
        synth_images = G.synthesis(ws, noise_mode='const')

        # Downsample image to 256x256 if it's larger than that. VGG was built for 224x224 images.
        synth_images = (synth_images + 1) * (255/2)
        if synth_images.shape[2] > 256:
            synth_images = F.interpolate(synth_images, size=(256, 256), mode='area')

        # Features for synth images.
        synth_features = vgg16(synth_images, resize_images=False, return_lpips=True)
        dist = (target_features - synth_features).square().sum()

        # Noise regularization.
        reg_loss = 0.0
        for v in noise_bufs.values():
            noise = v[None,None,:,:] # must be [1,1,H,W] for F.avg_pool2d()
            while True:
                reg_loss += (noise*torch.roll(noise, shifts=1, dims=3)).mean()**2
                reg_loss += (noise*torch.roll(noise, shifts=1, dims=2)).mean()**2
                if noise.shape[2] <= 8:
                    break
                noise = F.avg_pool2d(noise, kernel_size=2)
        loss = dist + reg_loss * regularize_noise_weight

        # Step
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        logprint(f'step {step+1:>4d}/{num_steps}: dist {dist:<4.2f} loss {float(loss):<5.2f}')

        # Save projected W for each optimization step.
        w_out[step] = w_opt.detach()[0]

        # Normalize noise.
        with torch.no_grad():
            for buf in noise_bufs.values():
                buf -= buf.mean()
                buf *= buf.square().mean().rsqrt()

    return w_out.repeat([1, G.mapping.num_ws, 1])
#network_pkl='/home/jupyter-skanta2/Sandeep_Pytorch/results_256/00002--auto8-resumecustom/network-snapshot-002419.pkl'
network_pkl='/home/jupyter-skanta2/Collaboratory/Sandeep/results_mammo_256_2/00006--auto4-kimg30700-batch128/network-snapshot-012083.pkl'
#target_fname='/home/jupyter-skanta2/Sandeep_Pytorch/Synthetic_Images_Mixed/Black_C_female_1.jpg'
outdir='/home/jupyter-skanta2/Collaboratory/Sandeep/results_mammo_projecting_images_cropped'
save_video=False
seed=303
num_steps=1000
# import pandas as pd
# df1=pd.read_csv('/home/jupyter-skanta2/Collaboratory/race_detection_data/Emory_CXR_test_censored.csv')
# np.random.seed(seed)
# torch.manual_seed(seed)
for i in range(len(imgs)): #
    # Load networks.
    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as fp:
        G = legacy.load_network_pkl(fp)['G_ema'].requires_grad_(False).to(device) # type: ignore

    # Load target image.
    try:
        img=cv2.imread('/home/jupyter-skanta2'+str(imgs[i]))
        img1=img.astype('uint8')
        img= np.array(PIL.Image.fromarray(img1).convert('L'))
            # Otsu's thresholding after Gaussian filtering
        if np.mean(img[:,:255])<np.mean(img[:,img.shape[1]-255:]):
            img=cv2.flip(img,1)
        blur = cv2.GaussianBlur(img,(5,5),0)
        #print(blur)
        _, breast_mask = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

        cnts, _ = cv2.findContours(breast_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnt = max(cnts, key = cv2.contourArea)
        x, y, w, h = cv2.boundingRect(cnt)

        image=np.array(PIL.Image.fromarray(img[y:y+h, x:x+w]).convert('RGB'))
        image=cv2.resize(image,(256,256))
        image=image.astype('uint8')
        target_pil=np.array(PIL.Image.fromarray(image).convert('L'))
        #target_pil = imgs[i]
        normalized = (target_pil.astype(np.uint16) - target_pil.min()) * 255.0 / (target_pil.max() - target_pil.min())
        normalized_3ch = np.stack((normalized,)*3, axis=-1).astype(np.uint8)
        target_pil = PIL.Image.fromarray(normalized_3ch, mode='RGB')
        #target_pil = PIL.Image.fromarray(target_pil, 'RGB')
        w, h = target_pil.size
        s = min(w, h)
        target_pil = target_pil.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
        target_pil = target_pil.resize((G.img_resolution, G.img_resolution), PIL.Image.LANCZOS)
        target_uint8 = np.array(target_pil, dtype=np.uint8)
    except:
        continue

    # Optimize projection.
    start_time = perf_counter()
    projected_w_steps = project(
        G,
        target=torch.tensor(target_uint8.transpose([2, 0, 1]), device=device), # pylint: disable=not-callable
        num_steps=num_steps,
        device=device,
        verbose=True
    )
    print (f'Elapsed: {(perf_counter()-start_time):.1f} s')

    # Render debug output: optional video and projected image and W vector.
#     os.makedirs(outdir, exist_ok=True)
#     if save_video:
#         #video = imageio.get_writer(f'{outdir}/proj.mp4', mode='I', fps=10, codec='libx264', bitrate='16M')
#         #print (f'Saving optimization progress video "{outdir}/proj.mp4"')
#         for projected_w in projected_w_steps:
#             synth_image = G.synthesis(projected_w.unsqueeze(0), noise_mode='const')
#             synth_image = (synth_image + 1) * (255/2)
#             synth_image = synth_image.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
#             video.append_data(np.concatenate([target_uint8, synth_image], axis=1))
#         video.close()

    # Save final projected frame and W vector.
    #target_pil.save(f'{outdir}/target.png')
    projected_w = projected_w_steps[-1]
    synth_image = G.synthesis(projected_w.unsqueeze(0), noise_mode='const')
    synth_image = (synth_image + 1) * (255/2)
    synth_image = synth_image.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
    #PIL.Image.fromarray(target_uint8, 'RGB').save(f'{outdir}/target_8_.png')
    PIL.Image.fromarray(synth_image, 'RGB').save(f'{outdir}/proj.png')
    os.rename('/home/jupyter-skanta2/Collaboratory/Sandeep/results_mammo_projecting_images_cropped/proj.png','/home/jupyter-skanta2/Collaboratory/Sandeep/results_mammo_projecting_images_cropped/proj'+'_'+str(i)+'.png')
#     try:
#         if os.path.isfile('/home/jupyter-skanta2/Sandeep_Pytorch/results_CXR_2/'+i.split('/')[-1].split('.')[0]+'_'+j+'.'+i.split('/')[-1].split('.')[1]):
#             pass
#         else:
#             os.rename('/home/jupyter-skanta2/Sandeep_Pytorch/results_CXR_2/proj.png','/home/jupyter-skanta2/Sandeep_Pytorch/results_CXR_2/'+i.split('/')[-1].split('.')[0]+'_'+j+'.'+i.split('/')[-1].split('.')[1])
#     except:
#         if os.path.isfile('/home/jupyter-skanta2/Sandeep_Pytorch/results_CXR_2/'+i.split('/')[-1].split('.')[0]+'_'+j.split('/')[0]+'.'+i.split('/')[-1].split('.')[1]):
#             pass
#         else:
#             os.rename('/home/jupyter-skanta2/Sandeep_Pytorch/results_CXR_2/proj.png','/home/jupyter-skanta2/Sandeep_Pytorch/results_CXR_2/'+i.split('/')[-1].split('.')[0]+'_'+j.split('/')[0]+'.'+i.split('/')[-1].split('.')[1])
    #np.savez(f'{outdir}/projected_w.npz', w=projected_w.unsqueeze(0).cpu().numpy())