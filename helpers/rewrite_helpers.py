import os
from tqdm import tqdm
import torch
import torch as ch
import torch.nn.functional as F
from torchvision.transforms import ToPILImage
from tqdm import tqdm
import numpy as np
import helpers.context_helpers as coh
from tools import renormalize, nethook

import pickle as pkl
from helpers.wrappers import Conv2d
from PIL import Image
import numpy as np
import torchvision.transforms as T


def downscale_mask(mask, tgt_size, threshold=None):
    src_size = mask.shape[-1]
    print('src_size',src_size)
    print('tgt_size',tgt_size)
    factor = src_size // tgt_size
    assert src_size == factor * tgt_size
    pooled = F.avg_pool2d(mask, factor, stride=factor)
    if threshold is not None:
        return pooled > threshold
    else:
        return pooled

def target_weights(target_model):
    return [p for n, p in target_model.named_parameters()
            if 'weight' in n][0]

def projected_conv(weight, direction, unfold=False):
    if len(weight.shape) == 5:
        cosine_map = torch.einsum('goiyx, di -> godyx', weight, direction)
        result = torch.einsum('godyx, di -> goiyx', cosine_map, direction)
    else:
        if unfold:
            direction_r = direction.unsqueeze(2).unsqueeze(3)
            direction_r = direction_r.reshape(direction_r.shape[0],
                                               weight.shape[1],
                                               weight.shape[2],
                                               weight.shape[3]).transpose(2, 3)
            cosine_map = torch.einsum('oiyx, diyx -> od', weight, direction_r)
            result = torch.einsum('od, diyx -> oiyx', cosine_map, direction_r)
        else:
            cosine_map = torch.einsum('oiyx, di -> odyx', weight, direction)
            result = torch.einsum('odyx, di -> oiyx', cosine_map, direction)
    return result

def edit_classifier_weights(keys, values, context=None, target_model=None, 
                           niter=2001, piter=10, lr=0.05, 
                           low_rank_insert=True, low_rank_gradient=False,
                           unfold=False, mask=None):
    
    def update_callback(it, loss, pbar=None):
        if it % 50 == 0 or it == niter - 1:
            loss_info = (f'lr {lr:.4f}\titer {it: 6d}/{niter: 6d}'
                         f'\tloss {loss.item():.4f}')
            if pbar:
                pbar.set_description(str(loss))
            else:
                print(loss_info)
    
    # tune the weights
    tuned_weights = []
    i = 0
    for (key,val) in zip(keys, values):
        if i==4:
            break
        try:
            key, val = [d.detach() for d in [key, val]]
        except:
            val = val.detach()

        def compute_loss(weight, bias, mask=None):
            adjusted_val = F.conv2d(key, weight, bias=bias, stride=1, padding=1)
            reps = val, adjusted_val
            if mask is not None:
                mask = downscale_mask(mask, val.shape[-1], None)
                mask = mask.sqrt()
                reps = [r * mask for r in reps]
            return torch.nn.functional.l1_loss(*reps) / len(val)

        # set up optimizer
        with open('./original_params/biases','rb') as f:
            biases = pkl.load(f)
        bias = biases[i]

        with open('./original_params/weights','rb') as f:
            weights = pkl.load(f)
        weight = weights[i].requires_grad_()
        weight_orig = weight.clone()
        params = [weight]
        if low_rank_insert or low_rank_gradient:
            with torch.no_grad():
                ortho_weight = weight - projected_conv(weight, context, unfold=unfold)
                
        optimizer = torch.optim.Adam(params, lr=lr)

        pbar = tqdm(range(niter))
        for it in pbar:
            with torch.enable_grad():
                loss = compute_loss(weight, bias, mask)
                optimizer.zero_grad()
                loss.backward()

                if it == 0: loss_orig = loss.item()

                if low_rank_gradient:
                    weight.grad[...] = projected_conv(weight.grad, context, unfold=unfold)
                optimizer.step()
                if update_callback is not None:
                    update_callback(it, loss, pbar=pbar)
                if low_rank_insert and (it % piter == 0 or it == niter - 1):
                    with torch.no_grad():
                        weight[...] = (
                            ortho_weight + projected_conv(weight, context, unfold=unfold))

        print("Loss (orig, final):", loss_orig, loss.item())
        print("L2 norm of weight change:", ch.norm(weight_orig - weight).item())
        tuned_weights.append(weight)
        i+=1

    # save tuned weights
    with open('./tuned_weights','wb') as f:
        pkl.dump(tuned_weights, f)
    

def edit_classifier(args, key_path, value_path, train_data=None, 
               context_model=None, 
               target_model=None, 
               val_loader=None,
               caching_dir=None):
    
    if train_data is not None:
        assert args.ntrain <= len(train_data['imgs'])
        cp_imgs = ch.cat([train_data['imgs'][:args.ntrain], 
                          train_data['modified_imgs'][:args.ntrain]]).float()
        cp_masks = ch.cat([train_data['masks'][:args.ntrain], 
                           train_data['masks'][:args.ntrain]]).float()
        
        Nims = len(cp_imgs)
    

    if args.mode_rewrite == 'editing':
        if context_model is not None:
            _, ZM_k = coh.get_cov_matrix(val_loader, context_model, 
                                    batch_size=2000, 
                                    key_method='zca',
                                    caching_dir=caching_dir)
        
            assert (target_model is not None) and (ZM_k is not None)
            context_k = coh.get_context_key(train_data['modified_imgs'].float(), 
                                            train_data['masks'], 
                                            context_model, ZM_k, rank=args.rank)
            with ch.no_grad(): context_model(cp_imgs.cuda())
            kstar = coh.features['pre']
            vstar = coh.features['post'][:Nims//2].detach().clone()
            kstar = (kstar[0][Nims//2:].detach().clone(), 
                     kstar[1][Nims//2:].detach().clone()) if not args.arch.startswith('vgg') \
                    else kstar[Nims//2:].detach().clone()
            mstar = ch.max(cp_masks[:Nims//2], dim=1, keepdims=True)[0]
        
        else:
            with open(value_path,'rb') as f:
                vstar = pkl.load(f)

            with open(key_path,'rb') as f:
                kstar = pkl.load(f)

        
        if target_model is not None:
            edit_classifier_weights(target_model, kstar, vstar, 
                                       context_k, niter=args.nsteps, 
                                       piter=args.nsteps_proj, lr=args.lr, 
                                       low_rank_insert=args.restrict_rank, 
                                       mask=mstar.cuda() if args.use_mask else None)
        else:
            edit_classifier_weights(kstar, vstar, niter=args.nsteps, 
                                       piter=args.nsteps_proj, lr=args.lr, 
                                       low_rank_insert=args.restrict_rank)

    else:
        if args.arch == 'resnet50':
            first_layer = f'layer{args.layernum + 1}.final.conv3'  
        elif args.arch == 'resnet18':
            first_layer = f'layer{args.layernum + 1}.final.conv2'  
        elif args.arch.startswith('vgg'):
            first_layer = f'layer{args.layernum}.conv'
        else:
            first_layer = f'visual.layer{args.layernum + 1}.final'  
                    
                
        if args.mode_rewrite == 'finetune_local':
            edit_params = [target_weights(target_model)]
        else:
            edit_model = nethook.subsequence(context_model, 
                                             first_layer=first_layer,
                                             share_weights=True)
            
            edit_params = edit_model.parameters()
            
            if args.arch.startswith('clip'):
                edit_params = []
                for name, param in edit_model.named_parameters():
                    if 'visual' in name:
                        edit_params.append(param)

        optimizer = ch.optim.SGD(edit_params, lr=args.lr)
        compute_loss = torch.nn.CrossEntropyLoss()
        pbar = tqdm(range(args.nsteps))
        
        imgs = train_data['modified_imgs'][:args.ntrain].float()
        target_label = np.unique(train_data['labels'][:args.ntrain].numpy())
        assert len(target_label) == 1
        
        tgts = ch.tensor([target_label[0]] * len(imgs))
        
        with torch.enable_grad():
            for i in pbar:
                loss = compute_loss(context_model(imgs.cuda()), tgts.cuda())
                optimizer.zero_grad()
                loss.backward()
                pbar.set_description(str(loss))
                optimizer.step()
        loss.detach()
       
    return context_model