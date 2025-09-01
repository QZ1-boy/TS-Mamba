from thop import profile
import torch
import sys
import os
import torch.nn as nn
import time
from torchstat import stat
import warnings
warnings.filterwarnings('ignore')

from basicsr.archs.tsaabl_arch import TSAabl
from basicsr.archs.tsa_arch import TSA
from basicsr.archs.tmp_arch import TMP


############# TSA abl ####################
modelS_ours = TSAabl()
modelS_ours_path = '/share3/home/zqiang/TMP/experiments/TSA_train_REDS_Abl/models/net_g_1.pth'
# modelS_ours.load_state_dict(torch.load(modelS_ours_path, map_location='cpu'))
modelS_ours.load_state_dict(torch.load(modelS_ours_path)['params'], strict=True)
modelS_ours = modelS_ours.cuda()
inputs = torch.rand((1, 7, 3, 180, 320)).cuda()
modelS_ours = modelS_ours.cuda()
inputs = inputs.unsqueeze(0)
start_time = time.time()
macs, param = (profile(modelS_ours, inputs))
print('[TSA_abl] Time:', (time.time()-start_time)/7, 'FPS(1/s):', 7/(time.time() -start_time), 'FLOPs (G):', 2*macs/1000/1000/1000/7, 'MACs (G):', macs/1000/1000/1000/7, 'Param (M):', param/1000/1000)




# ############# TSA  ####################
# modelS_ours = TSA()
# modelS_ours_path = '/share3/home/zqiang/TMP/experiments/TSA_train_REDS_zero/models/net_g_5000.pth'
# # modelS_ours.load_state_dict(torch.load(modelS_ours_path, map_location='cpu'))
# modelS_ours.load_state_dict(torch.load(modelS_ours_path)['params'], strict=True)
# modelS_ours = modelS_ours.cuda()
# inputs = torch.rand((1, 7, 3, 180, 320)).cuda()
# modelS_ours = modelS_ours.cuda()
# inputs = inputs.unsqueeze(0)
# start_time = time.time()
# macs, param = (profile(modelS_ours, inputs))
# print('[TSA] Time:', (time.time()-start_time)/7, 'FPS(1/s):', 7/(time.time() -start_time), 'FLOPs (G):', 2*macs/1000/1000/1000/7, 'MACs (G):', macs/1000/1000/1000/7, 'Param (M):', param/1000/1000)

# # ############# TMP  ####################
# modelS_ours = TMP()
# modelS_ours_path = '/share3/home/zqiang/TMP/experiments/TMP_train_Vimeo_BD_bk/models/net_g_50000.pth'
# # modelS_ours.load_state_dict(torch.load(modelS_ours_path, map_location='cpu'))
# modelS_ours.load_state_dict(torch.load(modelS_ours_path)['params'], strict=True)
# modelS_ours = modelS_ours.cuda()
# inputs = torch.rand((1, 7, 3, 180, 320)).cuda()
# modelS_ours = modelS_ours.cuda()
# inputs = inputs.unsqueeze(0)
# start_time = time.time()
# macs, param = (profile(modelS_ours, inputs))
# print('[TMP] Time:', (time.time()-start_time)/7, 'FPS(1/s):', 7/(time.time() -start_time), 'FLOPs (G):', 2*macs/1000/1000/1000/7, 'MACs (G):', macs/1000/1000/1000/7, 'Param (M):', param/1000/1000)