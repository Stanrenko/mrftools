from mrftools import config as cc
import mrftools.utils_simu as us
import mrftools.image_series as imgser
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import importlib
importlib.reload(us)
importlib.reload(cc)
import pickle
from mrftools import io
import mrftools.utils_mrf as ut_mrf
from mrftools.trajectory import Radial
import mrftools.utils_reco as main
import os
importlib.reload(main)
import logging
logging.basicConfig(level=logging.INFO)
from mrftools.trajectory import *
from mutools import io
 
L0 = 20


DirName = r'/mnt/rmn_files/0_Wip/New/1_Methodological_Developments/1_Methodologie_3T/&3_2017_Fingerprinting_FF_T1/3_Data/0_Phantom/250701'
datfile = 'meas_MID00022_FID48638_raFin_T1T2_FA5_RF1700us'

data,dico_seqParams = main.extract_data(os.path.join(DirName, datfile + '.dat'),dens_adj=True)

SEQ_CONFIG = cc.SEQ_CONFIG_pSSFP4
SEQ_CONFIG['dTR'] = dico_seqParams['dTR']
DICT_CONFIG = cc.DICT_CONFIG2six 
DICT_CONFIG_LIGHT = cc.DICT_LIGHT_CONFIG2six


# dico_folder = r'/mnt/rmn_files/0_Wip/New/1_Methodological_Developments/1_Methodologie_3T/&3_2017_Fingerprinting_FF_T1/2_Codes_Info/python/mrftools/dico'
# main.generate_dictionaries_mrf_generic(SEQ_CONFIG,DICT_CONFIG,DICT_CONFIG_LIGHT, useGPU = True, batch_size = {'water': 50000, 'fat': 50000}, dest=dico_folder, diconame=f"dico_pSSFP_acq_L0{L0}",is_build_phi=True,L0=L0)

# dico_file = f'/mnt/rmn_files/0_Wip/New/1_Methodological_Developments/1_Methodologie_3T/&3_2017_Fingerprinting_FF_T1/2_Codes_Info/python/mrftools/dico/dico_pSSFP_acq_L0{L0}_TR2.1009999999999995_reco5000.pkl'
dico_file = f'/mnt/rmn_files/0_Wip/New/1_Methodological_Developments/1_Methodologie_3T/&3_2017_Fingerprinting_FF_T1/2_Codes_Info/python/mrftools/dico/dico_pSSFP_acq_L0{L0}_TR{dico_seqParams['dTR']}_reco5000.pkl'
# dico_file = f'/mnt/rmn_files/0_Wip/New/1_Methodological_Developments/1_Methodologie_3T/&3_2017_Fingerprinting_FF_T1/2_Codes_Info/python/mrftools/dico/dico_pSSFP_acq_L0{L0}_TR3.3449999999999998_reco5000.pkl'
# dico_file = f'/mnt/rmn_files/0_Wip/New/1_Methodological_Developments/1_Methodologie_3T/&3_2017_Fingerprinting_FF_T1/2_Codes_Info/python/mrftools/dico/dico_pSSFP_acq_L0{L0}_TR2.5949999999999998_reco5000.pkl'
# dico_file = f'/mnt/rmn_files/0_Wip/New/1_Methodological_Developments/1_Methodologie_3T/&3_2017_Fingerprinting_FF_T1/2_Codes_Info/python/mrftools/dico/dico_pSSFP_acq_L0{L0}_TR3.11_reco5000.pkl'

with open(dico_file, 'rb') as fichier:
    dico = pickle.load(fichier)


ll = 10e-6
niter = 10




# datfile = 'meas_MID00027_FID46837_raFin_T1T2_FA5_RF1200us_FAbis'

data,dico_seqParams = main.extract_data(os.path.join(DirName, datfile + '.dat'),dens_adj=True)
b1 = main.calculate_sensitivity_map(data)
os.makedirs(os.path.join(DirName, datfile), exist_ok=True)
volumes_singular = main.build_volumes_singular_iterative(data, b1, dico['phi'], L0=L0,niter=niter,regularizer="wavelet",dens_adj=True,lambd=ll,mu=1)
io.write(os.path.join(DirName, datfile, f"volumes_singular_angle_L0{L0}_lambd_{int(np.round(ll*10e6))}_pSSFP.mha"), np.angle(volumes_singular))
io.write(os.path.join(DirName, datfile, f"volumes_singular_abs_L0{L0}_lambd_{int(np.round(ll*10e6))}_pSSFP.mha"), np.abs(volumes_singular))
mask = main.build_mask_from_singular_volume(volumes_singular, threshold=0.02)
all_maps = main.build_maps(volumes_singular,mask,dico_file,useGPU=True,split=60,return_cost=True,pca=L0,volumes_type="singular", clustering_windows= {"wT1": 2000, "wT2": 100, "fT1": 400, "fT2": 100, "att": 1.0, "df": 0.120})
os.makedirs(os.path.join(DirName, datfile+ f"/fit_us_L0{L0}_lambd_{int(np.round(ll*10e6))}_niter_{niter}_pSSFP"), exist_ok=True)
main.save_maps(all_maps, file_seqParams=None, keys=["ff", "wT1", "wT2", "att", "df"], dest=os.path.join(DirName, datfile) + f"/fit_us_L0{L0}_lambd_{int(np.round(ll*10e6))}_niter_{niter}_pSSFP")



