from mrftools import config as cc
import mrftools.utils_simu as us
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import importlib
importlib.reload(us)
importlib.reload(cc)
import mrftools.main_functions_2D as main_fun2D
import pickle
from mrftools import io
import mrftools.utils_mrf as ut_mrf
from mrftools.trajectory import Radial
import mrftools.main_functions_2D as main
import os
importlib.reload(main)
import logging
logging.basicConfig(level=logging.INFO)

json_file = 'dicoB0'
phantom_file = r'/mnt/rmn_files/0_Wip/New/1_Methodological_Developments/1_Methodologie_3T/&3_2017_Fingerprinting_FF_T1/3_Data/4_Simulations/DataBase_Num_Ph_Python/V5/Phantom7/paramMap.pkl'
phantom_folder = r'/mnt/rmn_files/0_Wip/New/1_Methodological_Developments/1_Methodologie_3T/&3_2017_Fingerprinting_FF_T1/3_Data/4_Simulations/DataBase_Num_Ph_Python/V5/Phantom7'

with open(phantom_file, 'rb') as fichier:
    paramMap = pickle.load(fichier)

with open(f"../config/config_{json_file}.json", 'r', encoding='utf-8') as fichier:
    dict_config = json.load(fichier)

SEQ_CONFIG = cc.SEQ_CONFIG4
# SEQ_CONFIG = cc.SEQ_CONFIG5

# imgseries = us.generate_ImgSeries_T1MRF_generic(sequence_config=SEQ_CONFIG, dict_config=dict_config, maps=paramMap)
# trajectory = Radial(total_nspokes = 1400, npoint = 256)
# imgseries_us = ut_mrf.undersampling_operator_new(imgseries, trajectory, np.ones((256,256)), ntimesteps=175, light_memory_usage=False)

# io.write('../data/phantom6/imgseries.mha', np.abs(imgseries))
# io.write('../data/phantom6/imgseries_us.mha', np.abs(imgseries_us))
# io.write('../data/phantom6/imgseries_angle.mha', np.angle(imgseries))
# io.write('../data/phantom6/imgseries_angle_us.mha', np.angle(imgseries_us))

# with open('../data/phantom6/imgseries.pkl', 'wb') as fichier:
#     pickle.dump(imgseries, fichier)
# with open('../data/phantom6/imgseries_us.pkl', 'wb') as fichier:
#     pickle.dump(imgseries_us, fichier)



DICT_CONFIG = cc.DICT_CONFIG2bis 
DICT_CONFIG_LIGHT = cc.DICT_LIGHT_CONFIG2bis

# DICT_CONFIG = cc.DICT_CONFIG6 
# DICT_CONFIG = cc.DICT_LIGHT_CONFIG6
# DICT_CONFIG_LIGHT = cc.DICT_LIGHT_CONFIG6


# main.generate_dictionaries_mrf_generic(SEQ_CONFIG,DICT_CONFIG,DICT_CONFIG_LIGHT, useGPU = True, batch_size = {'water': 50000, 'fat': 50000}, dest='../dico',diconame="dico_pSSFP",is_build_phi=True,L0=40)

dico_file = '../dico/dico_pSSFP_TR1.17_reco5000.pkl'


with open('../data/phantom6/imgseries.pkl', 'rb') as fichier:
    imgseries = pickle.load(fichier)
    
imgseries = imgseries[:,np.newaxis,:,:]
mask = paramMap['mask'][np.newaxis,:,:]

all_maps = main.build_maps(imgseries,mask,dico_file,useGPU=True,split=40,return_cost=True,pca=175,volumes_type="raw", clustering_windows= {"wT1": 2000, "wT2": 80, "fT1": 400, "fT2": 100, "att": 1.0, "df": 0.120})
main.save_maps(all_maps, file_seqParams=None, keys=["ff", "wT1", "wT2", "att", "df"], dest='../data/phantom6')

# io.write(os.path.join(phantom_folder, "wT1_map.mha"), paramMap['WATER_T1'][:,:,np.newaxis])
# io.write(os.path.join(phantom_folder, "wT2_map.mha"), paramMap['WATER_T2'][:,:,np.newaxis])
# io.write(os.path.join(phantom_folder, "attB1_map.mha"), paramMap['att'][:,:,np.newaxis])
# io.write(os.path.join(phantom_folder, "df_map.mha"), paramMap['DF'][:,:,np.newaxis])
# io.write(os.path.join(phantom_folder, "ff_map.mha"), paramMap['FF'][:,:,np.newaxis])
# io.write(os.path.join(phantom_folder, "mask.mha"), paramMap['mask'][:,:,np.newaxis])
# io.write(os.path.join(phantom_folder, "fT1_map.mha"), paramMap['FAT_T1'][:,:,np.newaxis])
# io.write(os.path.join(phantom_folder, "fT2_map.mha"), paramMap['FAT_T2'][:,:,np.newaxis])
