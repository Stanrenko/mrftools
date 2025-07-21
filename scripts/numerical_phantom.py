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

json_file = 'dicoB0'
phantom_file = r'/mnt/rmn_files/0_Wip/New/1_Methodological_Developments/1_Methodologie_3T/&3_2017_Fingerprinting_FF_T1/3_Data/4_Simulations/DataBase_Num_Ph_Python/V5/Phantom8/paramMap.pkl'
phantom_folder = r'/mnt/rmn_files/0_Wip/New/1_Methodological_Developments/1_Methodologie_3T/&3_2017_Fingerprinting_FF_T1/3_Data/4_Simulations/DataBase_Num_Ph_Python/V5/Phantom8'

with open(phantom_file, 'rb') as fichier:
    paramMap = pickle.load(fichier)

# with open(f"../config/config_{json_file}.json", 'r', encoding='utf-8') as fichier:
#     dict_config = json.load(fichier)


# SEQ_CONFIG = cc.SEQ_CONFIG5

# imgseries = us.generate_ImgSeries_T1MRF_generic(sequence_config=SEQ_CONFIG, dict_config=dict_config, maps=paramMap)
# trajectory = Radial(total_nspokes = 1400, npoint = 256)
# imgseries_us = ut_mrf.undersampling_operator_new(imgseries, trajectory, np.ones((256,256)), ntimesteps=175, light_memory_usage=False)

# io.write('../data/phantom8/imgseries.mha', np.abs(imgseries))
# io.write('../data/phantom8/imgseries_us.mha', np.abs(imgseries_us))
# io.write('../data/phantom8/imgseries_angle.mha', np.angle(imgseries))
# io.write('../data/phantom8/imgseries_angle_us.mha', np.angle(imgseries_us))

# with open('../data/phantom8/imgseries.pkl', 'wb') as fichier:
#     pickle.dump(imgseries, fichier)
# with open('../data/phantom8/imgseries_us.pkl', 'wb') as fichier:
#     pickle.dump(imgseries_us, fichier)


L0 = 8
############# Dico Generation ############# 

# SEQ_CONFIG = cc.SEQ_CONFIG_pSSFP4
# DICT_CONFIG = cc.DICT_CONFIG2quat 
# DICT_CONFIG_LIGHT = cc.DICT_LIGHT_CONFIG2bis


# dico_folder = r'/mnt/rmn_files/0_Wip/New/1_Methodological_Developments/1_Methodologie_3T/&3_2017_Fingerprinting_FF_T1/2_Codes_Info/python/mrftools/dico'
# main.generate_dictionaries_mrf_generic(SEQ_CONFIG,DICT_CONFIG,DICT_CONFIG_LIGHT, useGPU = True, batch_size = {'water': 50000, 'fat': 50000}, dest=dico_folder, diconame=f"dico_pSSFP_generic8_L0{L0}",is_build_phi=True,L0=L0)


######### Phantom Generation #############


# with open(phantom_file, 'rb') as fichier:
#     imgseries = pickle.load(fichier)
    
# imgseries = imgseries[:,np.newaxis,:,:]
# mask = paramMap['mask'][np.newaxis,:,:]

# all_maps = main.build_maps(imgseries,mask,dico_file,useGPU=True,split=40,return_cost=True,pca=175,volumes_type="raw", clustering_windows= {"wT1": 2000, "wT2": 80, "fT1": 400, "fT2": 100, "att": 1.0, "df": 0.120})
# all_maps = main.build_maps(imgseries,mask,dico_file,useGPU=True,split=40,return_cost=True,pca=20,volumes_type="raw")
# main.save_maps(all_maps, file_seqParams=None, keys=["ff", "wT1", "wT2", "att", "df"], dest='../data/phantom8')
# main.save_maps(all_maps, file_seqParams=None, keys=["ff", "wT1", "wT2", "attB1", "df"], dest='../data/phantom6')

# io.write(os.path.join(phantom_folder, "wT1_map.mha"), paramMap['WATER_T1'][:,:,np.newaxis])
# io.write(os.path.join(phantom_folder, "wT2_map.mha"), paramMap['WATER_T2'][:,:,np.newaxis])
# io.write(os.path.join(phantom_folder, "attB1_map.mha"), paramMap['att'][:,:,np.newaxis])
# io.write(os.path.join(phantom_folder, "df_map.mha"), paramMap['DF'][:,:,np.newaxis])
# io.write(os.path.join(phantom_folder, "ff_map.mha"), paramMap['FF'][:,:,np.newaxis])
# io.write(os.path.join(phantom_folder, "mask.mha"), paramMap['mask'][:,:,np.newaxis])
# io.write(os.path.join(phantom_folder, "fT1_map.mha"), paramMap['FAT_T1'][:,:,np.newaxis])
# io.write(os.path.join(phantom_folder, "fT2_map.mha"), paramMap['FAT_T2'][:,:,np.newaxis])


####### Singular Volume Generation #############

SEQ_CONFIG = cc.SEQ_CONFIG_pSSFP4
seq = us.t1_mfr_seq(SEQ_CONFIG)
imgseries = imgser.MapFromDict(name = 'imgseries', paramMap=paramMap)
imgseries.buildParamMap()
imgseries.build_ref_images_v2(seq, useGPU=True)
trajectory = Radial(total_nspokes = 1600, npoint = 512)

kdata = np.asarray(imgseries.generate_kdata(trajectory, useGPU=False), dtype=np.complex64)[np.newaxis,np.newaxis,:,:]


print("Performing radial density adjustment")
npoint = kdata.shape[-1]
density = np.abs(np.linspace(-1, 1, npoint))
density = np.abs(np.linspace(-1, 1, npoint))
density = np.expand_dims(density, tuple(range(kdata.ndim - 1)))
kdata*=density
        
        
b1 = np.ones((1,1,int(kdata.shape[3]/2),int(kdata.shape[3]/2)), dtype=np.complex64)   

dico_file = f'/mnt/rmn_files/0_Wip/New/1_Methodological_Developments/1_Methodologie_3T/&3_2017_Fingerprinting_FF_T1/2_Codes_Info/python/mrftools/dico/dico_pSSFP_generic4_L0{L0}_TR1.17_reco5000.pkl'

with open(dico_file, 'rb') as fichier:
    dico = pickle.load(fichier)

ll = 10
niter = 10

volumes_singular = main.build_volumes_singular_iterative(kdata, b1, dico['phi'], L0=L0,niter=niter,regularizer="wavelet",dens_adj=True,lambd=ll,mu=1)
phantom_folder = r'/mnt/rmn_files/0_Wip/New/1_Methodological_Developments/1_Methodologie_3T/&3_2017_Fingerprinting_FF_T1/3_Data/4_Simulations/DataBase_Num_Ph_Python/V5/Phantom8'

# io.write(os.path.join(phantom_folder, f"volumes_singular_abs_L0{L0}_lambd_{ll}_niter_{niter}_pSSFP_generic4.mha"), np.abs(volumes_singular))
# io.write(os.path.join(phantom_folder, f"volumes_singular_angle_L0{L0}_lambd_{ll}_niter_{niter}_pSSFP_generic4.mha"), np.angle(volumes_singular))


####### Mapping #######

volumes_singular = volumes_singular[:,np.newaxis,:,:]
mask = paramMap['mask'][np.newaxis,:,:]

all_maps = main.build_maps(volumes_singular,mask,dico_file,useGPU=True,split=40,return_cost=True,pca=L0,volumes_type="singular", clustering_windows= {"wT1": 2000, "wT2": 80, "fT1": 400, "fT2": 100, "att": 1.0, "df": 0.120})
os.makedirs(phantom_folder+f"/fit_us_L0{L0}_lambd_{ll}_niter_{niter}_pSSFP_generic4", exist_ok=True)
main.save_maps(all_maps, file_seqParams=None, keys=["ff", "wT1", "wT2", "att", "df"], dest=phantom_folder+f"/fit_us_L0{L0}_lambd_{ll}_niter_{niter}_pSSFP_generic4")