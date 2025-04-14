import json
import pathlib
import numpy as np
import itertools
import epgpy as epg
import pickle
from epgpy import common, operators, statematrix, utils

from .utils_mrf import groupby
from .dictmodel import Dictionary
from mrftools import io
import matplotlib.pyplot as plt
import os

class T1MRF_generic:
    def __init__(self, SEQ_CONFIG):
        """ build sequence """
        self.seqlen = len(SEQ_CONFIG['TE'])
        self.READOUT = SEQ_CONFIG['readout']
        self.TI = SEQ_CONFIG['TI']
        self.TE = SEQ_CONFIG['TE']
        self.FA = SEQ_CONFIG['FA']*SEQ_CONFIG['B1']
        TR = [i + SEQ_CONFIG['dTR'] for i in self.TE]
        self.TR = TR
        self.inversion = epg.T(180, 0) 
        
        self.T_recovery=SEQ_CONFIG['T_recovery']
        # self.nrep=nrep
        # self.rep=rep
        seq=[]
        if SEQ_CONFIG['readout'] == 'FLASH_ideal': 
            self.spl = epg.SPOILER
            self.PHI = [50 * i * (i + 1) / 2 for i in range(len(self.TE))]
        elif SEQ_CONFIG['readout'] == 'FLASH': 
            self.spl = epg.S(1)
            self.PHI = [50 * i * (i + 1) / 2 for i in range(len(self.TE))]
        elif SEQ_CONFIG['readout'] == 'FISP': 
            self.spl = epg.S(1)
            self.PHI = np.array([0]*len(self.TE))
        elif SEQ_CONFIG['readout'] == 'pSSFP': 
            self.spl = epg.S(1)
            self.PHI = [1 * i * (i + 1) / 2 for i in range(len(self.TE))]
        elif SEQ_CONFIG['readout'] == 'pSSFP_2': 
            self.spl = epg.S(1)
            self.PHI = np.array([1 * i * (i + 1) / 2 for i in range(int(np.round(len(self.TE)/2)))] + [(-1) * i * (i + 1) / 2 for i in range(int(np.round(len(self.TE)/2)))])            
        elif SEQ_CONFIG['readout'] == 'TrueFISP':
            self.spl = epg.NULL
            self.PHI = np.array([0, 180] * (len(self.TE) // 2) + [0] * (len(self.TE) % 2)) 

    def __call__(self, T1, T2, g, att, cs, frac, **kwargs):
        """ simulate sequence """
        self.init = [epg.PD(frac[j]) for j in range(len(frac))]
        self.rf = [epg.T(self.FA[i]*att, self.PHI[i]) for i in range(self.seqlen)] 
        self.adc = [epg.Adc(phase=-self.rf[i].phi) for i in range(self.seqlen)]
        self.rlx0 = epg.E(self.TI, T1, T2, duration=True) # inversion delay
        self.rlx1 = [[epg.E(self.TE[i], T1, T2, cs[j] + g, duration=True) for i in range(len(self.FA))]  for j in range(len(cs))] 
        self.rlx2 = [[epg.E(self.TR[i] - self.TE[i], T1, T2, cs[j] + g, duration=True) for i in range(len(self.FA))] for j in range(len(cs))] 
        self.rlx3 = epg.E(self.T_recovery, T1, T2, duration=True) # inversion delay
                
        seq = [[self.init[j]] + [self.inversion] + [self.rlx0] + [[self.rf[i], self.rlx1[j][i], self.adc[i], self.rlx2[j][i], self.spl] for i in range(self.seqlen)] for j in range(len(cs))] 
  
        result=np.asarray(epg.simulate(seq, disp=True, max_nstate=30))
        
        result = np.reshape(result, [len(cs),len(self.FA),*np.shape(result)[1:]])
        result = np.sum(np.asarray(frac)[:, np.newaxis, np.newaxis] * result , axis=0)
        
        return result

class T1MRFSS:
    def __init__(self, FA, TI, TE, TR, B1,T_recovery,nrep,rep=None):
        """ build sequence """
        seqlen = len(TE)
        self.TR=TR
        self.inversion = epg.T(180, 0) # perfect inversion
        self.T_recovery=T_recovery
        self.nrep=nrep
        self.rep=rep
        seq=[]
        for r in range(nrep):
            curr_seq = [epg.Offset(TI)]
            for i in range(seqlen):
                echo = [
                    epg.T(FA * B1[i], 90),
                    epg.Wait(TE[i]),
                    epg.ADC,
                    epg.Wait(TR[i] - TE[i]),
                    epg.SPOILER,
                ]
                curr_seq.extend(echo)
            recovery=[epg.Wait(T_recovery)]
            curr_seq.extend(recovery)
            self.len_rep = len(curr_seq)
            seq.extend(curr_seq)
        self._seq = seq

    def __call__(self, T1, T2, g, att,**kwargs):
        """ simulate sequence """
        seq=[]
        rep=self.rep
        for r in range(self.nrep):
            curr_seq=self._seq[r*self.len_rep:(r+1)*(self.len_rep)]
            curr_seq=[self.inversion, epg.modify(curr_seq, T1=T1, T2=T2, att=att, g=g)]
            seq.extend(curr_seq)
        #seq = [self.inversion, epg.modify(self._seq, T1=T1, T2=T2, att=att, g=g,calc_deriv=calc_deriv)]
        
        result=np.asarray(epg.simulate(seq, **kwargs))
        if rep is not None:
            result = result.reshape((self.nrep, -1) + result.shape[1:])[rep]
        return result

def modifier_inv(op, **kwargs):
    """default modifier to handle 'T1', 'T2', 'g' and 'att' keywords
    TODO: handle differential operators (options gradients and hessian)
    """
    # print(op)
    # if isinstance(op, operators.T):
    #     print(op.alpha)
    
    if isinstance(op, operators.T) and ((type(op.alpha)==int) or (type(op.alpha)==float)) and (op.alpha == 180):
        # add B1 attenuation
        att = kwargs.get("att")
        if att is None or np.allclose(att, 1):
            pass  # nothing to do
        else:
            # update T operator
            op = operators.T(op.alpha * att, op.phi, name=op.name, duration=op.duration)
            op.name += "#"
     
    return op

class T1MRFSS_ImperfectInv:
    def __init__(self, FA, TI, TE, TR, B1,T_recovery,nrep,rep=None):
        """ build sequence """
        seqlen = len(TE)
        self.TR=TR
        # self.inversion = epg.T(180, 0) # perfect inversion
        self.T_recovery=T_recovery
        self.nrep=nrep
        self.rep=rep
        seq=[]
        for r in range(nrep):
            curr_seq = [epg.Offset(TI)]
            for i in range(seqlen):
                echo = [
                    epg.T(FA * B1[i], 90),
                    epg.Wait(TE[i]),
                    epg.ADC,
                    epg.Wait(TR[i] - TE[i]),
                    epg.SPOILER,
                ]
                curr_seq.extend(echo)
            recovery=[epg.Wait(T_recovery)]
            curr_seq.extend(recovery)
            self.len_rep = len(curr_seq)
            seq.extend(curr_seq)
        self._seq = seq

    def __call__(self, T1, T2, g, att,att_inv,**kwargs):
        """ simulate sequence """
        seq=[]
        rep=self.rep
        for r in range(self.nrep):
            curr_seq=self._seq[r*self.len_rep:(r+1)*(self.len_rep)]
            curr_seq=[epg.T(180*att_inv, 0), epg.modify(curr_seq, T1=T1, T2=T2, att=att, g=g)]
            seq.extend(curr_seq)
        #seq = [self.inversion, epg.modify(self._seq, T1=T1, T2=T2, att=att, g=g,calc_deriv=calc_deriv)]
        
        result=np.asarray(epg.simulate(seq, **kwargs))
        if rep is not None:
            result = result.reshape((self.nrep, -1) + result.shape[1:])[rep]
        return result


def create_new_seq(FA_list,TE_list,min_TR_delay,TI,FA_factor=5):
    seq_config_new={}
    seq_config_new["FA"]=FA_factor
    seq_config_new["TI"]=TI
    seq_config_new["TE"] = list(np.array(TE_list[1:]) * 10 ** 3)
    seq_config_new["TR"] = list((np.array(TE_list[1:])+min_TR_delay) * 10 ** 3)
    seq_config_new["B1"] = list(np.array(FA_list[1:]) * 180 / np.pi / 5)
    
    
    

    return seq_config_new

def generate_epg_dico_T1MRFSS_from_sequence(sequence_config,filedictconf,recovery,rep=2,overwrite=True,sim_mode="mean",start=None,window=None, dest=None,prefix_dico="dico"):
    if type(filedictconf)==str:
        with open(filedictconf) as f:
            dict_config = json.load(f)
    
    elif type(filedictconf)==dict:
        dict_config=filedictconf



    # generate signals
    wT1 = dict_config["water_T1"]
    fT1 = dict_config["fat_T1"]
    wT2 = dict_config["water_T2"]
    fT2 = dict_config["fat_T2"]
    att = dict_config["B1_att"]
    df = dict_config["delta_freqs"]
    df = [- value / 1000 for value in df]  # temp
    # df = np.linspace(-0.1, 0.1, 101)

    TR_total = np.sum(sequence_config["TR"])

    sequence_config["T_recovery"] = recovery*1000
    sequence_config["nrep"] = rep

    TR_delay=np.round(sequence_config["TR"][0]-sequence_config["TE"][0],2)

    seq = T1MRFSS(**sequence_config)


    fat_amp = np.array(dict_config["fat_amp"])
    fat_cs = dict_config["fat_cshift"]
    fat_cs = [- value / 1000 for value in fat_cs]  # temp

    # other options
    if window is None:
        window = dict_config["window_size"]


    if start is None:
        dictfile = prefix_dico  +"_TR{}_reco{}.dict".format(str(TR_delay),str(recovery))
    else:
        dictfile = prefix_dico + "_TR{}_reco{}_start{}.dict".format(str(TR_delay),str(recovery),start)

    if dest is not None:
        dictfile = str(pathlib.Path(dest) / pathlib.Path(dictfile).name)
    # print("Generating dictionary {}".format(dictfile))

    # water
    print("Generate water signals.")
    water = seq(T1=wT1, T2=wT2, att=[[att]], g=[[[df]]])
    water = water.reshape((rep, -1) + water.shape[1:])[-1]

    if sim_mode == "mean":
        water = [np.mean(gp, axis=0) for gp in groupby(water, window)]
    elif sim_mode == "mid_point":
        if start is None:
            start=(int(window / 2) - 1)

        water = water[start:-1:window]
    else:
        raise ValueError("Unknow sim_mode")

    # fat
    print("Generate fat signals.")
    # eval = "dot(signal, amps)"
    # args = {"amps": fat_amp}
    # merge df and fat_cs df to dict
    fatdf = [[cs + f for cs in fat_cs] for f in df]
    fat = seq(T1=[fT1], T2=fT2, att=[[att]], g=[[[fatdf]]])#, eval=eval, args=args)
    fat=fat @ fat_amp
    fat = fat.reshape((rep, -1) + fat.shape[1:])[-1]

    if sim_mode == "mean":
        fat = [np.mean(gp, axis=0) for gp in groupby(fat, window)]
    elif sim_mode == "mid_point":
        if start is None:
            start=(int(window / 2) - 1)
        fat = fat[start:-1:window]
    else:
        raise ValueError("Unknow sim_mode")

    water = np.array(water)
    fat = np.array(fat)
    # join water and fat
    print("Build dictionary.")
    keys = list(itertools.product(wT1, fT1, att, df))
    values = np.stack(np.broadcast_arrays(water, fat), axis=-1)
    values = np.moveaxis(values.reshape(len(values), -1, 2), 0, 1)

    # print("Save dictionary.")
    mrfdict = Dictionary(keys, values)
    # mrfdict.save(dictfile, overwrite=overwrite)
    hdr={"sequence_config":sequence_config,"dict_config":dict_config,"recovery":recovery,"initial_repetitions":rep,"window":window,"sim_mode":sim_mode,"param_names":("wT1","fT1","att","df")}
    return mrfdict,hdr,dictfile

def generate_epg_dico_T1MRF_generic_from_sequence(sequence_config,filedictconf,overwrite=True,sim_mode="mean",start=None,window=None, dest=None,prefix_dico="dico_MRF_generic"):

    if type(filedictconf)==str:
        with open(filedictconf) as f:
            dict_config = json.load(f)
    
    elif type(filedictconf)==dict:
        dict_config=filedictconf



    # generate signals
    wT1 = dict_config["water_T1"]
    fT1 = dict_config["fat_T1"]
    wT2 = dict_config["water_T2"]
    fT2 = dict_config["fat_T2"]
    att = dict_config["B1_att"]
    df = dict_config["delta_freqs"]
    df = [- value / 1000 for value in df]  # temp
    # df = np.linspace(-0.1, 0.1, 101)
    
    combinations = list(itertools.product(fT1, fT2, att, df))
    fat_T1, fat_T2, fat_att, fat_df = zip(*combinations)
    
    combinations = list(itertools.product(wT1, wT2, att, df))
    water_T1, water_T2, water_att, water_df = zip(*combinations)    

    TR_delay=sequence_config["dTR"]

    seq = T1MRF_generic(sequence_config)

    water_amp = [1]
    water_cs = [0]
    fat_amp = np.array(dict_config["fat_amp"])
    fat_cs = dict_config["fat_cshift"]
    fat_cs = [- value / 1000 for value in fat_cs]  # temp

    # other options
    if window is None:
        window = dict_config["window_size"]


    if start is None:
        dictfile = prefix_dico  +"_TR{}_reco{}.dict".format(str(TR_delay),str(sequence_config['T_recovery']))
    else:
        dictfile = prefix_dico + "_TR{}_reco{}_start{}.dict".format(str(TR_delay),str(sequence_config['T_recovery']),start)

    if dest is not None:
        dictfile = str(pathlib.Path(dest) / pathlib.Path(dictfile).name)
    # print("Generating dictionary {}".format(dictfile))


    print("Generate water signals.")
    water = seq(T1=np.array(water_T1), T2=np.array(water_T2), att=np.array(water_att), g=np.array(water_df), cs=water_cs, frac=water_amp)


    if sim_mode == "mean":
        water = [np.mean(gp, axis=0) for gp in groupby(water, window)]
    elif sim_mode == "mid_point":
        if start is None:
            start=(int(window / 2) - 1)

        water = water[start:-1:window]
    else:
        raise ValueError("Unknow sim_mode")
    
    water = np.reshape(water, (np.shape(water)[0], np.shape(wT1)[0],np.shape(wT2)[0], 1, 1, np.shape(att)[0], np.shape(df)[0] ))

    # fat
    print("Generate fat signals.")
    fat = seq(T1=np.array(fat_T1), T2=np.array(fat_T2), att=np.array(fat_att), g=np.array(fat_df), cs=fat_cs, frac=fat_amp)   
    
    if sim_mode == "mean":
        fat = [np.mean(gp, axis=0) for gp in groupby(fat, window)]
    elif sim_mode == "mid_point":
        if start is None:
            start=(int(window / 2) - 1)
        fat = fat[start:-1:window]
    else:
        raise ValueError("Unknow sim_mode")

    fat = np.reshape(fat, (np.shape(fat)[0], 1, 1, np.shape(fT1)[0],np.shape(fT2)[0], np.shape(att)[0], np.shape(df)[0] ))
    
    water = np.array(water)
    fat = np.array(fat)
    # join water and fat
    print("Build dictionary.")
    keys = list(itertools.product(wT1, wT2, fT1, fT2, att, df))
    values = np.stack(np.broadcast_arrays(water, fat), axis=-1)
    
    values = np.moveaxis(values.reshape(len(values), -1, 2), 0, 1)

    # print("Save dictionary.")
    mrfdict = Dictionary(keys, values)
    # mrfdict.save(dictfile, overwrite=overwrite)
    hdr={"sequence_config":sequence_config,"dict_config":dict_config,"recovery":sequence_config['T_recovery'],"window":window,"sim_mode":sim_mode}
    return mrfdict,hdr,dictfile


def load_sequence_file(fileseq,recovery,min_TR_delay):

    if type(fileseq)==str:
        with open(fileseq, "r") as file:
            seq_config = json.load(file)
    elif type(fileseq)==dict:
        seq_config = fileseq

    TI = seq_config["TI"] * 10 ** -3
    TE = list(np.array(seq_config["TE"]) * 10 ** -3)
    TR = list(np.array(TE)+min_TR_delay)
    FA = seq_config["FA"]
    B1 = seq_config["B1"]
    # B1[:600]=[1.5]*600
    B1 = list(1 * np.array(B1))


    TR[-1] = TR[-1] + recovery

    TR_list = [TI] + TR
    TE_list = [0] + TE
    FA_list = [np.pi] + list(np.array(B1) * FA * np.pi / 180)
    return TR_list,FA_list,TE_list

# def generate_dictionaries(sequence_file,reco,min_TR_delay,dictconf,dictconf_light,TI=8.32):

#     _,FA_list,TE_list=load_sequence_file(sequence_file,reco,min_TR_delay/1000)
#     seq_config=create_new_seq(FA_list,TE_list,min_TR_delay,TI)

#     dictfile,hdr=generate_epg_dico_T1MRFSS_from_sequence(seq_config,dictconf,FA_list,TE_list,reco,min_TR_delay/1000,TI=TI)
#     dictfile_light,hdr_light=generate_epg_dico_T1MRFSS_from_sequence(seq_config,dictconf_light,FA_list,TE_list,reco,min_TR_delay/1000,TI=TI)

#     dico_full_with_hdr={"hdr":hdr,
#                         "hdr_light":hdr_light,
#                         "dictfile":dictfile,
#                         "dictfile_light":dictfile_light}
    
#     dico_full_name=str.split(dictfile,".dict")[0]+".pkl"
#     with open(dico_full_name,"wb") as file:
#         pickle.dumps(dico_full_with_hdr)

#     return

def generate_ImgSeries_T1MRF_generic(sequence_config,dict_config, maps, sim_mode="mean",start=None,window=None):

    mask_np = np.asarray(maps["mask"])
    idx = np.where(mask_np > 0)
    
    # generate signals
    wT1 = maps["WATER_T1"][idx]
    fT1 = maps["FAT_T1"][idx]
    wT2 = maps["WATER_T2"][idx]
    fT2 = maps["FAT_T2"][idx]
    att = maps["att"][idx]
    df = maps["DF"][idx]
    df = np.asarray(df)/1000
    ff = maps["FF"][idx]


    # TR_delay=sequence_config["dTR"]

    seq = T1MRF_generic(sequence_config)


    water_amp = [1]
    water_cs = [0]
    fat_amp = np.array(dict_config["fat_amp"])
    fat_cs = dict_config["fat_cshift"]
    fat_cs = [- value / 1000 for value in fat_cs]  # temp

    # other options
    if window is None:
        window = dict_config["window_size"]


    # if start is None:
    #     dictfile = prefix_dico  +"_TR{}_reco{}.dict".format(str(TR_delay),str(sequence_config['T_recovery']))
    # else:
    #     dictfile = prefix_dico + "_TR{}_reco{}_start{}.dict".format(str(TR_delay),str(sequence_config['T_recovery']),start)

    # if dest is not None:
    #     dictfile = str(pathlib.Path(dest) / pathlib.Path(dictfile).name)
    # print("Generating dictionary {}".format(dictfile))

    # water
    print("Generate water signals.")
    water = seq(T1=wT1, T2=wT2, att=att, g=df, cs=water_cs, frac=water_amp)
    
    if sim_mode == "mean":
        water = [np.mean(gp, axis=0) for gp in groupby(water, window)]
    elif sim_mode == "mid_point":
        if start is None:
            start=(int(window / 2) - 1)

        water = water[start:-1:window]
    else:
        raise ValueError("Unknow sim_mode")

    # fat
    print("Generate fat signals.")
    # eval = "dot(signal, amps)"
    # args = {"amps": fat_amp}
    # merge df and fat_cs df to dict
    fat = seq(T1=fT1, T2=fT2, att=att, g=df, cs=fat_cs, frac=fat_amp)#, eval=eval, args=args)

    if sim_mode == "mean":
        fat = [np.mean(gp, axis=0) for gp in groupby(fat, window)]
    elif sim_mode == "mid_point":
        if start is None:
            start=(int(window / 2) - 1)
        fat = fat[start:-1:window]
    else:
        raise ValueError("Unknow sim_mode")

    water = np.array(water)
    fat = np.array(fat)
    
    signal = (1-np.asarray(ff))*water + np.asarray(ff)*fat
    
    # # join water and fat
    # print("Build dictionary.")
    # keys = list(itertools.product(wT1, wT2, fT1, fT2, att, df))
    # values = np.stack(np.broadcast_arrays(water, fat), axis=-1)
    # values = np.moveaxis(values.reshape(len(values), -1, 2), 0, 1)

    # # print("Save dictionary.")
    # mrfdict = Dictionary(keys, values)
    # # mrfdict.save(dictfile, overwrite=overwrite)
    # hdr={"sequence_config":sequence_config,"dict_config":dict_config,"recovery":sequence_config['T_recovery'],"initial_repetitions":sequence_config['nrep'],"window":window,"sim_mode":sim_mode}
    imgseries = np.zeros((signal.shape[0], maps["mask"].shape[0], maps["mask"].shape[1]), dtype=np.complex64)
    imgseries = imgseries.reshape(signal.shape[0], -1)
    imgseries[:, mask_np.flatten() > 0] = signal
    imgseries = imgseries.reshape(signal.shape[0], mask_np.shape[0], mask_np.shape[1])
    
    return imgseries


def make_numPhantom(DirName, flagSave=False, paramPhantom=None):      
    
    if paramPhantom is None:
        paramPhantom = {'WATER_T1': [1000, 1800],
                        'WATER_T2': [20, 40],
                        'FAT_T1': [433],
                        'FAT_T2': [113],
                        'FF': [0, 1],
                        'DF': [-100, 100],
                        'att': [0.3, 1.3]
                        }
    mask = io.read(os.path.join(DirName,'roi.mhd'))
    mask = mask[:,:,2]
    val = np.unique(mask)
    val = np.delete(val, 0) 
    
    
    # Dimensions de l'image
    Resol_Img = mask.shape

    # Création de MM
    MM = np.zeros(np.shape(mask))
    MM[mask > 0] = 1

    
    # Génération des cartes Df et B1
    X = np.sqrt(np.linspace(0, 1, Resol_Img[0]))[:, None]  # Colonne
    Y = np.sqrt(np.linspace(0, 1, Resol_Img[1]))[None, :]  # Ligne
    
    DF = ((X @ Y) * (paramPhantom['DF'][1] - paramPhantom['DF'][0]) + paramPhantom['DF'][0]) * MM
    att = ((X @ Y) * (paramPhantom['att'][1] - paramPhantom['att'][0]) + paramPhantom['att'][0])
    att = np.flip(att, axis=0) * MM  # Flip vertical

    # Initialisation de la carte des paramètres
    paramMap = {
        "WATER_T1": np.zeros(np.shape(mask), dtype=float),
        "WATER_T2": np.zeros(np.shape(mask), dtype=float),
        "FAT_T1": np.zeros(np.shape(mask), dtype=float),
        "FAT_T2": np.zeros(np.shape(mask), dtype=float),
        "FF": np.zeros(np.shape(mask), dtype=float),
        "att": att,
        "DF": DF, 
        "mask": MM
    }

    # Attribution des valeurs spécifiques aux tissus
    paramMap["WATER_T1"][(mask == val[0]) | (mask == val[1]) | (mask == val[2])] = 1200
    paramMap["WATER_T2"][(mask == val[0]) | (mask == val[1]) | (mask == val[2])] = 30
    paramMap["FAT_T1"][(mask == val[0]) | (mask == val[1]) | (mask == val[2])] = paramPhantom['FAT_T1']
    paramMap["FAT_T2"][(mask == val[0]) | (mask == val[1]) | (mask == val[2])] = paramPhantom['FAT_T2']
    
    paramMap["FF"][(mask == val[0]) | (mask == val[1]) | (mask == val[2])] = 1

    # Attribution aléatoire pour les autres tissus
    for i in range(3, len(val)):
        paramMap["WATER_T1"][mask == val[i]] = np.random.uniform(paramPhantom['WATER_T1'][0], paramPhantom['WATER_T1'][1])
        paramMap["WATER_T2"][mask == val[i]] = np.random.uniform(paramPhantom['WATER_T2'][0], paramPhantom['WATER_T2'][1])
        paramMap["FAT_T1"][mask == val[i]] = paramPhantom['FAT_T1']
        paramMap["FAT_T2"][mask == val[i]] = paramPhantom['FAT_T2']
        paramMap["FF"][mask == val[i]] = np.random.uniform(paramPhantom['FF'][0], paramPhantom['FF'][1])

    # Contraintes sur les valeurs des cartes
    paramMap["WATER_T1"][paramMap["WATER_T1"] < 0] = 0
    paramMap["WATER_T2"][paramMap["WATER_T2"] < 0] = 0
    paramMap["FF"][paramMap["FF"] < 0] = 0
    paramMap["FF"][paramMap["FF"] > 1] = 1
    paramMap["att"][paramMap["att"] > 1] = 1
    
    if flagSave:
        num_file = 1
        while os.path.exists(os.path.join(DirName, f"Phantom{num_file}")):
            num_file += 1

        save_dir = os.path.join(DirName, f"Phantom{num_file}")
        os.makedirs(save_dir, exist_ok=True)

        pickle_file = os.path.join(save_dir, "paramMap.pkl")
        with open(pickle_file, "wb") as f:
            pickle.dump(paramMap, f)

        print(f"paramMap saved in {pickle_file}")
        paramPhantom_serializable = convert_ndarray_to_list(paramPhantom)
        with open(os.path.join(save_dir, 'paramPhantom.json'), "w") as f:
            json.dump(paramPhantom_serializable, f)
        
        # Création de la figure
        fig, axes = plt.subplots(2, 4, figsize=(12, 10))  # 2x2 subplots
        titles = ["T1_water (ms)", "T2_water (ms)", "T1_fat (ms)", "T2_fat (ms)", "FF", "B1", "Df (Hz)", "mask"]
        keys = ["WATER_T1", "WATER_T2", "FAT_T1", "FAT_T2", "FF", "att", "DF", "mask"]

        for ax, key, title in zip(axes.flat, keys, titles):
            im = ax.imshow(np.transpose(paramMap[key], axes=(1, 0)), cmap="viridis", origin="upper")
            ax.set_title(title)
            ax.axis("off")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        # axes.flat[-1].set_visible(False)
        
        plt.tight_layout()

        # Sauvegarde en JPG
        image_file = os.path.join(save_dir, "paramMap.jpg")
        plt.savefig(image_file, dpi=300)
        plt.close()

        print(f"Figure saved in {image_file}")
        
    return paramMap

def convert_ndarray_to_list(d):
    for key, value in d.items():
        if isinstance(value, np.ndarray):
            d[key] = value.tolist()
        elif isinstance(value, list):  # Pour les listes de listes ou d'autres types
            d[key] = [v.tolist() if isinstance(v, np.ndarray) else v for v in value]
    return d 