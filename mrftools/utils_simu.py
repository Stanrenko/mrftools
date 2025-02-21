import json
import pathlib
import numpy as np
import itertools
import epgpy as epg
import pickle

from .utils_mrf import groupby
from .dictmodel import Dictionary


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
    hdr={"sequence_config":sequence_config,"dict_config":dict_config,"recovery":recovery,"initial_repetitions":rep,"window":window,"sim_mode":sim_mode}
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

