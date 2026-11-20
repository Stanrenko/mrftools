import json
import pathlib
import numpy as np
import itertools
import epgpy as epg
import pickle
from epgpy import common, operators, statematrix, utils

from .utils_mrf import groupby
from .dictmodel import Dictionary


from epgpy.sequence import Sequence, Variable, operators


class T1MRF_generic:
    def __init__(self, SEQ_CONFIG):
        """ build sequence """
        self.seqlen = len(SEQ_CONFIG['TE'])
        self.READOUT = SEQ_CONFIG['readout']

        if "TI" in SEQ_CONFIG:
            self.TI = SEQ_CONFIG['TI']
            self.inversion = epg.T(180, 0) 
        else:
            self.TI=None
            self.inversion = None

        self.TE = SEQ_CONFIG['TE']
        self.FA = SEQ_CONFIG['FA']*SEQ_CONFIG['B1']
        TR = [i + SEQ_CONFIG['dTR'] for i in self.TE]
        self.TR = TR
        
        
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
        elif SEQ_CONFIG['readout'] == 'pSSFP_generic': 
            self.spl = epg.S(1)
            self.PHI = [SEQ_CONFIG['PHI'][i] * i * (i + 1) / 2 for i in range(len(self.TE))]
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
        if self.TI is not None:
            self.rlx0 = epg.E(self.TI, T1, T2, duration=True) # inversion delay
        self.rlx1 = [[epg.E(self.TE[i], T1, T2, cs[j] + g, duration=True) for i in range(len(self.FA))]  for j in range(len(cs))] 
        self.rlx2 = [[epg.E(self.TR[i] - self.TE[i], T1, T2, cs[j] + g, duration=True) for i in range(len(self.FA))] for j in range(len(cs))] 
        self.rlx3 = epg.E(self.T_recovery, T1, T2, duration=True) # recovery

        if self.TI is not None:    
            seq = [[self.init[j]] + [self.inversion] + [self.rlx0] + [[self.rf[i], self.rlx1[j][i], self.adc[i], self.rlx2[j][i], self.spl] for i in range(self.seqlen)] for j in range(len(cs))] 
        else:
            seq = [[self.init[j]]  + [[self.rf[i], self.rlx1[j][i], self.adc[i], self.rlx2[j][i], self.spl] for i in range(self.seqlen)] for j in range(len(cs))] 


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


class T1MRFSS_NoInv:
    def __init__(self, FA, TE, TR, B1,T_recovery,nrep,rep=None):
        """ build sequence """
        seqlen = len(TE)
        self.TR=TR
        #self.inversion = epg.T(180, 0) # perfect inversion
        self.T_recovery=T_recovery
        self.nrep=nrep
        self.rep=rep
        seq=[]
        for r in range(nrep):
            curr_seq = [epg.Offset(0.000001)]
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

    def __call__(self, T1, T2, g, att, calc_deriv=False,**kwargs):
        """ simulate sequence """
        seq=[]
        rep=self.rep
        #print(self._seq)
        for r in range(self.nrep):
            curr_seq=self._seq[r*self.len_rep:(r+1)*(self.len_rep)]
            curr_seq=[epg.modify(curr_seq, T1=T1, T2=T2, att=att, g=g,calc_deriv=calc_deriv)]
            seq.extend(curr_seq)
        #seq = [self.inversion, epg.modify(self._seq, T1=T1, T2=T2, att=att, g=g,calc_deriv=calc_deriv)]
        if not(calc_deriv):
            result=np.asarray(epg.simulate(seq, **kwargs))
            if rep is None:#returning all repetitions
                return result
            else:#returning only the rep
                result = result.reshape((self.nrep, -1) + result.shape[1:])[rep]
                return result

        else:
            return epg.simulate(seq,calc_deriv=calc_deriv, **kwargs)




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
    hdr={"sequence_config":sequence_config,"dict_config":dict_config,"recovery":recovery,"initial_repetitions":rep,"window":window,"sim_mode":sim_mode,"param_names":("wT1","fT1","attB1","df")}
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

