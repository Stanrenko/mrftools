import numpy as np
from epgpy.sequence import Sequence, Variable, operators
import epgpy as epg
import mrftools.config as cc
import itertools
import os
import time

SEQ_CONFIG = cc.SEQ_CONFIG4

att = Variable("att")
T1 = Variable("T1")
T2 = Variable("T2")
g = Variable("g")
cs = Variable("cs")
frac = Variable("frac")

seqlen = len(SEQ_CONFIG['TE'])
READOUT = SEQ_CONFIG['readout']
TI = SEQ_CONFIG['TI']
TE = SEQ_CONFIG['TE']
FA = SEQ_CONFIG['FA']*SEQ_CONFIG['B1']
TR = [i + SEQ_CONFIG['dTR'] for i in TE]
T_recovery=SEQ_CONFIG['T_recovery']



if SEQ_CONFIG['readout'] == 'FLASH_ideal': 
    spl = operators.SPOILER
    PHI = [50 * i * (i + 1) / 2 for i in range(len(TE))]
elif SEQ_CONFIG['readout'] == 'FLASH': 
    spl = operators.S(1)
    PHI = [50 * i * (i + 1) / 2 for i in range(len(TE))]
elif SEQ_CONFIG['readout'] == 'FISP': 
    spl = operators.S(1)
    PHI = np.array([0]*len(TE))
elif SEQ_CONFIG['readout'] == 'pSSFP': 
    spl = operators.S(1)
    PHI = [1 * i * (i + 1) / 2 for i in range(len(TE))]
elif SEQ_CONFIG['readout'] == 'pSSFP_2': 
    spl = operators.S(1)
    PHI = np.array([1 * i * (i + 1) / 2 for i in range(int(np.round(len(TE)/2)))] + [(-1) * i * (i + 1) / 2 for i in range(int(np.round(len(TE)/2)))])            
elif SEQ_CONFIG['readout'] == 'TrueFISP':
    spl = operators.NULL
    PHI = np.array([0, 180] * (len(TE) // 2) + [0] * (len(TE) % 2)) 

# init = [operators.PD(frac[j]) for j in range(len(frac))]
init = operators.PD(frac)
inversion = operators.T(180, 0)
rf = [operators.T(FA[i]*att, PHI[i]) for i in range(seqlen)] 
adc = [operators.Adc(phase=-rf[i].phi, reduce=1) for i in range(seqlen)]
rlx0 = operators.E(TI, T1, T2, duration=True) 
# rlx1 = [[operators.E(TE[i], T1, T2, cs[j] + g, duration=True) for i in range(len(FA))]  for j in range(len(cs))] 
rlx1 = [operators.E(TE[i], T1, T2, cs + g, duration=True) for i in range(len(FA))] 
# rlx2 = [[operators.E(TR[i] - TE[i], T1, T2, cs[j] + g, duration=True) for i in range(len(FA))] for j in range(len(cs))] 
rlx2 = [operators.E(TR[i] - TE[i], T1, T2, cs + g, duration=True) for i in range(len(FA))]
rlx3 = operators.E(T_recovery, T1, T2, duration=True)
        
seq = Sequence([init] + [inversion] + [rlx0] + [[rf[i], rlx1[i], adc[i], rlx2[i], spl] for i in range(seqlen)])
# seq = Sequence([inversion] + [rlx0] + [[rf[i], rlx1[i], adc[i], rlx2[i], spl] for i in range(seqlen)])


DICT_CONFIG = cc.DICT_CONFIG6
wT1 = list(DICT_CONFIG["water_T1"])
fT1 = list(DICT_CONFIG["fat_T1"])
wT2 = list(DICT_CONFIG["water_T2"])
fT2 = list(DICT_CONFIG["fat_T2"])
att = list(DICT_CONFIG["B1_att"])
df = list(DICT_CONFIG["delta_freqs"])
df = [- value / 1000 for value in df]
cs = [0.0]  

combinations = list(itertools.product(wT1, wT2, att, df))
water_T1, water_T2, water_att, water_df = zip(*combinations)

epg.set_array_module('cupy')
start = time.time()
sig = seq.signal(T1=np.array(water_T1[0:100]), T2 = np.array(water_T2[0:100]), att=np.array(water_att[0:100]), g = np.array(water_df[0:100])[:,np.newaxis], cs = [DICT_CONFIG['fat_cshift']], frac = [DICT_CONFIG['fat_amp']], options={"disp":True})
end = time.time()
print(f"Execution time: {end - start: .6f} seconds")
# sig = seq.signal(T1=water_T1[0:9], T2 = water_T2[0:9], att=water_att[0:9], g = water_df[0:9], cs = [0], options={"disp":True})
# sig = seq.signal(T1=[1000,1100], T2 = [40,45], att=[1.0, 0.3], cs = [0, -200], g = [0, -10], options={"disp":True})