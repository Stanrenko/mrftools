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

json_file = 'dicoT2'
key = 'water_T2'

with open(f"../config/config_{json_file}.json", 'r', encoding='utf-8') as fichier:
    dict_config = json.load(fichier)
    
weights = np.array([
    [1, 0],      # Première combinaison
    [0.75, 0.25],# Deuxième combinaison
    [0.5, 0.5]   # Troisième combinaison
])

paramSeq = cc.SEQ_CONFIG_pSSFP4
paramSeq['readout'] = 'pSSFP_generic'

# paramSeq = cc.SEQ_CONFIG2
# paramSeq['readout'] = 'FLASH'

paramSeq['T_recovery'] = 5000
paramSeq['eta'] = 1
paramSeq['nrep'] = 1

# mrfdict,hdr,dictfile = us.generate_epg_dico_T1MRF_generic_from_sequence(paramSeq, dict_config)
mrfdict,hdr,dictfile = us.generate_epg_dico_T1MRF(paramSeq,dict_config,useGPU=True, batch_size=None, dest=None,prefix_dico="{}".format("dico"))



variable = hdr['dict_config'][key]

mrfdict_values_ff = np.tensordot(mrfdict.values, weights.T, axes=([2], [0]))


colormap = cm.get_cmap('plasma', len(variable))
normalize = plt.Normalize(vmin=0, vmax=len(variable) - 1)
plt.style.use('dark_background')

# Création de la figure et des axes
fig, axes = plt.subplots(2, 3, figsize=(18, 12), sharey='row', sharex='col')

# Boucle sur les fractions de graisse
for i in range(mrfdict_values_ff.shape[2]):
    # Axes pour l'amplitude et la phase
    ax_amp = axes[0, i]
    ax_phase = axes[1, i]
    
    # Boucle sur les valeurs de T2
    for j in range(len(variable)):
        # Extraction de la série temporelle complexe
        time_series = mrfdict_values_ff[j, :, i]
        
        # Calcul de l'amplitude et de la phase
        amplitude = np.abs(time_series)
        phase = np.angle(time_series)
        
        # Tracé de l'amplitude
        ax_amp.plot(amplitude, color=colormap(normalize(j)), label=f"{key} = {variable[j]}")
        
        # Tracé de la phase
        ax_phase.plot(phase, color=colormap(normalize(j)))

    # Configuration des axes
    ff_value = weights[i][1] / (weights[i][1] + weights[i][0])
    ax_amp.set_title(f"FF = {ff_value:.2f}")
    ax_amp.set_xlabel("time")
    ax_phase.set_xlabel("time")
    
    if i == 0:
        ax_amp.set_ylabel("Amplitude")
        ax_phase.set_ylabel("Phase (rad)")

# Ajout d'une légende commune
handles, labels = ax_amp.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper right')

# Ajustement de la mise en page
plt.tight_layout()
# plt.savefig(f"../fig/{paramSeq['readout']}_config_{json_file}_FA{paramSeq['FA']}.png", dpi=300)
plt.savefig(f"/mnt/rmn_files/0_Wip/New/1_Methodological_Developments/1_Methodologie_3T/&3_2017_Fingerprinting_FF_T1/2_Codes_Info/python/mrf_dev/fig/{paramSeq['readout']}4_config_{json_file}_FA{paramSeq['FA']}.png", dpi=300)
plt.show()
plt.close()