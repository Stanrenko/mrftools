
import finufft
import SimpleITK as sitk
import json
import pathlib

from .utils_mrf import *
from .utils_simu import *
from .trajectory import Radial


def extract_data(filename,dens_adj=True):
    ''' 
    Extracts data from raw data file
    inputs:
    filename - Siemens .dat file (str)
    dens_adj - whether to apply radial density adjustment (bool)
    outputs:
    data - numpy array containing k-space data nb_slices x nb_channels x nb_segments x npoint
    dico_seqParams - dictionary containing sequence parameters
    '''
    
    data,dico_seqParams=read_rawdata_2D(filename)

    if dens_adj:
        
        print("Performing radial density adjustment")
        npoint = data.shape[-1]
        density = np.abs(np.linspace(-1, 1, npoint))
        density = np.abs(np.linspace(-1, 1, npoint))
        density = np.expand_dims(density, tuple(range(data.ndim - 1)))
        data*=density

    print(dico_seqParams)

    return data,dico_seqParams
    


def calculate_sensitivity_map(kdata,res=16,hanning_filter=True,density_adj=False):
    '''
    Calculates coil sensitivity maps
    inputs:
    kdata - numpy array containing k-space data nb_slices x nb_channels x nb_segments x npoint
    res - k-space data cutoff (int)
    hanning_filter - bool
    density_adj - bool
    outputs:
    b1 - coil sensitivity map of size nb_slices x nb_channels x npoint/2 x npoint/2
    '''
    nb_allspokes=kdata.shape[-2]
    npoint=kdata.shape[-1]

    image_size=(int(npoint/2),int(npoint/2))

    trajectory=Radial(total_nspokes=nb_allspokes,npoint=npoint)
    traj_all = trajectory.get_traj().astype("float32")
    traj_all=traj_all.reshape(-1,traj_all.shape[-1])
    npoint = kdata.shape[-1]
    center_res = int(npoint / 2)

    nb_channels=kdata.shape[1]
    nb_slices=kdata.shape[0]

    if density_adj:
        density = np.abs(np.linspace(-1, 1, npoint))
        density = np.expand_dims(density, tuple(range(kdata.ndim - 1)))
        kdata*=density

    kdata_for_sensi = np.zeros_like(kdata)

    if hanning_filter:
        if kdata.ndim==3:#Tried :: syntax but somehow it introduces errors in the allocation
            kdata_for_sensi[:,:, (center_res - int(res / 2)):(center_res + int(res / 2))]=kdata[:,:,
                                                                                       (center_res - int(res / 2)):(
                                                                                                   center_res + int(
                                                                                               res / 2))]*np.expand_dims(np.hanning(2*int(res/2)),axis=(0,1))
        else:
            kdata_for_sensi[:, :,:, (center_res - int(res / 2)):(center_res + int(res / 2))] = kdata[:, :,:,
                                                                                             (center_res - int(res / 2)):(
                                                                                                     center_res + int(
                                                                                                 res / 2))]*np.expand_dims(np.hanning(2*int(res/2)),axis=(0,1,2))


    else:
        if kdata.ndim==3:#Tried :: syntax but somehow it introduces errors in the allocation
            kdata_for_sensi[:,:, (center_res - int(res / 2)):(center_res + int(res / 2))] = kdata[:,:,
                                                                                       (center_res - int(res / 2)):(
                                                                                                   center_res + int(
                                                                                               res / 2))]
        else:
            kdata_for_sensi[:, :,:, (center_res - int(res / 2)):(center_res + int(res / 2))] = kdata[:, :,:,
                                                                                             (center_res - int(res / 2)):(
                                                                                                     center_res + int(
                                                                                                 res / 2))]

    
    coil_sensitivity=np.zeros((nb_slices,nb_channels,)+image_size,dtype=kdata.dtype)
    print(kdata_for_sensi.shape)

    for sl in tqdm(range(nb_slices)):
        coil_sensitivity[sl]=finufft.nufft2d1(
            np.ascontiguousarray(traj_all[:, 0]), 
            np.ascontiguousarray(traj_all[:, 1]),
            np.ascontiguousarray(kdata_for_sensi[sl].reshape(nb_channels,-1)), 
            image_size)
    
    #coil_sensitivity=coil_sensitivity.reshape(*kdata.shape[:-2],*image_size)
    print(coil_sensitivity.shape)

    if coil_sensitivity.ndim==3:
        print("Ndim 3)")
        b1 = coil_sensitivity / np.linalg.norm(coil_sensitivity, axis=0)
        #b1 = b1 / np.max(np.abs(b1.flatten()))
    else:#first dimension contains slices
        print("Ndim > 3")
        b1=coil_sensitivity.copy()
        for i in range(coil_sensitivity.shape[0]):
            b1[i]=coil_sensitivity[i] / np.linalg.norm(coil_sensitivity[i], axis=0)
            #b1[i]=b1[i] / np.max(np.abs(b1[i].flatten()))
    return b1






def build_volumes(kdata,b1_all_slices,ntimesteps=175):
    '''
    build time serie of ntimesteps undersampled volumes for all slices
    inputs:
    kdata - numpy array containing k-space data nb_slices x nb_channels x nb_segments x npoint
    b1_all_slices - coil sensitivity map of size nb_slices x nb_channels x npoint/2 x npoint/2
    ntimesteps - number of undersampled image in the serie
    outputs:
    volumes_all_slices - time serie of undersampled volumes size ntimesteps x nb_slices x npoint/2 x npoint/2 (numpy array)

    '''
    nb_slices,nb_channels,nb_allspokes,npoint=kdata.shape
    image_size=(int(npoint/2),int(npoint/2))

    radial_traj=Radial(total_nspokes=nb_allspokes,npoint=npoint)
    volumes_all_slices=[]
    for sl in range(0,nb_slices):


        print("Processing slice {} out of {}".format(sl+1,kdata.shape[0]))
        kdata_all_channels=kdata[sl,:,:,:]
        b1=b1_all_slices[sl]

        volumes=simulate_radial_undersampled_images_multi(kdata_all_channels,radial_traj,image_size,b1=b1,density_adj=False,ntimesteps=ntimesteps)
        print(volumes.shape)

        volumes_all_slices.append(volumes)
       
        
    volumes_all_slices=np.array(volumes_all_slices)
    volumes_all_slices=np.moveaxis(volumes_all_slices,0,1)
    return volumes_all_slices


def build_masks(kdata,b1_all_slices,threshold_factor=1/25):
    '''
    builds mask for all slices
    inputs:
    kdata - numpy array containing k-space data nb_slices x nb_channels x nb_segments x npoint
    b1_all_slices - coil sensitivity map of size nb_slices x nb_channels x npoint/2 x npoint/2
    threshold_factor - threshold for histogram cutoff
    outputs:
    mask - mask of size nb_slices x npoint/2 x npoint/2 (numpy array)

    '''

    nb_slices,nb_channels,nb_allspokes,npoint=kdata.shape
    image_size=(int(npoint/2),int(npoint/2))

    radial_traj=Radial(total_nspokes=nb_allspokes,npoint=npoint)
    masks_all_slices=[]
    for sl in range(0,nb_slices):
        mask=build_mask_single_image_multichannel(kdata[sl],radial_traj,image_size,b1=b1_all_slices[sl],density_adj=False,threshold_factor=threshold_factor)
        masks_all_slices.append(mask)

    masks_all_slices=np.array(masks_all_slices)

    return(masks_all_slices)



def check_dico(dico_hdr, seqParams):
    '''
    checks dico echo spacing against acquisition sequence echo spacing
    inputs:
    dico_hdr - dictionary containing dictionary parameters
    file_seqParams - file containing acquisition sequence parameters (.pkl)
    outputs: 
    '''
    if isinstance(seqParams, str):
        with open(seqParams, 'rb') as fp:
            dico_seqParams = pickle.load(fp)
    elif isinstance(seqParams, dict):
        dico_seqParams = seqParams
    else:
        raise ValueError(seqParams)

    sequence_config=dico_hdr["sequence_config"]
    TR_delay=np.round(sequence_config["TR"][0]-sequence_config["TE"][0],2)

    if np.abs(dico_seqParams["dTR"]-TR_delay)>0.01:
        raise ValueError("Echo spacing from the dictionary ({} ms) does not match sequence echo spacing ({} ms)".format(dico_seqParams["dTR"],TR_delay))
    
    else:
        print("Dictionary OK: Echo spacing from the dictionary ({} ms) close to sequence echo spacing ({} ms)".format(dico_seqParams["dTR"],TR_delay))



def build_maps(volumes_all_slices,masks_all_slices,dictfile,dictfile_light,useGPU=True,split=100,return_cost=False,pca=6):
    '''
    builds MRF maps using bi-component dictionary matching (Slioussarenko et al. MRM 2024)
    inputs:
    volumes_all_slices - time serie of undersampled volumes size ntimesteps x nb_slices x npoint/2 x npoint/2 (numpy array)
    masks_all_slices - mask of size nb_slices x npoint/2 x npoint/2 (numpy array)
    dictfile - full dictionary for pattern matching (.dict)
    dictfile_light - coarse dictionary for preliminary clustering (.dict)
    file_config - optimization options
    useGPU - wheter to use GPU
    split - signal batch count for memory management (int)
    return_cost - whether to return additional maps (e.g. proton density, phase and cost)
    pca - number of temporal pca components retained (int)

    outputs:
    all_maps: tuple containing for all iterations 
            (maps - dictionary with parameter maps for all keys
             mask - numpy array
             cost map (OPTIONAL)
             phase map - numpy array (OPTIONAL)
             proton density map - numpy array (OPTIONAL)
             matched_signals - numpy array  (OPTIONAL))

    '''

    try:
        import cupy
    except:
        print("Could not import cupy - not using gpu")
        useGPU=False

    optimizer = SimpleDictSearch(mask=masks_all_slices, split=split, pca=True,
                                             threshold_pca=pca,dictfile_light=dictfile_light,threshold_ff=0.9,return_cost=return_cost,useGPU_dictsearch=useGPU)
            
    all_maps=optimizer.search_patterns_test_multi_2_steps_dico(dictfile,volumes_all_slices)
    
    return all_maps


    
def save_maps(all_maps, file_seqParams, keys = ["ff","wT1","attB1","df"]):
    '''
    generate the map images in .mha format for all params and stores the optimisation results in a .pkl
    inputs:
    all_maps - tuple containing for all iterations 
            (maps - dictionary with parameter maps for all keys
             mask - numpy array
             cost map (OPTIONAL)
             phase map - numpy array (OPTIONAL)
             proton density map - numpy array (OPTIONAL)
             matched_signals - numpy array  (OPTIONAL))

    file_seqParams - file containing acquisition sequence parameters (.pkl)
    keys - parameters for which to generate the .mha
    
    outputs:
    '''

    file = open(file_seqParams, "rb")
    dico_seqParams = pickle.load(file)
    file.close()

    path, _ = os.path.split(file_seqParams)
    print(path)


    file_map=os.path.join(path,"maps.pkl")
    with open(file_map,"wb") as file:
        pickle.dump(all_maps, file)

    for k in tqdm(keys) :
                
        map_rebuilt = all_maps[0][0]
        mask = all_maps[0][1]
        map_all_slices = makevol(map_rebuilt[k], mask > 0)            
        map_all_slices,geom=convertArrayToImageHelper(dico_seqParams,map_all_slices)
        curr_volume=sitk.GetImageFromArray(map_all_slices)
        
        curr_volume.SetSpacing(geom["spacing"])
        curr_volume.SetOrigin(geom["origin"])
        
        file_map=os.path.join(path,"{}_map.mha".format(k))
        sitk.WriteImage(curr_volume,file_map)




def generate_dictionaries(sequence_file,reco,min_TR_delay,dictconf,dictconf_light,TI=8.32, dest=None):
    '''
    Generates dictionaries from sequence and dico configuration files
    inputs:
    sequence_file - sequence acquistion parameters (.json)
    reco - waiting time at the end of each MRF repetition (seconds)
    min_TR_delay - waiting time from echo time to next RF pulse (ms)
    dictconf - dictionary parameter grid for full dictionary (.json)
    dictconf_light - dictionary parameter grid for light dictionary (.json)
    TI - inversion time (ms)

    outputs:
    saves the dictionaries paths and headers in a .pkl file
    '''
    sequence_file = str(sequence_file)
    dictconf = str(dictconf)
    dictconf_light = str(dictconf_light)

    _,FA_list,TE_list=load_sequence_file(sequence_file,reco,min_TR_delay/1000)
    seq_config=create_new_seq(FA_list,TE_list,min_TR_delay/1000,TI)

    dictfile,hdr=generate_epg_dico_T1MRFSS_from_sequence(seq_config,dictconf,reco, dest=dest)
    dictfile_light,hdr_light=generate_epg_dico_T1MRFSS_from_sequence(seq_config,dictconf_light,reco, dest=dest)
    
    dico_full_with_hdr={"hdr":hdr,
                        "hdr_light":hdr_light,
                        "dictfile":dictfile,
                        "dictfile_light":dictfile_light}
    
    dico_full_name = str.split(dictfile,".dict")[0]+".pkl"
    with open(dico_full_name,"wb") as file:
        pickle.dump(dico_full_with_hdr,file)

    return
