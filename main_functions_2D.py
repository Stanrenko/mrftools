from utils_mrf import *
from utils_simu import *
from trajectory import Radial
import finufft
import SimpleITK as sitk
import argparse
from PIL import Image
import sys
import json

DEFAULT_OPTIM_CONFIG="./config/config_build_maps.json"

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
        coil_sensitivity[sl]=finufft.nufft2d1(traj_all[:, 0], traj_all[:, 1], kdata_for_sensi[sl].reshape(nb_channels,-1), image_size)
    
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



def check_dico(dico_hdr,file_seqParams):
    '''
    checks dico echo spacing against acquisition sequence echo spacing
    inputs:
    dico_hdr - dictionary containing dictionary parameters
    file_seqParams - file containing acquisition sequence parameters (.pkl)
    outputs: 
    '''
    file = open(file_seqParams, "rb")
    dico_seqParams = pickle.load(file)
    file.close()

    sequence_config=dico_hdr["sequence_config"]
    TR_delay=np.round(sequence_config["TR"][0]-sequence_config["TE"][0],2)

    if np.abs(dico_seqParams["dTR"]-TR_delay)>0.01:
        raise ValueError("Echo spacing from the dictionary ({} ms) does not match sequence echo spacing ({} ms)".format(dico_seqParams["dTR"],TR_delay))
    
    else:
        print("Dictionary OK: Echo spacing from the dictionary ({} ms) close to sequence echo spacing ({} ms)".format(dico_seqParams["dTR"],TR_delay))



def build_maps(volumes_all_slices,masks_all_slices,dictfile,dictfile_light,file_config=DEFAULT_OPTIM_CONFIG):
    '''
    builds MRF maps using bi-component dictionary matching (Slioussarenko et al. MRM 2024)
    inputs:
    volumes_all_slices - time serie of undersampled volumes size ntimesteps x nb_slices x npoint/2 x npoint/2 (numpy array)
    masks_all_slices - mask of size nb_slices x npoint/2 x npoint/2 (numpy array)
    dictfile - full dictionary for pattern matching (.dict)
    dictfile_light - coarse dictionary for preliminary clustering (.dict)
    file_config - optimization options

    outputs:
    all_maps: tuple containing for all iterations 
            (maps - dictionary with parameter maps for all keys
             mask - numpy array
             cost map (OPTIONAL)
             phase map - numpy array (OPTIONAL)
             proton density map - numpy array (OPTIONAL)
             matched_signals - numpy array  (OPTIONAL))

    '''
    with open(file_config,"rb") as file:
        optim_config=json.load(file)

    return_cost=optim_config["return_cost"]
    optimizer = SimpleDictSearch(mask=masks_all_slices, split=optim_config["split"], pca=True,
                                             threshold_pca=optim_config["pca"],dictfile_light=dictfile_light,threshold_ff=0.9,return_cost=return_cost)
            
    all_maps=optimizer.search_patterns_test_multi_2_steps_dico(dictfile,volumes_all_slices)
    
    return all_maps


    
def save_maps(all_maps,file_seqParams,keys = ["ff","wT1","attB1","df"]):
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




def generate_dictionaries(sequence_file,reco,min_TR_delay,dictconf,dictconf_light,TI=8.32):
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
    _,FA_list,TE_list=load_sequence_file(sequence_file,reco,min_TR_delay/1000)
    seq_config=create_new_seq(FA_list,TE_list,min_TR_delay/1000,TI)

    dictfile,hdr=generate_epg_dico_T1MRFSS_from_sequence(seq_config,dictconf,reco)
    dictfile_light,hdr_light=generate_epg_dico_T1MRFSS_from_sequence(seq_config,dictconf_light,reco)
    
    dico_full_with_hdr={"hdr":hdr,
                        "hdr_light":hdr_light,
                        "dictfile":dictfile,
                        "dictfile_light":dictfile_light}
    
    dico_full_name=str.split(dictfile,".dict")[0]+".pkl"
    with open(dico_full_name,"wb") as file:
        pickle.dump(dico_full_with_hdr,file)

    return

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    
    funcs = {"extract_data": extract_data,
             "calculate_sensi":calculate_sensitivity_map,
             "build_volumes":build_volumes,
             "build_masks":build_masks,
             "build_maps":build_maps,
             "generate_dico":generate_dictionaries
             }

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")

    parser_extractdata = subparsers.add_parser('extract_data')
    parser_extractdata.add_argument('--filename', type=str)

    parser_sensi = subparsers.add_parser('calculate_sensi')
    parser_sensi.add_argument('--filekdata', type=str,nargs='?', const="data/kdata.npy", default="data/kdata.npy")

    parser_volumes = subparsers.add_parser('build_volumes')
    parser_volumes.add_argument('--filekdata', type=str,nargs='?', const="data/kdata.npy", default="data/kdata.npy")
    parser_volumes.add_argument('--fileb1',type=str,nargs='?', const="data/b1.npy", default="data/b1.npy")

    parser_masks = subparsers.add_parser('build_masks')
    parser_masks.add_argument('--filekdata', type=str,nargs='?', const="data/kdata.npy", default="data/kdata.npy")
    parser_masks.add_argument('--fileb1', type=str,nargs='?', const="data/b1.npy", default="data/b1.npy")

    parser_maps = subparsers.add_parser('build_maps')
    parser_maps.add_argument('--filevolumes', type=str,nargs='?', const="data/volumes.npy", default="data/volumes.npy")
    parser_maps.add_argument('--filemasks', type=str,nargs='?', const="data/masks.npy", default="data/masks.npy")
    parser_maps.add_argument('--fileseq', type=str,nargs='?', const="data/dico_seqParams.pkl", default="data/dico_seqParams.pkl")
    parser_maps.add_argument('--dictfiles', type=str,nargs='?', const="dico/mrf_dictconf_Dico2_Invivo_TR1.11_reco5.0.pkl", default="dico/mrf_dictconf_Dico2_Invivo_TR1.11_reco5.0.pkl")

    parser_dico = subparsers.add_parser('generate_dico')
    parser_dico.add_argument('--sequencefile', type=str,nargs='?', const="dico/mrf_sequence_adjusted.json", default="dico/mrf_sequence_adjusted.json")
    parser_dico.add_argument('--dictconf', type=str,nargs='?', const="dico/mrf_dictconf_Dico2_Invivo.json", default="dico/mrf_dictconf_Dico2_Invivo.json")
    parser_dico.add_argument('--dictconflight', type=str,nargs='?', const="dico/mrf_dictconf_Dico2_Invivo_light_for_matching.json", default="dico/mrf_dictconf_Dico2_Invivo_light_for_matching.json")
    parser_dico.add_argument('--reco', type=float,nargs='?', const=5.0, default=5.0) # seconds
    parser_dico.add_argument('--echospacing', type=float) # ms
    parser_dico.add_argument('--TI', type=float,nargs='?', const=8.32, default=8.32) # ms 
    
    
    
    args = parser.parse_args()

    chosen_func=funcs[args.command]

    if args.command=="extract_data":    

        filename=args.filename
        kdata,dico_seqParams=chosen_func(filename)

        path, _ = os.path.split(filename)
        print(path)

        file_kdata=os.path.join(path,"kdata.npy")
        file_seqParams=os.path.join(path,"dico_seqParams.pkl")
        np.save(file_kdata,kdata)

        file = open(file_seqParams, "wb")
        pickle.dump(dico_seqParams, file)
        file.close()


    elif args.command=="calculate_sensi":  
        file_kdata= args.filekdata
        kdata=np.load(file_kdata)
        b1=chosen_func(kdata)

        path, _ = os.path.split(file_kdata)
        print(path)

        file_b1=os.path.join(path,"b1.npy")
        file_b1_image=os.path.join(path,"b1.jpg")
        np.save(file_b1,b1)


        sl = int(b1.shape[0]/2)

        list_images=list(np.abs(b1[sl,:,:]))
        plot_image_grid(list_images,(6,6),title="Sensivity map for slice {}".format(sl),save_file=file_b1_image)


    elif args.command=="build_volumes":  
        file_kdata= args.filekdata
        file_b1=args.fileb1
        kdata=np.load(file_kdata)
        b1=np.load(file_b1)
        volumes=chosen_func(kdata,b1)

        path, _ = os.path.split(file_kdata)
        print(path)

        file_volumes=os.path.join(path,"volumes.npy")
        file_volumes_gif=os.path.join(path,"volumes.gif")

        np.save(file_volumes,volumes)
        gif=[]
        sl=int(volumes.shape[1]/2)
        volume_for_gif = np.abs(volumes[:,sl])
        for i in range(volume_for_gif.shape[0]):
            img = Image.fromarray(np.uint8(volume_for_gif[i]/np.max(volume_for_gif[i])*255), 'L')
            img=img.convert("P")
            gif.append(img)

                
        gif[0].save(file_volumes_gif,save_all=True, append_images=gif[1:], optimize=False, duration=100, loop=0)
    
    elif args.command=="build_masks":

        file_kdata= args.filekdata
        file_b1=args.fileb1
        kdata=np.load(file_kdata)
        b1=np.load(file_b1)
        masks=chosen_func(kdata,b1)

        path, _ = os.path.split(file_kdata)
        print(path)

        file_masks=os.path.join(path,"masks.npy")
        file_masks_gif=os.path.join(path,"masks.gif")

        np.save(file_masks,masks)
        gif=[]
        volume_for_gif = np.abs(masks)
        for i in range(volume_for_gif.shape[0]):
            img = Image.fromarray(np.uint8(volume_for_gif[i]/np.max(volume_for_gif[i])*255), 'L')
            img=img.convert("P")
            gif.append(img)

                
        gif[0].save(file_masks_gif,save_all=True, append_images=gif[1:], optimize=False, duration=100, loop=0)
    
    elif args.command=="build_maps":

        file_volumes= args.filevolumes
        file_masks=args.filemasks
        file_seq=args.fileseq
        dictfiles=args.dictfiles
        print(dictfiles)
        volumes=np.load(file_volumes)
        masks=np.load(file_masks)
        
        with open(dictfiles,"rb") as file:
            dico_full_with_hdr=pickle.load(file)
        
        dictfile=dico_full_with_hdr["dictfile"]
        dictfile_light=dico_full_with_hdr["dictfile_light"]
        dico_hdr=dico_full_with_hdr["hdr"]
        
        check_dico(dico_hdr,file_seq)
        all_maps=build_maps(volumes,masks,dictfile,dictfile_light)
        save_maps(all_maps,file_seq)

    elif args.command=="generate_dico":

        sequence_file= args.sequencefile
        reco=args.reco
        min_TR_delay=args.echospacing
        dictconf=args.dictconf
        dictconf_light=args.dictconflight
        TI=args.TI


        generate_dictionaries(sequence_file,reco,min_TR_delay,dictconf,dictconf_light,TI=8.32)

    else:
        raise("Value Error : Unknown Function")

