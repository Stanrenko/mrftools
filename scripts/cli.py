import argparse
import os, sys
from PIL import Image
import numpy as np
import pathlib
from mrftools.config import DICT_CONFIG,DICT_LIGHT_CONFIG,SEQ_CONFIG
from mrftools.utils_reco import *


ROOT = pathlib.Path(__file__).parent.parent
DATA = ROOT / 'data'
DICT = ROOT / 'mrftools' / 'dico_config'
# CONF = ROOT / 'resources' / 'config'

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    
    funcs = {"extract_data": extract_data,
             "calculate_sensi":calculate_sensitivity_map,
             "build_volumes":build_volumes,
             "build_volumes_singular":build_volumes_singular_iterative,
             "build_masks":build_masks,
             "build_masks_singular":build_mask_from_singular_volume,
             "build_maps":build_maps,
             "generate_dico":generate_dictionaries
             }

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")

    parser_extractdata = subparsers.add_parser('extract_data')
    parser_extractdata.add_argument('--filename', type=str, required=True)

    parser_sensi = subparsers.add_parser('calculate_sensi')
    parser_sensi.add_argument('--filekdata', type=str,nargs='?', const=DATA / "kdata.npy", default=DATA / "kdata.npy")

    parser_volumes = subparsers.add_parser('build_volumes')
    parser_volumes.add_argument('--filekdata', type=str,nargs='?', const=DATA / "kdata.npy", default=DATA / "kdata.npy")
    parser_volumes.add_argument('--fileb1',type=str,nargs='?', const=DATA / "b1.npy", default=DATA / "b1.npy")

    parser_volumes_singular = subparsers.add_parser('build_volumes_singular')
    parser_volumes_singular.add_argument('--filekdata', type=str,nargs='?', const=DATA / "kdata.npy", default=DATA / "kdata.npy")
    parser_volumes_singular.add_argument('--fileb1',type=str,nargs='?', const=DATA / "b1.npy", default=DATA / "b1.npy")
    parser_volumes_singular.add_argument('--dictdir', type=str, default='./mrf_dict')
    parser_volumes_singular.add_argument('--dictfiles', type=str,nargs='?', const="dico_TR1.11_reco5.0.pkl", default="dico_TR1.11_reco5.0.pkl")
    parser_volumes_singular.add_argument('--L0', type=int,nargs='?', const=6, default=6)
    parser_volumes_singular.add_argument('--niter', type=int,nargs='?', const=0, default=0)
    parser_volumes_singular.add_argument('--regularizer', type=str,nargs='?', const=None, default=None)
    parser_volumes_singular.add_argument('--lambd', type=float,nargs='?', const=None, default=None) 
    

    parser_masks = subparsers.add_parser('build_masks')
    parser_masks.add_argument('--filekdata', type=str,nargs='?', const=DATA / "kdata.npy", default=DATA / "kdata.npy")
    parser_masks.add_argument('--fileb1', type=str,nargs='?', const=DATA / "b1.npy", default=DATA / "b1.npy")
    

    parser_masks_singular = subparsers.add_parser('build_masks_singular')
    parser_masks_singular.add_argument('--filevolumes', type=str,nargs='?', const=DATA / "volumes_singular.npy", default=DATA / "volumes_singular.npy")
    parser_masks_singular.add_argument('--l', type=int,nargs='?', const=0, default=0)
    parser_masks_singular.add_argument('--threshold', type=float,nargs='?', const=0.025, default=0.025) 

    parser_maps = subparsers.add_parser('build_maps')
    parser_maps.add_argument('--filevolumes', type=str,nargs='?', const=DATA / "volumes.npy", default=DATA / "volumes.npy")
    parser_maps.add_argument('--filemasks', type=str,nargs='?', const=DATA / "masks.npy", default=DATA / "masks.npy")
    parser_maps.add_argument('--fileseq', type=str,nargs='?', const=DATA / "dico_seqParams.pkl", default=DATA / "dico_seqParams.pkl")
    parser_maps.add_argument('--dictdir', type=str, default='./mrf_dict')
    parser_maps.add_argument('--dictfiles', type=str,nargs='?', const="dico_TR1.11_reco5.0.pkl", default="dico_TR1.11_reco5.0.pkl")
    # parser_maps.add_argument('--optim-config', type=str, default=CONF / "config_build_maps.json")
    parser_maps.add_argument('--pca', type=int,nargs='?', const=6, default=6)
    parser_maps.add_argument('--split', type=int,nargs='?', const=100, default=100)
    parser_maps.add_argument('--useGPU', type=bool,nargs='?', const=True, default=True)
    parser_maps.add_argument('--returncost', type=bool,nargs='?', const=False, default=False)
    parser_maps.add_argument('--volumestype', type=str,nargs='?', const="raw", default="raw")

    parser_dico = subparsers.add_parser('generate_dico')
    parser_dico.add_argument('--dictdir', type=str, default='./mrf_dict')
    parser_dico.add_argument('--sequencefile', type=str,nargs='?', const=None, default=None)
    parser_dico.add_argument('--dictconf', type=str,nargs='?', const=None, default=None)
    parser_dico.add_argument('--dictconflight', type=str,nargs='?', const=None, default=None)
    parser_dico.add_argument('--reco', type=float,nargs='?', const=5.0, default=5.0) # seconds
    parser_dico.add_argument('--echospacing', type=float, default=1.11) # ms
    parser_dico.add_argument('--TI', type=float,nargs='?', const=8.32, default=8.32) # ms 
    parser_dico.add_argument('--isbuildphi', type=bool,nargs='?', const=False, default=False) 
    parser_dico.add_argument('--force', type=bool,nargs='?', const=False, default=False) 
    parser_dico.add_argument('--pca', type=int,nargs='?', const=6, default=6)
    
    
    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        parser.exit()

    chosen_func=funcs[args.command]

    if args.command=="extract_data":    

        filename=args.filename
        kdata,dico_seqParams=chosen_func((filename))

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


    elif args.command=="build_volumes_singular":  
        file_kdata= args.filekdata
        file_b1=args.fileb1

        kdata=np.load(file_kdata)
        b1=np.load(file_b1)

        L0=args.L0
        lambd=args.lambd

        dictdir = pathlib.Path(args.dictdir)
        dictfiles=str(dictdir / args.dictfiles)

        with open(dictfiles,"rb") as file:
            dico_full_with_hdr=pickle.load(file)

        if "phi" not in dico_full_with_hdr.keys():
            print("No phi in dico")
            mrfdict=dico_full_with_hdr["mrfdict"]
            dico_full_with_hdr["phi"]=build_phi(mrfdict)

        phi=dico_full_with_hdr["phi"]

        niter=args.niter
        regularizer=args.regularizer
        if niter >0:
            if regularizer=="wavelet":
                if lambd is None:
                    lambd=2e-4
                mu=1
                kwargs_prox={
                    "wav_type":'db4',
                    "wav_level":None,
                    "axes":(2,3)
                }
            elif regularizer=="LLR":
                if lambd is None:
                    lambd=0.1
                mu=1
                kwargs_prox={
                    "blck":[1,8,8],
                    "strd":[1,2,2]
                }

            volumes_singular=build_volumes_singular_iterative(kdata,b1,phi,L0,niter,regularizer,lambd=lambd,mu=mu,**kwargs_prox)
        
        else:
            volumes_singular=build_volumes_singular_iterative(kdata,b1,phi,L0)


        path, _ = os.path.split(file_kdata)
        print(path)

        file_volumes=os.path.join(path,"volumes_singular.npy")
        file_volumes_gif=os.path.join(path,"volumes_singular.gif")

        np.save(file_volumes,volumes_singular)
        gif=[]
        sl=int(volumes_singular.shape[1]/2)
        volume_for_gif = np.abs(volumes_singular[:,sl])
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


    elif args.command=="build_masks_singular":

        file_volumes= args.filevolumes
        l=args.l
        threshold=args.threshold
        volumes_singular=np.load(file_volumes)
        masks=build_mask_from_singular_volume(volumes_singular,l,threshold=threshold)

        path, _ = os.path.split(file_volumes)
        print(path)

        file_masks=os.path.join(path,"masks_singular.npy")
        file_masks_gif=os.path.join(path,"masks_singular.gif")

        np.save(file_masks,masks)
        gif=[]
        volume_for_gif = np.abs(masks)
        for i in range(volume_for_gif.shape[0]):
            img = Image.fromarray(np.uint8(volume_for_gif[i]/np.max(volume_for_gif[i])*255), 'L')
            img=img.convert("P")
            gif.append(img)

                
        gif[0].save(file_masks_gif,save_all=True, append_images=gif[1:], optimize=False, duration=100, loop=0)
    
    elif args.command=="build_maps":
        # optim_config = args.optim_config
        file_volumes= str(args.filevolumes)
        file_masks=str(args.filemasks)
        file_seq=str(args.fileseq)
        dictdir = pathlib.Path(args.dictdir)
        dictfiles=str(dictdir / args.dictfiles)
        pca=int(args.pca)
        split=int(args.split)
        useGPU=bool(args.useGPU)
        return_cost=bool(args.returncost)
        volumes_type=str(args.volumestype)
        
        print(dictfiles)
        volumes=np.load(file_volumes)
        masks=np.load(file_masks)
        
        with open(dictfiles,"rb") as file:
            dico_full_with_hdr=pickle.load(file)
        
        # dictfile_light=dico_full_with_hdr["dictfile_light"]
        dico_hdr=dico_full_with_hdr["hdr"]
        clustering_windows={
            "wT1":600,
            "fT1":100,
            "df":0.03,
            "attB1":0.25
        }

        print("Clustering  windows : {}".format(clustering_windows))

        check_dico(dico_hdr,file_seq)
        all_maps=build_maps(volumes,masks,dictfiles,split=split,pca=pca,useGPU=useGPU,return_cost=return_cost,volumes_type=volumes_type,clustering_windows=clustering_windows)
        save_maps(all_maps,file_seq)

    elif args.command=="generate_dico":
        dictdir = pathlib.Path(args.dictdir)
        sequence_file= args.sequencefile
        reco=args.reco
        min_TR_delay=args.echospacing
        dictconf=args.dictconf
        dictconf_light=args.dictconflight
        TI=args.TI
        is_build_phi=args.isbuildphi
        force=args.force
        L0=int(args.pca)

        if sequence_file is None:
            print("No sequence config was given - using default SEQ_CONFIG")
            sequence_file=SEQ_CONFIG
        if dictconf is None:
            print("No dict config was given - using default DICT_LIGHT")
            dictconf=DICT_CONFIG
        if dictconf_light is None:
            print("No dict light config was given - using default DICT_LIGHT_CONFIG")
            dictconf_light=DICT_LIGHT_CONFIG        

        if not(force) and (dictdir.exists() and list(dictdir.glob('*'))):
            print('Output directory {} not emtpy. Aborting'.format(dictdir))
            parser.exit()
        dictdir.mkdir(parents=True, exist_ok=True)

        generate_dictionaries(sequence_file,reco,min_TR_delay,dictconf,dictconf_light,TI=8.32, dest=dictdir,is_build_phi=is_build_phi,L0=L0)

    else:
        raise("Value Error : Unknown Function")

