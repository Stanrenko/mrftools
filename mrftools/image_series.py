

# from mrftools.utils_simu import simulate_gen_eq_signal

import epgpy as epg
epg.set_array_module('numpy')
# from mrftools.utils_mrf import create_random_map,voronoi_volumes,normalize_image_series,build_mask_from_volume,generate_kdata,build_mask_single_image,buildROImask,correct_mvt_kdata,create_map
from .dictmodel import *

from mrftools.utils_mrf import makevol
import itertools
# from mrfsim import groupby,makevol,load_data,loadmat
import finufft
from tqdm import tqdm
from mrftools.trajectory import *

import matplotlib.pyplot as plt


DEFAULT_wT2 = 40
DEFAULT_fT1 = 300
DEFAULT_fT2 = 80

DEFAULT_ROUNDING_wT1=0
DEFAULT_ROUNDING_wT2=0
DEFAULT_ROUNDING_fT1=0
DEFAULT_ROUNDING_fT2=0
DEFAULT_ROUNDING_df=3 #df in kHz but chemical shifts generally order of magnitude in Hz
DEFAULT_ROUNDING_attB1=2
DEFAULT_ROUNDING_ff=2

DEFAULT_IMAGE_SIZE =(256,256)

UNITS ={
    "attB1":"a.u",
    "df":"kHz",
    "wT1":"ms",
    "fT1":"ms",
    "wT2":"ms",
    "fT2":"ms",
    "ff":"a.u"
}

PARAMS_WINDOWS={
    "attB1":[0,2],
    "df":[-0.5,0.5],
    "wT1":[0,2000],
    "fT1":[0,1000],
    "wT2":[0,500],
    "fT2":[0,500],
    "ff":[0,1]


}

def dump_function(func):
    """
    Decorator to print function call details.

    This includes parameters names and effective values.
    """

    def wrapper(*args, **kwargs):

        print(f"{func.__module__}.{func.__qualname__} ")
        return func(*args, **kwargs)

def wrapper_rounding(func):
    def wrapper(self, *args, **kwargs):
        print("Building Param Map")
        func(self,*args,**kwargs)
        if self.paramDict["rounding"]:
            print("Warning : Values in the initial map are being rounded")
            for paramName in self.paramMap.keys():
                self.roundParam(paramName,self.paramDict["rounding_"+paramName])

        if "dict_overrides" in self.paramDict:
            print("Warning : Overriding map values for params {}" .format(self.paramDict["dict_overrides"].keys()))
            for paramName in self.paramDict["dict_overrides"].keys():
                self.paramMap[paramName]=np.ones(self.paramMap[paramName].shape)*self.paramDict["dict_overrides"][paramName]

    return wrapper


class ImageSeries(object):

    def __init__(self,name,dict_config={},**kwargs):
        self.name=name
        self.dict_config=dict_config
        self.paramDict = kwargs
        if "image_size" not in self.paramDict:
            self.paramDict["image_size"]=DEFAULT_IMAGE_SIZE

        if "nb_rep" not in self.paramDict:
            self.paramDict["nb_rep"]=1

        # if "sim_mode" not in self.paramDict:
        #     self.paramDict["sim_mode"]="mean" #mid_point

        if "gen_mode" not in self.paramDict:
            self.paramDict["gen_mode"]=None #loop

        self.image_size=self.paramDict["image_size"]
        self.images_series=None
        self.cached_images_series=None

        self.mask =np.ones(self.image_size)
        self.paramMap=None

        self.list_movements=[]

        if "rounding" not in self.paramDict:
            self.paramDict["rounding"]=False
        else:
            if self.paramDict["rounding"]:
                if "rounding_wT1" not in self.paramDict:
                    self.paramDict["rounding_wT1"] = DEFAULT_ROUNDING_wT1
                if "rounding_wT2" not in self.paramDict:
                    self.paramDict["rounding_wT2"] = DEFAULT_ROUNDING_wT2
                if "rounding_fT1" not in self.paramDict:
                    self.paramDict["rounding_fT1"] = DEFAULT_ROUNDING_fT1
                if "rounding_fT2" not in self.paramDict:
                    self.paramDict["rounding_fT2"] = DEFAULT_ROUNDING_fT2
                if "rounding_ff" not in self.paramDict:
                    self.paramDict["rounding_ff"] = DEFAULT_ROUNDING_ff
                if "rounding_attB1" not in self.paramDict:
                    self.paramDict["rounding_attB1"] = DEFAULT_ROUNDING_attB1
                if "rounding_df" not in self.paramDict:
                    self.paramDict["rounding_df"] = DEFAULT_ROUNDING_df


        self.fat_amp=[0.0586, 0.0109, 0.0618, 0.1412, 0.66, 0.0673]
        fat_cs = [-101.1, 208.3, 281.0, 305.7, 395.6, 446.2]
        self.fat_cs = [- value / 1000 for value in fat_cs]  # temp


    def build_ref_images(self,seq,norm=None,phase=None,epg_opt = {"disp": True, 'max_nstate':30},useGPU=True):

        if useGPU:
            epg.set_array_module('cupy')
        else:
            epg.set_array_module('numpy')

        print("Building Ref Images")
        if self.paramMap is None:
            return ValueError("buildparamMap should be called prior to image simulation")

        list_keys = ["wT1","wT2","fT1","fT2","attB1","df","ff"]
        for k in list_keys:
            if k not in self.paramMap:
                raise ValueError("key {} should be in the paramMap".format(k))

        map_all_on_mask = np.stack(list(self.paramMap.values())[:-1], axis=-1)
        map_ff_on_mask = self.paramMap["ff"]

        params_all = np.reshape(map_all_on_mask, (-1, 6))
        params_unique = np.unique(params_all, axis=0)

        wT1_in_map = np.unique(params_unique[:, 0])
        wT2_in_map = np.unique(params_unique[:, 1])
        fT1_in_map = np.unique(params_unique[:, 2])
        fT2_in_map = np.unique(params_unique[:, 3])
        attB1_in_map = np.unique(params_unique[:, 4])
        df_in_map = np.unique(params_unique[:, 5])

        # Simulating the image sequence

        # water
        print("Simulating Water Signal")

        if self.paramDict["gen_mode"]=="loop":
            water_list=[]
            print("Simulation in loop mode")
            for param in tqdm(params_unique):
                current_water = seq(T1=param[0], T2=param[1], att=param[4], g=param[5],options= epg_opt)
                water_list.append(current_water)
            water = np.squeeze(np.array(water_list))
            del water_list
            water=water.T

        else:
            water = seq(T1=wT1_in_map, T2=wT2_in_map, att=[[attB1_in_map]], g=[[[df_in_map]]],options= epg_opt)



        # fat
        print("Simulating Fat Signal")


        if self.paramDict["gen_mode"] == "loop":
            fat_list = []
            print("Simulation in loop mode")
            for param in tqdm(params_unique):
                current_fat = seq(T1=param[2], T2=param[3], att=param[4], g=[cs + param[5] for cs in self.fat_cs],options= epg_opt)#,eval=eval,args=args)
                current_fat=current_fat @ self.fat_amp
                fat_list.append(current_fat)
            fat = np.squeeze(np.array(fat_list))
            del fat_list
            fat = fat.T

        else:
            # merge df and fat_cs df to dict
            fatdf_in_map = [[cs + f for cs in self.fat_cs] for f in df_in_map]
            fat = seq(T1=[fT1_in_map], T2=fT2_in_map, att=[[attB1_in_map]], g=[[[fatdf_in_map]]],options= epg_opt)#, eval=eval, args=args)
            fat = fat @ self.fat_amp

        # join water and fat
        print("Build dictionary.")
        if self.paramDict["gen_mode"] == "loop":
            keys=[tuple(param) for param in params_unique]
            values=np.stack((water, fat),axis=-1)
            values = np.moveaxis(values, 0, 1)
        else :
            keys = list(itertools.product(wT1_in_map, wT2_in_map, fT1_in_map, fT2_in_map, attB1_in_map, df_in_map))
            print(water.shape)
            print(fat.shape)
            values = np.stack(np.broadcast_arrays(water, fat), axis=-1)
            values = np.moveaxis(values.reshape(len(values), -1, 2), 0, 1)
        mrfdict = Dictionary(keys, values)
        

        images_series = np.zeros(self.image_size + (values.shape[-2],), dtype=np.complex_)

        print("Building image series")
        images_in_mask = np.array([mrfdict[tuple(pixel_params)][:, 0] * (1 - map_ff_on_mask[i]) + mrfdict[tuple(
            pixel_params)][:, 1] * (map_ff_on_mask[i]) for (i, pixel_params) in enumerate(map_all_on_mask)])

        if norm is not None :
            images_in_mask *= np.expand_dims(norm,axis=1)

        if phase is not None:
            images_in_mask *= np.expand_dims(np.exp(1j*phase),axis=1)

        print("Image series built")

        images_series[self.mask > 0, :] = images_in_mask

        images_series = np.moveaxis(images_series, -1, 0)

        self.images_series=images_series

        self.cached_images_series=images_series

    def build_ref_images_v2(self,seq, useGPU=True, norm=None,phase=None,epg_opt = {"disp": True, 'max_nstate':30}):

        # mask_np = np.asarray(maps["mask"])
        # idx = np.where(mask_np > 0)
        
        # generate signals
        wT1 = self.paramMap['wT1']
        fT1 = self.paramMap['fT1']
        wT2 = self.paramMap['wT2']
        fT2 = self.paramMap['fT2']
        att = self.paramMap['attB1']
        df = self.paramMap['df']
        df = np.asarray(df)/1000
        ff = self.paramMap['ff']

        if useGPU:
            epg.set_array_module('cupy')
        else:
            epg.set_array_module('numpy')
        # epg_opt = {"disp": True, 'max_nstate':30}
            
        # TR_delay=sequence_config["dTR"]

        # seq = T1MRF_generic(sequence_config)


        water_amp = [1]
        water_cs = [0]
        fat_amp = self.fat_amp
        fat_cs = self.fat_cs
    

        # other options
        # if window is None:
        #     window = dict_config["window_size"]


        # if start is None:
        #     dictfile = prefix_dico  +"_TR{}_reco{}.dict".format(str(TR_delay),str(sequence_config['T_recovery']))
        # else:
        #     dictfile = prefix_dico + "_TR{}_reco{}_start{}.dict".format(str(TR_delay),str(sequence_config['T_recovery']),start)

        # if dest is not None:
        #     dictfile = str(pathlib.Path(dest) / pathlib.Path(dictfile).name)
        # print("Generating dictionary {}".format(dictfile))

        # water
        print("Generate water signals.")
        
        water = seq(T1=np.array(wT1), T2=np.array(wT2), att=np.array(att), g=np.array(df)[:,np.newaxis], cs=[water_cs], frac=[water_amp], options= epg_opt)
        water = np.transpose(water, (1, 0))


        # fat
        print("Generate fat signals.")
        # eval = "dot(signal, amps)"
        # args = {"amps": fat_amp}
        # merge df and fat_cs df to dict
        fat = seq(T1=np.array(fT1), T2=np.array(fT2), att=np.array(att), g=np.array(df)[:,np.newaxis], cs=[fat_cs], frac=[fat_amp], options= epg_opt)#, eval=eval, args=args)
        fat = np.transpose(fat, (1, 0))
        

        water = np.array(water)
        fat = np.array(fat)
        
        signal = (1-np.asarray(ff))*water + np.asarray(ff)*fat
        
        imgseries = np.zeros((signal.shape[0], self.mask.shape[0], self.mask.shape[1]), dtype=np.complex64)
        imgseries = imgseries.reshape(signal.shape[0], -1)
        imgseries[:, self.mask.flatten() > 0] = signal
        imgseries = imgseries.reshape(signal.shape[0], self.mask.shape[0], self.mask.shape[1])
        
        if phase is not None:
            imgseries *= np.exp(1j*phase)

        if norm is not None :
            imgseries *= norm

        if phase is not None:
            images_in_mask *= np.expand_dims(np.exp(1j*phase),axis=1)
        
        self.images_series=imgseries

        #self.water_series = water_series
        #self.fat_series=fat_series

        self.cached_images_series=imgseries

    def generate_kdata(self,trajectory,eps=1e-4,movement_correction=False,perc=80,nthreads=1,fftw=0):
        print("Generating kdata")
        #nspoke = trajectory.paramDict["nspoke"]
        #npoint = trajectory.paramDict["npoint"]
        

        traj = trajectory.get_traj()
        print(self.images_series.dtype)
        #traj = traj.reshape((self.images_series.shape[0], -1, traj.shape[-1]))
        if self.images_series.dtype=="complex64":
            traj = traj.astype("float32")

        if not (traj.shape[-1] == len(self.image_size)):
            raise ValueError("Trajectory dimension does not match Image Space dimension")

        size = self.image_size
        if trajectory.applied_timesteps is None:
            images_series = self.images_series
        else:
            images_series=self.images_series[trajectory.applied_timesteps]
        # images_series =normalize_image_series(self.images_series)

        if traj.shape[-1] == 2:  # 2D
            kdata = [finufft.nufft2d2(t[:, 0], t[:, 1], p,nthreads=nthreads,fftw=fftw) for t, p in tqdm(zip(traj, images_series))]

                

        elif traj.shape[-1] == 3:  # 3D

                
            kdata = []
            for i in tqdm(range(len(traj))):
                kdata.append(finufft.nufft3d2(traj[i,:,2],traj[i, :, 0], traj[i, :, 1], images_series[i]))

        return kdata

    def generate_kdata_multi(self,trajectory,b1_all_slices,useGPU=False,eps=1e-4):
        b1_prev = np.ones(b1_all_slices[0].shape, dtype=b1_all_slices[0].dtype)
        b1_all = np.concatenate([np.expand_dims(b1_prev, axis=0), b1_all_slices], axis=0)

        kdata = []

        # images = copy(m.images_series)

        for i in tqdm(range(1, b1_all.shape[0])):
            self.images_series *= np.expand_dims(b1_all[i] / b1_all[i - 1], axis=0)
            kdata.append(np.array(self.generate_kdata(trajectory, useGPU=useGPU,eps=eps)))

        self.images_series /= np.expand_dims(b1_all[-1], axis=0)
        # del images

        kdata = np.array(kdata)

        return kdata



    def buildParamMap(self,mask=None):
        raise ValueError("should be implemented in child")

    def plotParamMap(self,key=None,figsize=(5,5),fontsize=5,save=False,sl=None):
        if len(self.image_size)==2:
            if key is None:
                keys=list(self.paramMap.keys())
                fig,axes=plt.subplots(1,len(keys),figsize=(len(keys)*figsize[0],figsize[1]))
                for i,k in enumerate(keys):
                    im=axes[i].imshow(makevol(self.paramMap[k],(self.mask>0)))
                    axes[i].set_title("{} Map {}".format(k,self.name))
                    axes[i].tick_params(axis='x', labelsize=fontsize)
                    axes[i].tick_params(axis='y', labelsize=fontsize)
                    cbar=fig.colorbar(im, ax=axes[i],fraction=0.046, pad=0.04)
                    cbar.ax.tick_params(labelsize=fontsize)
                if save:
                    plt.savefig("./figures/ParamMap_{}_all".format(self.name, key))
            else:
                fig,ax=plt.subplots(figsize=figsize)

                im=ax.imshow(makevol(self.paramMap[key],(self.mask>0)))
                ax.set_title("{} Map {}".format(key,self.name))
                ax.tick_params(axis='x', labelsize=fontsize)
                ax.tick_params(axis='y', labelsize=fontsize)
                cbar=fig.colorbar(im, ax=ax,fraction=0.046, pad=0.04)
                cbar.ax.tick_params(labelsize=fontsize)

                if save:
                    plt.savefig("./figures/ParamMap_{}_{}".format(self.name,key))
        elif len(self.image_size)==3:
            if sl is None:
                print("WARNING : plotting the paramMap for slice 0 as sl number was not provided")
                sl=0

            if key is None:
                keys = list(self.paramMap.keys())
                fig, axes = plt.subplots(1, len(keys), figsize=(len(keys) * figsize[0], figsize[1]))
                for i, k in enumerate(keys):
                    im = axes[i].imshow(makevol(self.paramMap[k], (self.mask > 0))[sl,:,:])
                    axes[i].set_title("{} Map {} Slice {}".format(k, self.name,sl))
                    axes[i].tick_params(axis='x', labelsize=fontsize)
                    axes[i].tick_params(axis='y', labelsize=fontsize)
                    cbar = fig.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
                    cbar.ax.tick_params(labelsize=fontsize)
                if save:
                    plt.savefig("./figures/ParamMap_{}_sl_{}_all".format(self.name, key,sl))
            else:
                fig, ax = plt.subplots(figsize=figsize)

                im = ax.imshow(makevol(self.paramMap[key], (self.mask > 0))[sl,:,:])
                ax.set_title("{} Map {}".format(key, self.name))
                ax.tick_params(axis='x', labelsize=fontsize)
                ax.tick_params(axis='y', labelsize=fontsize)
                cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                cbar.ax.tick_params(labelsize=fontsize)

                if save:
                    plt.savefig("./figures/ParamMap_{}_{}_sl_{}".format(self.name, key,sl))




        plt.show()

    def roundParam(self,paramName,decimals=0):
        self.paramMap[paramName]=np.round(self.paramMap[paramName],decimals=decimals)

    def reset_image_series(self):
        self.images_series=self.cached_images_series

    def compare_patterns(self,pixel_number):

        fig,(ax1,ax2,ax3) = plt.subplots(1,3)
        ax1.plot(np.real(self.cached_images_series[:,pixel_number[0],pixel_number[1]]),label="original image - real part")
        ax1.plot(np.real(self.images_series[:, pixel_number[0],pixel_number[1]]), label="transformed image - real part")
        ax1.legend()
        ax2.plot(np.imag(self.cached_images_series[:, pixel_number[0], pixel_number[1]]),
                 label="original image - imaginary part")
        ax2.plot(np.imag(self.images_series[:, pixel_number[0], pixel_number[1]]),
                 label="transformed - imaginary part")
        ax2.legend()

        ax3.plot(np.abs(self.cached_images_series[:, pixel_number[0], pixel_number[1]]),
                 label="original image - norm")
        ax3.plot(np.abs(self.images_series[:, pixel_number[0], pixel_number[1]]),
                 label="transformed - norm")
        ax3.legend()

        plt.show()



class MapFromDict(ImageSeries):

    def __init__(self, name, **kwargs):
        super().__init__(name, {}, **kwargs)



        if "paramMap" not in self.paramDict:
            raise ValueError("paramMap key value argument containing param map file path should be given for MapFromFile")

        if "default_wT2" not in self.paramDict:
            self.paramDict["default_wT2"]=DEFAULT_wT2
        if "default_fT2" not in self.paramDict:
            self.paramDict["default_fT2"]=DEFAULT_fT2
        if "default_fT1" not in self.paramDict:
            self.paramDict["default_fT1"]=DEFAULT_fT1

    @wrapper_rounding
    def buildParamMap(self):



        paramMap = self.paramDict["paramMap"]

        map_wT1=paramMap["wT1"]
        self.image_size=map_wT1.shape

        if "mask" in self.paramDict:
            mask=self.paramDict["mask"]

        else:
            mask = np.zeros(self.image_size)
            mask[map_wT1>0]=1.0

        self.mask=mask

        map_wT2 = mask*self.paramDict["default_wT2"]
        map_fT1 = mask*self.paramDict["default_fT1"]
        map_fT2 = mask*self.paramDict["default_fT2"]

        map_all = np.stack((paramMap["wT1"], map_wT2, map_fT1, map_fT2, paramMap["attB1"], paramMap["df"], paramMap["ff"]), axis=-1)
        map_all_on_mask = map_all[mask > 0]

        self.paramMap = {
            "wT1": map_all_on_mask[:, 0],
            "wT2": map_all_on_mask[:, 1],
            "fT1": map_all_on_mask[:, 2],
            "fT2": map_all_on_mask[:, 3],
            "attB1": map_all_on_mask[:, 4],
            "df": map_all_on_mask[:, 5],
            "ff": map_all_on_mask[:, 6]
        }
