

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


    def build_ref_images(self,seq,norm=None,phase=None):
        wT1 = self.paramMap['wT1']
        fT1 = self.paramMap['fT1']
        wT2 = self.paramMap['wT2']
        fT2 = self.paramMap['fT2']
        att = self.paramMap['attB1']
        df = self.paramMap['df']
        df = np.asarray(df)/1000
        ff = self.paramMap['ff']

        water_amp = [1]
        water_cs = [0]
        fat_amp = self.fat_amp
        fat_cs = self.fat_cs

        print("Generate water signals.")
            
        water = seq(T1=np.array(wT1), T2=np.array(wT2), att=np.array(att), g=np.array(df), cs=water_cs, frac=water_amp).squeeze()
        water= np.transpose(water, (1, 0))

        print("Generate fat signals.")
        fat = seq(T1=np.array(fT1), T2=np.array(fT2), att=np.array(att), g=np.array(df),cs=fat_cs,frac=fat_amp)#, eval=eval, args=args)
        fat = np.transpose(fat, (1, 0))


        water = np.array(water)
        fat = np.array(fat)
        signal = (1-np.asarray(ff[:,None]))*water + np.asarray(ff[:,None])*fat

        if norm is not None:
            signal*=norm[:,None]
        if phase is not None:
            signal*=np.exp(1j*phase[:,None])


        print("Building image series")

        images_series = np.zeros(self.image_size + (signal.shape[-1],), dtype=np.complex128)
        images_series[self.mask > 0, :] = signal
        self.images_series=np.moveaxis(images_series, -1, 0)

        print("Image series built")


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
