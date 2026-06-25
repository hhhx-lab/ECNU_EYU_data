import math
import torch
from monai.config import KeysCollection
from monai.transforms.compose import MapTransform
from torch import clone as clone
import numpy as np
import random

def zscore_then_rescale(arr, target_min=-1.0, target_max=1.0):
    """逐样本 z-score 标准化后再 rescale 到 [target_min, target_max]，替代 min-max 归一化。"""
    mean = np.mean(arr)
    std = np.std(arr)
    if std == 0:
        return arr
    z = (arr - mean) / std
    z_min, z_max = np.min(z), np.max(z)
    if z_max == z_min:
        return np.full_like(arr, (target_min + target_max) / 2)
    return (z - z_min) / (z_max - z_min) * (target_max - target_min) + target_min


class GaussianNoiseTumourExtended(MapTransform):
    """
    Adds sphere-shaped noise to the volume (making sure the entire tumor region is noise).
 The size of the sphere depends on the largest size (of the 3 axis).
    """
    def __init__(self, keys: KeysCollection, normalization="minmax"):
        super().__init__(keys)
        self.keys = keys
        self.normalization = normalization
    def __call__(self, data):
        d = dict(data)
        scan_key = self.keys  # e.g. "scan_t1c" / "scan_t1n" / "scan_t2w" / "scan_t2f"
        scan_data = d[scan_key]
        _, max_x, max_y, max_z = scan_data.shape
        scan_crop = clone(scan_data)
        label = d["label"]
        label_crop = clone(label)

        x_extreme_dif = d["x_extreme_max"] - d["x_extreme_min"]
        y_extreme_dif = d["y_extreme_max"] - d["y_extreme_min"]
        z_extreme_dif = d["z_extreme_max"] - d["z_extreme_min"]

        x_pad = (96 - x_extreme_dif) / 2
        y_pad = (96 - y_extreme_dif) / 2
        z_pad = (96 - z_extreme_dif) / 2

        if x_pad < 0:
            C_x = -0.5
        else:
            C_x = 0.5

        if y_pad < 0:
            C_y = -0.5
        else:
            C_y = 0.5

        if z_pad < 0:
            C_z = -0.5
        else:
            C_z = 0.5

        x_base = d["x_extreme_min"] - int(x_pad)
        x_top = d["x_extreme_max"] + int(x_pad+C_x) 
        y_base = d["y_extreme_min"] - int(y_pad) 
        y_top = d["y_extreme_max"] + int(y_pad+C_y) 
        z_base = d["z_extreme_min"] - int(z_pad) 
        z_top = d["z_extreme_max"] + int(z_pad+C_z) 
        
        # Verifying the need for padding
        x_base_pad = 0
        y_base_pad = 0
        z_base_pad = 0
        x_top_pad = 0
        y_top_pad = 0
        z_top_pad = 0

        if x_base < 0:
            x_base_pad = -x_base
            x_base = 0
            
        if y_base < 0:
            y_base_pad = -y_base
            y_base = 0
            
        if z_base < 0:
            z_base_pad = -z_base
            z_base = 0
            
        if x_top > max_x:
            x_top_pad = x_top-max_x
            x_top = max_x
            
        if y_top > max_y:
            y_top_pad = y_top-max_y
            y_top = max_y
            
        if z_top > max_z:
            z_top_pad = z_top-max_z
            z_top = max_z
        ##################################
        # Crop the label
        label_crop = label_crop[:, x_base : x_top, y_base : y_top, z_base : z_top]
        
        # Crop and Normalise the scan
        scan_crop = scan_crop[:, x_base : x_top, y_base : y_top, z_base : z_top]

        if torch.sum(scan_crop)==0:
            raise CustomError("It is an empty case")

        if self.normalization == "zscore":
            scan_crop = zscore_then_rescale(scan_crop, target_min=-1.0, target_max=1.0)
        else:
            scan_crop = self.rescale_array(arr=scan_crop, minv=-1, maxv=1)
        d["scan_crop"] = scan_crop

        # Scan and label with padding for 96, 96, 96 (if needed)
        scan_crop_pad = clone(scan_crop)
        scan_crop_pad = np.pad(scan_crop_pad, pad_width=((0,0), (x_base_pad,x_top_pad), (y_base_pad,y_top_pad), (z_base_pad,z_top_pad)), mode='constant', constant_values=(-1, -1))
        label_crop_pad = clone(label_crop)
        label_crop_pad = np.pad(label_crop_pad, pad_width=((0,0), (x_base_pad,x_top_pad), (y_base_pad,y_top_pad), (z_base_pad,z_top_pad)), mode='constant', constant_values=(0, 0))

        # Computing the noise size
        max_size = max(d['x_size'], d["y_size"], d["z_size"])
        exp_base = self.norm_exp_base(value=max_size)

        scan_noisy = self.add_gaussian_noise_extended(scan=scan_crop_pad, label=label_crop_pad, exp_base=exp_base)
        if self.normalization == "zscore":
            scan_noisy = zscore_then_rescale(scan_noisy, target_min=-1.0, target_max=1.0)
        else:
            scan_noisy = self.rescale_array_numpy(arr=scan_noisy, minv=-1, maxv=1)

        d[scan_key] = scan_data
        d[f"{scan_key}_crop"] = scan_crop
        d[f"{scan_key}_crop_pad"] = scan_crop_pad
        d[f"{scan_key}_noisy"] = scan_noisy
        d["label_crop"] = label_crop  
        d["label_crop_pad"] = label_crop_pad

        return d

    def rescale_array(self, arr, minv, maxv): #monai function adapted
        """
        Rescale the values of numpy array `arr` to be from `minv` to `maxv`.
        """
        mina = torch.min(arr)
        maxa = torch.max(arr)
        if mina == maxa:
            return arr * minv
        # normalize the array first
        norm = (arr - mina) / (maxa - mina) 
        # rescale by minv and maxv, which is the normalized array by default 
        return (norm * (maxv - minv)) + minv  

    def rescale_array_numpy(self, arr, minv, maxv): #monai function adapted
        """
        Rescale the values of numpy array `arr` to be from `minv` to `maxv`.
        """
        mina = np.min(arr)
        maxa = np.max(arr)
        if mina == maxa:
            return arr * minv
        # normalize the array first
        norm = (arr - mina) / (maxa - mina) 
        # rescale by minv and maxv, which is the normalized array by default 
        return (norm * (maxv - minv)) + minv  
    
    def distance_3d(self, point1, point2):
        """
        Compute the distance between two points
        Parameters:
                point1 (tuple): Point 1 coordinates
                point2 (tuple): Point 2 coordinates
        Returns:
                distance (float): Distance between the two points
                """
        x1, y1, z1 = point1
        x2, y2, z2 = point2
        distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)
        return distance
    
    def norm_exp_base(self, value):
        """
        Rescale the value to fit between 1.1 and 1.3, having as max 96 and min 28.
        """
        m = - 0.2/68
        c = 1.1 - 96*m
        return (m)*value + c

    def add_gaussian_noise_extended(self, scan, label, exp_base):
        """
        Adds Gaussian noise to the scan to mask the tumour
            Parameters:
                    scan (array): Scan to add Gaussian noise
            Returns:
                    scan (array): Scan with Gaussian noise
        """
        scan_noisy = np.copy(scan)
        noise =  np.full((1,96,96,96), 1000.)
        point1 = (48,48,48) # Point in the center
        for x_axis in range(0, 96):
            for y_axis in range(0, 96):
                for z_axis in range(0, 96):
                    if True in label[:, x_axis, y_axis, z_axis]:
                        noise[0,x_axis,y_axis,z_axis] = torch.randn(1)
                    else:
                        distance = self.distance_3d(point1=point1, point2=(x_axis+1,y_axis+1,z_axis+1))
                        prob = 83/(exp_base**distance+82)
                        if random.random() <= prob:
                            noise[0,x_axis,y_axis,z_axis] = torch.randn(1)
        #noise = rescale_gaussian_noise(noise, -1, 1)
        
        np.copyto(scan_noisy, noise, where= np.logical_and(noise<100 , scan_noisy!=-1))
        return scan_noisy
    
    