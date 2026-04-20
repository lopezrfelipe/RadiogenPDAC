from typing import Union, Tuple, Callable, List
import numpy as np
from batchgenerators.transforms.abstract_transforms import AbstractTransform


def mask_augment_contrast(
    data_sample: np.ndarray,
    contrast_range: Union[Tuple[float, float], Callable[[], float]] = (0.75, 1.25),
    preserve_range: bool = True):
    
    if callable(contrast_range):
        factor = contrast_range()
    else:
        if np.random.random() < 0.5 and contrast_range[0] < 1:
            factor = np.random.uniform(contrast_range[0], 1)
        else:
            factor = np.random.uniform(max(contrast_range[0], 1), contrast_range[1])

    mn = data_sample.mean()
    if preserve_range:
        minm = data_sample.min()
        maxm = data_sample.max()

    data_sample = (data_sample - mn) * factor + mn

    if preserve_range:
        data_sample[data_sample < minm] = minm
        data_sample[data_sample > maxm] = maxm

    return data_sample



class MaskTransform(AbstractTransform):
    def __init__(self, apply_to_channels: List[int], mask_idx_in_seg: int = 0, set_outside_to: int = 0,
                 data_key: str = "data", seg_key: str = "seg"):
        """
        Sets everything outside the mask to 0. CAREFUL! outside is defined as < 0, not =0 (in the Mask)!!!
        """
        self.apply_to_channels = apply_to_channels
        self.seg_key = seg_key
        self.data_key = data_key
        self.set_outside_to = set_outside_to
        self.mask_idx_in_seg = mask_idx_in_seg

    def __call__(self, **data_dict):
        mask = data_dict[self.seg_key][:, self.mask_idx_in_seg] < 0
        for c in self.apply_to_channels:
            data_dict[self.data_key][:, c][mask] = self.set_outside_to
        return data_dict



# class MaskAugTransform(AbstractTransform):
#     def __init__(self, 
#         data_key: str = "data", 
#         seg_key: str = "seg", 
#         contrast_range: Union[Tuple[float, float], Callable[[], float]] = (0.75, 1.25),
#         preserve_range: bool = True,
#         p_per_sample: float = 1):
        
#         self.seg_key = seg_key
#         self.data_key = data_key
#         self.contrast_range = contrast_range
#         self.preserve_range = preserve_range
#         self.p_per_sample = p_per_sample

#     def __call__(self, **data_dict):
#         mask = (data_dict[self.seg_key][:, 0] > 0) & (data_dict[self.seg_key][:, 0] < 3)
#         if np.random.uniform() < self.p_per_sample:
#             data_dict[self.data_key][:, 0][mask] = mask_augment_contrast(
#                 data_dict[self.data_key][:, 0][mask],
#                 contrast_range=self.contrast_range,
#                 preserve_range=self.preserve_range)

#         return data_dict



class TumorAugTransform(AbstractTransform):
    def __init__(self, 
        data_key: str = "data", 
        seg_key: str = "seg", 
        p_per_sample: float = 1):
        
        self.seg_key = seg_key
        self.data_key = data_key
        self.p_per_sample = p_per_sample

    def __call__(self, **data_dict):
        if np.random.uniform() < self.p_per_sample:
            mask = (data_dict[self.seg_key][:, 0] > 0) & (data_dict[self.seg_key][:, 0] < 3)
            val = np.random.uniform(1.2, 2)
            data_dict[self.data_key][:, 0][mask] *= val
        return data_dict


class TumorAug1Transform(AbstractTransform):
    def __init__(self, 
        data_key: str = "data", 
        seg_key: str = "seg", 
        p_per_sample: float = 1):
        
        self.seg_key = seg_key
        self.data_key = data_key
        self.p_per_sample = p_per_sample

    def __call__(self, **data_dict):
        if np.random.uniform() < self.p_per_sample:
            for i in range(data_dict[self.data_key].shape[0]):
                mask = (data_dict[self.seg_key][i, 0] > 0) & (data_dict[self.seg_key][i, 0] < 3)
                global_min = data_dict[self.data_key][i, 0].min() * np.random.uniform(0, 0.5)  # a negative value
                data_dict[self.data_key][i, 0][mask] += global_min
        return data_dict


class TumorAug2Transform(AbstractTransform):
    def __init__(self, 
        data_key: str = "data", 
        seg_key: str = "seg", 
        p_per_sample: float = 1):
        
        self.seg_key = seg_key
        self.data_key = data_key
        self.p_per_sample = p_per_sample

    def __call__(self, **data_dict):
        if np.random.uniform() < self.p_per_sample:
            mask = (data_dict[self.seg_key][:, 0] > 0) & (data_dict[self.seg_key][:, 0] < 3)
            val = np.random.uniform(1.2, 2)
            data_dict[self.data_key][:, 0][mask] *= val
        return data_dict


class TumorAug3Transform(AbstractTransform):
    def __init__(self, 
        data_key: str = "data", 
        seg_key: str = "seg", 
        p_per_sample: float = 1):
        
        self.seg_key = seg_key
        self.data_key = data_key
        self.p_per_sample = p_per_sample

    def __call__(self, **data_dict):
        if np.random.uniform() < self.p_per_sample:
            mask = (data_dict[self.seg_key][:, 0] > 0) & (data_dict[self.seg_key][:, 0] < 3)
            data_dict[self.data_key][:, 0][mask] *= -1
        return data_dict


class CochleaAugTransform(AbstractTransform):
    def __init__(self, 
        data_key: str = "data", 
        seg_key: str = "seg", 
        p_per_sample: float = 1):
        
        self.seg_key = seg_key
        self.data_key = data_key
        self.p_per_sample = p_per_sample

    def __call__(self, **data_dict):
        if np.random.uniform() < self.p_per_sample:
            mask = data_dict[self.seg_key][:, 0] == 3 ####################################################################
            val = np.random.uniform(0.5, 1)
            data_dict[self.data_key][:, 0][mask] *= val
        return data_dict


class OrganAugTransform(AbstractTransform):
    def __init__(self, 
        data_key: str = "data", 
        seg_key: str = "seg", 
        organ_id: int = 0,
        p_per_sample: float = 1):
        self.seg_key = seg_key
        self.data_key = data_key
        self.organ_id = organ_id
        self.p_per_sample = p_per_sample

    def __call__(self, **data_dict):
        if np.random.uniform() < self.p_per_sample:
            mask = data_dict[self.seg_key][:, 0] == self.organ_id
            val = np.random.uniform(0.6, 1.2)
            if np.random.uniform() < 0.5:
                val = -val
            data_dict[self.data_key][:, 0][mask] *= val
        return data_dict


class LesionAugTransform(AbstractTransform):
    def __init__(self, 
        data_key: str = "data", 
        seg_key: str = "seg", 
        p_per_sample: float = 1):
        self.seg_key = seg_key
        self.data_key = data_key
        self.p_per_sample = p_per_sample

    def __call__(self, **data_dict):
        if np.random.uniform() < self.p_per_sample:
            mask = data_dict[self.seg_key][:, 0] == 1
            val = np.random.uniform(0.8, 1.2)
            data_dict[self.data_key][:, 0][mask] *= val
        return data_dict

