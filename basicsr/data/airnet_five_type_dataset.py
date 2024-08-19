import torch
from torchvision.transforms import ToPILImage, Compose, RandomCrop, ToTensor, Grayscale

from PIL import Image
import random
import numpy as np


import os
import os.path as osp
import random
from PIL import Image
import numpy as np

import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToPILImage, Compose, RandomCrop, ToTensor


# from .promptir_util import random_augmentation, crop_img
# from uclsr.utils.registry import DATASET_REGISTRY
from basicsr.data.airnet_util.image_utils import random_augmentation, crop_img
from basicsr.utils.registry import DATASET_REGISTRY

class Degradation(object):
    def __init__(self, opt):
        super(Degradation, self).__init__()
        self.opt = opt
        self.toTensor = ToTensor()
        self.crop_transform = Compose(
            [
                ToPILImage(),
                RandomCrop(opt["patch_size"]),
            ]
        )

    def _add_gaussian_noise(self, clean_patch, sigma):
        # noise = torch.randn(*(clean_patch.shape))
        # clean_patch = self.toTensor(clean_patch)
        noise = np.random.randn(*clean_patch.shape)
        noisy_patch = np.clip(clean_patch + noise * sigma,
                              0, 255).astype(np.uint8)
        # noisy_patch = torch.clamp(clean_patch + noise * sigma, 0, 255).type(torch.int32)
        return noisy_patch, clean_patch

    def _degrade_by_type(self, clean_patch, degrade_type):
        if degrade_type == 0:
            # denoise sigma=15
            degraded_patch, clean_patch = self._add_gaussian_noise(
                clean_patch, sigma=15
            )
        elif degrade_type == 1:
            # denoise sigma=25
            degraded_patch, clean_patch = self._add_gaussian_noise(
                clean_patch, sigma=25
            )
        elif degrade_type == 2:
            # denoise sigma=50
            degraded_patch, clean_patch = self._add_gaussian_noise(
                clean_patch, sigma=50
            )

        return degraded_patch, clean_patch

    def degrade(self, clean_patch_1, clean_patch_2, degrade_type=None):
        if degrade_type == None:
            degrade_type = random.randint(0, 3)
        else:
            degrade_type = degrade_type

        degrad_patch_1, _ = self._degrade_by_type(clean_patch_1, degrade_type)
        degrad_patch_2, _ = self._degrade_by_type(clean_patch_2, degrade_type)
        return degrad_patch_1, degrad_patch_2

    def single_degrade(self, clean_patch, degrade_type=None):
        if degrade_type == None:
            degrade_type = random.randint(0, 3)
        else:
            degrade_type = degrade_type

        degrad_patch_1, _ = self._degrade_by_type(clean_patch, degrade_type)
        return degrad_patch_1


@DATASET_REGISTRY.register()
class PromptTrainDataset(Dataset):
    def __init__(self, opt):
        super(PromptTrainDataset, self).__init__()
        self.opt = opt
        self.rs_ids = []
        self.hazy_ids = []
        self.D = Degradation(opt)
        self.de_temp = 0
        self.de_type = self.opt["de_type"]
        print(self.de_type)

        self.de_dict = {
            "denoise_15": 0,
            "denoise_25": 1,
            "denoise_50": 2,
            "derain": 3,
            "dehaze": 4,
            "rain13k": 5,
            "gopro": 6,
            "lol": 7,
            "real_rain": 8,
            "real_snow": 9,
            "real_fog": 10,
        }

        self._init_ids()
        self._merge_ids()

        self.crop_transform = Compose(
            [
                ToPILImage(),
                RandomCrop(opt["patch_size"]),
            ]
        )

        self.toTensor = ToTensor()

    def _init_ids(self):
        if (
            "denoise_15" in self.de_type
            or "denoise_25" in self.de_type
            or "denoise_50" in self.de_type
        ):
            self._init_clean_ids()
        if "derain" in self.de_type:
            self._init_rs_ids()
        if "dehaze" in self.de_type:
            self._init_hazy_ids()
        if "gopro" in self.de_type:
            self._init_gopro_ids()
        if "lol" in self.de_type:
            self._init_lol_ids()

        if "real_rain" in self.de_type:
            self._init_real_rain_ids()
        if "real_snow" in self.de_type:
            self._init_real_snow_ids()
        if "real_fog" in self.de_type:
            self._init_real_fog_ids()

        random.shuffle(self.de_type)

    def _init_real_snow_ids(self):
        temp_ids = []
        hazy = osp.join(self.opt["data_file_dir"], "real_snow/snow.txt")
        temp_ids += [
            osp.join(self.opt["real_snow"], 'input',  id_.strip()) for id_ in open(hazy)
        ]
        self.real_snow_ids = [
            {"clean_id": x, "de_type": self.de_dict["real_snow"]} for x in temp_ids
        ]

        self.real_snow_counter = 0

        self.num_real_snow = len(self.real_snow_ids)

        print("Total Real Snow Ids : {}".format(self.num_real_snow))

    def _init_real_rain_ids(self):
        temp_ids = []
        hazy = osp.join(self.opt["data_file_dir"], "real_rain/rain.txt")
        # 多了一层序号文件夹
        temp_ids += [
            osp.join(self.opt["real_rain"], 'input',
                     id_.split("_")[0], id_.strip())
            for id_ in open(hazy)
        ]
        self.real_rain_ids = [
            {"clean_id": x, "de_type": self.de_dict["real_rain"]} for x in temp_ids
        ]

        self.real_rain_counter = 0

        self.num_real_rain = len(self.real_rain_ids)
        print("Total Real Rain Ids : {}".format(self.num_real_rain))

    def _init_real_fog_ids(self):
        temp_ids = []
        hazy = osp.join(self.opt["data_file_dir"], "real_fog/fog.txt")
        temp_ids += [
            osp.join(self.opt["real_fog"], id_.strip()) for id_ in open(hazy)
        ]
        self.real_fog_ids = [
            {"clean_id": x, "de_type": self.de_dict["real_fog"]} for x in temp_ids
        ]

        self.real_fog_counter = 0

        self.num_real_fog = len(self.real_fog_ids)
        print("Total Real Fog Ids : {}".format(self.num_real_fog))

    def _init_clean_ids(self):
        ref_file = osp.join(self.opt["data_file_dir"],
                            "noisy/denoise_airnet.txt")
        temp_ids = []
        temp_ids += [id_.strip() for id_ in open(ref_file)]
        clean_ids = []
        name_list = os.listdir(self.opt["denoise_dir"])
        clean_ids += [
            self.opt["denoise_dir"] + id_
            for id_ in name_list
            if id_.strip() in temp_ids
        ]

        if "denoise_15" in self.de_type:
            self.s15_ids = [{"clean_id": x, "de_type": 0} for x in clean_ids]
            self.s15_ids = self.s15_ids * 3
            random.shuffle(self.s15_ids)
            self.s15_counter = 0
        if "denoise_25" in self.de_type:
            self.s25_ids = [{"clean_id": x, "de_type": 1} for x in clean_ids]
            self.s25_ids = self.s25_ids * 3
            random.shuffle(self.s25_ids)
            self.s25_counter = 0
        if "denoise_50" in self.de_type:
            self.s50_ids = [{"clean_id": x, "de_type": 2} for x in clean_ids]
            self.s50_ids = self.s50_ids * 3
            random.shuffle(self.s50_ids)
            self.s50_counter = 0

        self.num_clean = len(clean_ids)
        print("Total Denoise Ids : {}".format(self.num_clean))

    def _init_gopro_ids(self):
        temp_ids = []
        gopro = osp.join(self.opt["data_file_dir"], "gopro/gopro.txt")
        temp_ids += [self.opt["gopro_dir"] + id_.strip()
                     for id_ in open(gopro)]
        self.gopro_ids = [
            {"clean_id": x, "de_type": self.de_dict["gopro"]} for x in temp_ids
        ] * 10

        self.gopro_counter = 0

        self.num_gopro = len(self.gopro_ids)

        print("Total GoPro Ids : {}".format(self.num_gopro))

    def _init_lol_ids(self):
        temp_ids = []
        lol = osp.join(self.opt["data_file_dir"], "lol/lol.txt")
        temp_ids += [self.opt["lol_dir"] + id_.strip() for id_ in open(lol)]
        self.lol_ids = [
            {"clean_id": x, "de_type": self.de_dict["lol"]} for x in temp_ids
        ] * 50

        self.lol_counter = 0

        self.num_lol = len(self.lol_ids)
        print("Total LOL Ids : {}".format(self.num_lol))

    def _init_hazy_ids(self):
        temp_ids = []
        hazy = osp.join(self.opt["data_file_dir"], "hazy/hazy_outside.txt")
        temp_ids += [self.opt["dehaze_dir"] + id_.strip()
                     for id_ in open(hazy)]
        self.hazy_ids = [{"clean_id": x, "de_type": 4} for x in temp_ids]

        self.hazy_counter = 0

        self.num_hazy = len(self.hazy_ids)
        print("Total Hazy Ids : {}".format(self.num_hazy))

    def _init_rs_ids(self):
        temp_ids = []
        rs = osp.join(self.opt["data_file_dir"], "rainy/rainTrain.txt")
        temp_ids += [self.opt["derain_dir"] + id_.strip() for id_ in open(rs)]
        self.rs_ids = [{"clean_id": x, "de_type": 3} for x in temp_ids]
        self.rs_ids = self.rs_ids * 120

        self.rl_counter = 0
        self.num_rl = len(self.rs_ids)
        print("Total Rainy Ids : {}".format(self.num_rl))

    def _crop_patch(self, img_1, img_2):
        H = img_1.shape[0]
        W = img_1.shape[1]
        ind_H = random.randint(0, H - self.opt["patch_size"])
        ind_W = random.randint(0, W - self.opt["patch_size"])

        patch_1 = img_1[
            ind_H: ind_H + self.opt["patch_size"],
            ind_W: ind_W + self.opt["patch_size"],
        ]
        patch_2 = img_2[
            ind_H: ind_H + self.opt["patch_size"],
            ind_W: ind_W + self.opt["patch_size"],
        ]

        return patch_1, patch_2

    def _get_gt_name(self, rainy_name):
        gt_name = (
            rainy_name.split("rainy")[0] + "gt/norain-" +
            rainy_name.split("rain-")[-1]
        )
        return gt_name

    def _get_nonhazy_name(self, hazy_name):
        dir_name = hazy_name.split("synthetic")[0] + "original/"
        name = hazy_name.split("/")[-1].split("_")[0]
        suffix = "." + hazy_name.split(".")[-1]
        nonhazy_name = dir_name + name + suffix
        return nonhazy_name

    def _get_non_gopro_name(self, hazy_name):
        dir_name = hazy_name.split("input")[0] + "target/"
        name = hazy_name.split("/")[-1]
        nonhazy_name = dir_name + name
        return nonhazy_name

    def _get_non_lol_name(self, hazy_name):
        dir_name = hazy_name.split("low")[0] + "high/"
        name = hazy_name.split("/")[-1]
        nonhazy_name = dir_name + name
        return nonhazy_name

    def _get_clean_real_rain_name(self, hazy_name):
        input_file_name = osp.basename(hazy_name)
        gt_file_name = input_file_name.split("-")[0] + ".png"
        dir_name = hazy_name.replace("input/", "gt/").replace(
            input_file_name, gt_file_name
        )
        return dir_name

    def _get_clean_real_snow_name(self, hazy_name):
        dir_name = hazy_name.split("input/")[0] + "gt/"
        name = hazy_name.split("/")[-1]
        nonhazy_name = dir_name + name
        return nonhazy_name

    def _get_clean_real_fog_name(self, hazy_name):
        nonhazy_name = hazy_name.replace('input/', 'gt/')
        return nonhazy_name

    def _merge_ids(self):
        self.sample_ids = []

        if "denoise_15" in self.de_type:
            self.sample_ids += self.s15_ids
        if "denoise_25" in self.de_type:
            self.sample_ids += self.s25_ids

        if "denoise_50" in self.de_type:
            self.sample_ids += self.s50_ids

        if "derain" in self.de_type:
            self.sample_ids += self.rs_ids

        if "dehaze" in self.de_type:
            self.sample_ids += self.hazy_ids

        if "gopro" in self.de_type:
            self.sample_ids += self.gopro_ids

        if "lol" in self.de_type:
            self.sample_ids += self.lol_ids

        if "real_rain" in self.de_type:
            self.sample_ids += self.real_rain_ids
        if "real_snow" in self.de_type:
            self.sample_ids += self.real_snow_ids
        if "real_fog" in self.de_type:
            self.sample_ids += self.real_fog_ids

        print(len(self.sample_ids))

    def __getitem__(self, idx):
        sample = self.sample_ids[idx]
        de_id = sample["de_type"]

        if de_id < 3:
            if de_id == 0:
                clean_id = sample["clean_id"]
            elif de_id == 1:
                clean_id = sample["clean_id"]
            elif de_id == 2:
                clean_id = sample["clean_id"]

            clean_img = crop_img(
                np.array(Image.open(clean_id).convert("RGB")), base=16)
            clean_patch = self.crop_transform(clean_img)
            clean_patch = np.array(clean_patch)

            clean_name = clean_id.split("/")[-1].split(".")[0]

            clean_patch = random_augmentation(clean_patch)[0]

            degrad_patch = self.D.single_degrade(clean_patch, de_id)
        else:
            if de_id == 3:
                # Rain Streak Removal
                degrad_img = crop_img(
                    np.array(Image.open(sample["clean_id"]).convert("RGB")), base=16
                )
                clean_name = self._get_gt_name(sample["clean_id"])
                clean_img = crop_img(
                    np.array(Image.open(clean_name).convert("RGB")), base=16
                )
            elif de_id == 4:
                # Dehazing with SOTS outdoor training set
                degrad_img = crop_img(
                    np.array(Image.open(sample["clean_id"]).convert("RGB")), base=16
                )
                clean_name = self._get_nonhazy_name(sample["clean_id"])
                clean_img = crop_img(
                    np.array(Image.open(clean_name).convert("RGB")), base=16
                )
            elif de_id == 6:
                degrad_img = crop_img(
                    np.array(Image.open(sample["clean_id"]).convert("RGB")), base=16
                )
                clean_name = self._get_non_gopro_name(sample["clean_id"])
                clean_img = crop_img(
                    np.array(Image.open(clean_name).convert("RGB")), base=16
                )
            elif de_id == 7:
                # Dehazing with SOTS outdoor training set
                degrad_img = crop_img(
                    np.array(Image.open(sample["clean_id"]).convert("RGB")), base=16
                )
                clean_name = self._get_non_lol_name(sample["clean_id"])
                clean_img = crop_img(
                    np.array(Image.open(clean_name).convert("RGB")), base=16
                )
            elif de_id == 8:
                # Dehazing with SOTS outdoor training set
                degrad_img = crop_img(
                    np.array(Image.open(sample["clean_id"]).convert("RGB")), base=16
                )
                clean_name = self._get_clean_real_rain_name(sample["clean_id"])
                clean_img = crop_img(
                    np.array(Image.open(clean_name).convert("RGB")), base=16
                )
            elif de_id == 9:
                # Dehazing with SOTS outdoor training set
                degrad_img = crop_img(
                    np.array(Image.open(sample["clean_id"]).convert("RGB")), base=16
                )
                clean_name = self._get_clean_real_snow_name(sample["clean_id"])
                clean_img = crop_img(
                    np.array(Image.open(clean_name).convert("RGB")), base=16
                )
            elif de_id == 10:
                # Dehazing with SOTS outdoor training set
                degrad_img = crop_img(
                    np.array(Image.open(sample["clean_id"]).convert("RGB")), base=16
                )
                clean_name = self._get_clean_real_fog_name(sample["clean_id"])
                clean_img = crop_img(
                    np.array(Image.open(clean_name).convert("RGB")), base=16
                )
            degrad_patch, clean_patch = random_augmentation(
                *self._crop_patch(degrad_img, clean_img)
            )

        clean_patch = self.toTensor(clean_patch)
        degrad_patch = self.toTensor(degrad_patch)

        res = {
            "lq": degrad_patch,
            "gt": clean_patch,
            "img_type_word": de_id
        }

        return res

    def __len__(self):
        return len(self.sample_ids)

# @DATASET_REGISTRY.register()
# class PromptTrainDataset(Dataset):
#     def __init__(self, opt):
#         super(PromptTrainDataset, self).__init__()
#         self.opt = opt
#         self.rs_ids = []
#         self.hazy_ids = []
#         self.D = Degradation(opt)
#         self.de_temp = 0
#         self.de_type = self.opt["de_type"]
#         print(self.de_type)

#         self.de_dict = {
#             "denoise_15": 0,
#             "denoise_25": 1,
#             "denoise_50": 2,
#             "derain": 3,
#             "dehaze": 4,
#             "rain13k": 5,
#             "gopro": 6,
#             "lol": 7,
#             "real_rain": 8,
#             "real_snow": 9,
#             "real_fog": 10,
#         }

#         self._init_ids()
#         self._merge_ids()

#         self.crop_transform = Compose(
#             [
#                 ToPILImage(),
#                 RandomCrop(opt["patch_size"]),
#             ]
#         )

#         self.toTensor = ToTensor()

#     def _init_ids(self):
#         if (
#             "denoise_15" in self.de_type
#             or "denoise_25" in self.de_type
#             or "denoise_50" in self.de_type
#         ):
#             self._init_clean_ids()
#         if "derain" in self.de_type:
#             self._init_rs_ids()
#         if "dehaze" in self.de_type:
#             self._init_hazy_ids()
#         if "gopro" in self.de_type:
#             self._init_gopro_ids()
#         if "lol" in self.de_type:
#             self._init_lol_ids()

#         if "real_rain" in self.de_type:
#             self._init_real_rain_ids()
#         if "real_snow" in self.de_type:
#             self._init_real_snow_ids()
#         if "real_fog" in self.de_type:
#             self._init_real_fog_ids()

#         random.shuffle(self.de_type)

#     def _init_real_snow_ids(self):
#         temp_ids = []
#         hazy = osp.join(self.opt["data_file_dir"], "real_snow/snow.txt")
#         temp_ids += [
#             osp.join(self.opt["real_snow"], 'input',  id_.strip()) for id_ in open(hazy)
#         ]
#         self.real_snow_ids = [
#             {"clean_id": x, "de_type": self.de_dict["real_snow"]} for x in temp_ids
#         ]

#         self.real_snow_counter = 0

#         self.num_real_snow = len(self.real_snow_ids)

#         print("Total Real Snow Ids : {}".format(self.num_real_snow))

#     def _init_real_rain_ids(self):
#         temp_ids = []
#         hazy = osp.join(self.opt["data_file_dir"], "real_rain/rain.txt")
#         # 多了一层序号文件夹
#         temp_ids += [
#             osp.join(self.opt["real_rain"], 'input',
#                      id_.split("_")[0], id_.strip())
#             for id_ in open(hazy)
#         ]
#         self.real_rain_ids = [
#             {"clean_id": x, "de_type": self.de_dict["real_rain"]} for x in temp_ids
#         ]

#         self.real_rain_counter = 0

#         self.num_real_rain = len(self.real_rain_ids)
#         print("Total Real Rain Ids : {}".format(self.num_real_rain))

#     def _init_real_fog_ids(self):
#         temp_ids = []
#         hazy = osp.join(self.opt["data_file_dir"], "real_fog/fog.txt")
#         temp_ids += [
#             osp.join(self.opt["real_fog"], id_.strip()) for id_ in open(hazy)
#         ]
#         self.real_fog_ids = [
#             {"clean_id": x, "de_type": self.de_dict["real_fog"]} for x in temp_ids
#         ]

#         self.real_fog_counter = 0

#         self.num_real_fog = len(self.real_fog_ids)
#         print("Total Real Fog Ids : {}".format(self.num_real_fog))

#     def _init_clean_ids(self):
#         ref_file = osp.join(self.opt["data_file_dir"],
#                             "noisy/denoise_airnet.txt")
#         temp_ids = []
#         temp_ids += [id_.strip() for id_ in open(ref_file)]
#         clean_ids = []
#         name_list = os.listdir(self.opt["denoise_dir"])
#         clean_ids += [
#             self.opt["denoise_dir"] + id_
#             for id_ in name_list
#             if id_.strip() in temp_ids
#         ]

#         if "denoise_15" in self.de_type:
#             self.s15_ids = [{"clean_id": x, "de_type": 0} for x in clean_ids]
#             self.s15_ids = self.s15_ids * 3
#             random.shuffle(self.s15_ids)
#             self.s15_counter = 0
#         if "denoise_25" in self.de_type:
#             self.s25_ids = [{"clean_id": x, "de_type": 1} for x in clean_ids]
#             self.s25_ids = self.s25_ids * 3
#             random.shuffle(self.s25_ids)
#             self.s25_counter = 0
#         if "denoise_50" in self.de_type:
#             self.s50_ids = [{"clean_id": x, "de_type": 2} for x in clean_ids]
#             self.s50_ids = self.s50_ids * 3
#             random.shuffle(self.s50_ids)
#             self.s50_counter = 0

#         self.num_clean = len(clean_ids)
#         print("Total Denoise Ids : {}".format(self.num_clean))

#     def _init_gopro_ids(self):
#         temp_ids = []
#         gopro = osp.join(self.opt["data_file_dir"], "gopro/gopro.txt")
#         temp_ids += [self.opt["gopro_dir"] + id_.strip()
#                      for id_ in open(gopro)]
#         self.gopro_ids = [
#             {"clean_id": x, "de_type": self.de_dict["gopro"]} for x in temp_ids
#         ] * 10

#         self.gopro_counter = 0

#         self.num_gopro = len(self.gopro_ids)

#         print("Total GoPro Ids : {}".format(self.num_gopro))

#     def _init_lol_ids(self):
#         temp_ids = []
#         lol = osp.join(self.opt["data_file_dir"], "lol/lol.txt")
#         temp_ids += [self.opt["lol_dir"] + id_.strip() for id_ in open(lol)]
#         self.lol_ids = [
#             {"clean_id": x, "de_type": self.de_dict["lol"]} for x in temp_ids
#         ] * 50

#         self.lol_counter = 0

#         self.num_lol = len(self.lol_ids)
#         print("Total LOL Ids : {}".format(self.num_lol))

#     def _init_hazy_ids(self):
#         temp_ids = []
#         hazy = osp.join(self.opt["data_file_dir"], "hazy/hazy_outside.txt")
#         temp_ids += [self.opt["dehaze_dir"] + id_.strip()
#                      for id_ in open(hazy)]
#         self.hazy_ids = [{"clean_id": x, "de_type": 4} for x in temp_ids]

#         self.hazy_counter = 0

#         self.num_hazy = len(self.hazy_ids)
#         print("Total Hazy Ids : {}".format(self.num_hazy))

#     def _init_rs_ids(self):
#         temp_ids = []
#         rs = osp.join(self.opt["data_file_dir"], "rainy/rainTrain.txt")
#         temp_ids += [self.opt["derain_dir"] + id_.strip() for id_ in open(rs)]
#         self.rs_ids = [{"clean_id": x, "de_type": 3} for x in temp_ids]
#         self.rs_ids = self.rs_ids * 120

#         self.rl_counter = 0
#         self.num_rl = len(self.rs_ids)
#         print("Total Rainy Ids : {}".format(self.num_rl))

#     def _crop_patch(self, img_1, img_2):
#         H = img_1.shape[0]
#         W = img_1.shape[1]
#         ind_H = random.randint(0, H - self.opt["patch_size"])
#         ind_W = random.randint(0, W - self.opt["patch_size"])

#         patch_1 = img_1[
#             ind_H: ind_H + self.opt["patch_size"],
#             ind_W: ind_W + self.opt["patch_size"],
#         ]
#         patch_2 = img_2[
#             ind_H: ind_H + self.opt["patch_size"],
#             ind_W: ind_W + self.opt["patch_size"],
#         ]

#         return patch_1, patch_2

#     def _get_gt_name(self, rainy_name):
#         gt_name = (
#             rainy_name.split("rainy")[0] + "gt/norain-" +
#             rainy_name.split("rain-")[-1]
#         )
#         return gt_name

#     def _get_nonhazy_name(self, hazy_name):
#         dir_name = hazy_name.split("synthetic")[0] + "original/"
#         name = hazy_name.split("/")[-1].split("_")[0]
#         suffix = "." + hazy_name.split(".")[-1]
#         nonhazy_name = dir_name + name + suffix
#         return nonhazy_name

#     def _get_non_gopro_name(self, hazy_name):
#         dir_name = hazy_name.split("input")[0] #+ "target/"
#         name = hazy_name.split("/")[-1]
#         nonhazy_name = dir_name #+ name
#         #print(f'nonhazy_name_dehaze{nonhazy_name}')
#         return nonhazy_name

#     def _get_non_lol_name(self, hazy_name):
#         dir_name = hazy_name.split("low")[0] #+ "high/"
#         name = hazy_name.split("/")[-1]
#         nonhazy_name = dir_name #+ name
#         #print(f'nonhazy_name_lol:{nonhazy_name},dir_name:{dir_name},name:{name}')
#         return nonhazy_name

#     def _get_clean_real_rain_name(self, hazy_name):
#         input_file_name = osp.basename(hazy_name)
#         gt_file_name = input_file_name.split("-")[0] + ".png"
#         dir_name = hazy_name.replace("input/", "gt/").replace(
#             input_file_name, gt_file_name
#         )
#         return dir_name

#     def _get_clean_real_snow_name(self, hazy_name):
#         dir_name = hazy_name.split("input/")[0] + "gt/"
#         name = hazy_name.split("/")[-1]
#         nonhazy_name = dir_name + name
#         return nonhazy_name

#     def _get_clean_real_fog_name(self, hazy_name):
#         nonhazy_name = hazy_name.replace('input/', 'gt/')
#         return nonhazy_name

#     def _merge_ids(self):
#         self.sample_ids = []

#         if "denoise_15" in self.de_type:
#             self.sample_ids += self.s15_ids
#         if "denoise_25" in self.de_type:
#             self.sample_ids += self.s25_ids

#         if "denoise_50" in self.de_type:
#             self.sample_ids += self.s50_ids

#         if "derain" in self.de_type:
#             self.sample_ids += self.rs_ids

#         if "dehaze" in self.de_type:
#             self.sample_ids += self.hazy_ids

#         if "gopro" in self.de_type:
#             self.sample_ids += self.gopro_ids

#         if "lol" in self.de_type:
#             self.sample_ids += self.lol_ids

#         if "real_rain" in self.de_type:
#             self.sample_ids += self.real_rain_ids
#         if "real_snow" in self.de_type:
#             self.sample_ids += self.real_snow_ids
#         if "real_fog" in self.de_type:
#             self.sample_ids += self.real_fog_ids

#         print(len(self.sample_ids))

#     def __getitem__(self, idx):
#         sample = self.sample_ids[idx]
#         de_id = sample["de_type"]

#         if de_id < 3:
#             if de_id == 0:
#                 clean_id = sample["clean_id"]
#             elif de_id == 1:
#                 clean_id = sample["clean_id"]
#             elif de_id == 2:
#                 clean_id = sample["clean_id"]

#             clean_img = crop_img(
#                 np.array(Image.open(clean_id).convert("RGB")), base=16)
#             clean_patch = self.crop_transform(clean_img)
#             clean_patch = np.array(clean_patch)

#             clean_name = clean_id.split("/")[-1].split(".")[0]

#             clean_patch = random_augmentation(clean_patch)[0]

#             degrad_patch = self.D.single_degrade(clean_patch, de_id)
#         else:
#             if de_id == 3:
#                 # Rain Streak Removal
#                 degrad_img = crop_img(
#                     np.array(Image.open(sample["clean_id"]).convert("RGB")), base=16
#                 )
#                 clean_name = self._get_gt_name(sample["clean_id"])
#                 clean_img = crop_img(
#                     np.array(Image.open(clean_name).convert("RGB")), base=16
#                 )
#             elif de_id == 4:
#                 # Dehazing with SOTS outdoor training set
#                 degrad_img = crop_img(
#                     np.array(Image.open(sample["clean_id"]).convert("RGB")), base=16
#                 )
#                 clean_name = self._get_nonhazy_name(sample["clean_id"])
#                 clean_img = crop_img(
#                     np.array(Image.open(clean_name).convert("RGB")), base=16
#                 )
#             elif de_id == 6:
#                 degrad_img = crop_img(
#                     np.array(Image.open(sample["clean_id"]).convert("RGB")), base=16
#                 )
#                 clean_name = self._get_non_gopro_name(sample["clean_id"])
#                 clean_img = crop_img(
#                     np.array(Image.open(clean_name).convert("RGB")), base=16
#                 )
#             elif de_id == 7:
#                 # Dehazing with SOTS outdoor training set
#                 degrad_img = crop_img(
#                     np.array(Image.open(sample["clean_id"]).convert("RGB")), base=16
#                 )
#                 clean_name = self._get_non_lol_name(sample["clean_id"])
#                 clean_img = crop_img(
#                     np.array(Image.open(clean_name).convert("RGB")), base=16
#                 )
#             elif de_id == 8:
#                 # Dehazing with SOTS outdoor training set
#                 degrad_img = crop_img(
#                     np.array(Image.open(sample["clean_id"]).convert("RGB")), base=16
#                 )
#                 clean_name = self._get_clean_real_rain_name(sample["clean_id"])
#                 clean_img = crop_img(
#                     np.array(Image.open(clean_name).convert("RGB")), base=16
#                 )
#             elif de_id == 9:
#                 # Dehazing with SOTS outdoor training set
#                 degrad_img = crop_img(
#                     np.array(Image.open(sample["clean_id"]).convert("RGB")), base=16
#                 )
#                 clean_name = self._get_clean_real_snow_name(sample["clean_id"])
#                 clean_img = crop_img(
#                     np.array(Image.open(clean_name).convert("RGB")), base=16
#                 )
#             elif de_id == 10:
#                 # Dehazing with SOTS outdoor training set
#                 degrad_img = crop_img(
#                     np.array(Image.open(sample["clean_id"]).convert("RGB")), base=16
#                 )
#                 clean_name = self._get_clean_real_fog_name(sample["clean_id"])
#                 clean_img = crop_img(
#                     np.array(Image.open(clean_name).convert("RGB")), base=16
#                 )
#             degrad_patch, clean_patch = random_augmentation(
#                 *self._crop_patch(degrad_img, clean_img)
#             )

#         clean_patch = self.toTensor(clean_patch)
#         degrad_patch = self.toTensor(degrad_patch)

#         res = {
#             "lq": degrad_patch,
#             "gt": clean_patch,
#             "img_type_word": de_id
#         }

#         return res

#     def __len__(self):
#         return len(self.sample_ids)
