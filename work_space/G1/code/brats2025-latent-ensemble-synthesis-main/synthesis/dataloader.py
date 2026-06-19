import os
import pandas as pd
import numpy as np

import torch

import synthesis.utils as utils

TRAIN_MODALITIES = ("t1n", "t1c", "t2w", "t2f")
INFERENCE_MODALITIES = ("t1n", "t1c", "t2f")
INFERENCE_SPLITS = {"inference", "infer", "predict", "generation"}


def has_value(value):
    if pd.isna(value):
        return False
    if isinstance(value, str):
        return value.strip() != ""
    return True


def row_has_modalities(row, modalities):
    return all(has_value(row.get(modality, "")) for modality in modalities)


def is_inference_split(split):
    if split is None:
        return False
    return str(split).strip().lower() in INFERENCE_SPLITS


class PathsLoader:
    def __init__(self, df_path, data_path, load_latents=True, load_seg=False, attmasks_path=None, attmasks_shapes_list=None, load_org_img=False):
        self.df_path = pd.read_csv(df_path)
        self.data_path = data_path
        self.data_root = os.path.abspath(os.path.join(data_path, os.pardir))
        self.load_latents = load_latents
        self.load_seg = load_seg
        self.attmasks_path = attmasks_path
        self.attmasks_shapes_list = attmasks_shapes_list
        self.load_org_img = load_org_img

    def create_latent_name(self, img_name):
        if self.load_latents:
            return f"{img_name}_latent.npy"
        else:
            return f"{img_name}_rec.nii.gz"

    def resolve_seg_path(self, s_id, row):
        seg_value = row.get("seg", "")
        if has_value(seg_value):
            seg_path = str(seg_value).strip()
            if os.path.isabs(seg_path) and os.path.exists(seg_path):
                return seg_path
            for base_dir in ("input", "input_inference"):
                candidate = os.path.join(self.data_root, base_dir, s_id, seg_path)
                if os.path.exists(candidate):
                    return candidate

        for base_dir in ("input", "input_inference"):
            subject_dir = os.path.join(self.data_root, base_dir, s_id)
            if not os.path.isdir(subject_dir):
                continue
            for file_name in sorted(os.listdir(subject_dir)):
                if file_name.endswith((".nii.gz", ".nii")) and "-seg" in file_name:
                    return os.path.join(subject_dir, file_name)
        return ""

    def get_data_by_split(self, split="train"):
        complete_df = self.df_path.copy()
        # obtain train
        if split is not None and "split" in complete_df.columns:
            complete_df = complete_df[complete_df["split"] == split]

        inference_mode = is_inference_split(split)
        required_modalities = INFERENCE_MODALITIES if inference_mode else TRAIN_MODALITIES
        require_seg = self.load_seg or inference_mode
        instances = []
        for i, row in complete_df.iterrows():
            if not row_has_modalities(row, required_modalities):
                continue

            s_id = row["id"]
            seg_path = ""
            if require_seg:
                seg_path = self.resolve_seg_path(s_id, row)
                if not seg_path:
                    continue

            if self.load_org_img:
                modality_file_names = {modality: row[modality] for modality in required_modalities}
            else:
                modality_file_names = {}
                for modality in required_modalities:
                    modality_path = os.path.basename(row[modality]).split(".")[0]
                    modality_file_names[modality] = self.create_latent_name(modality_path)

            _instance = {}
            for modality in required_modalities:
                _instance[modality] = os.path.join(self.data_path, s_id, modality_file_names[modality])
            if require_seg:
                _instance["seg"] = seg_path

            if self.attmasks_path is not None and self.attmasks_shapes_list is not None:
                __attmask_path_names = []
                for i, attmask_shape in enumerate(self.attmasks_shapes_list):
                    __attmask_path_names.append(f"{self.attmasks_path}/{s_id}/{s_id}_attmask_{'_'.join(str(x) for x in attmask_shape)}.npy")
                _instance["attmasks"] = __attmask_path_names

            instances.append(_instance)

        return instances

    def get_inference_by_split(self, split="inference"):
        return self.get_data_by_split(split=split)




class PrepareDataset(torch.utils.data.Dataset):
    def get_modality_index(self, modality_name):
        modalities = ["t1n", "t1c", "t2w", "t2f"]
        if modality_name in modalities:
            return modalities.index(modality_name)
        else:
            raise ValueError(f"Modality name {modality_name} not found. Available options are: {modalities}")

    def __init__(self,
                 df_path,
                 data_path,
                 split = "train",
                 to_modality_name = "t2w",
                 mode = "3-to-1",
                 from_modality_name = None,
                 seed = 42,
                 max_images = None,
                 new_shape = None, #(64, 64, 48) # 64x64x48
                 load_only_latents=True,
                 n_latent_channels=4,
                 load_seg=False,
                attmasks_path=None, # path to the attention masks
                 attmasks_shapes_list=None, # list of shapes for the attention masks
                 path_histograms=None,
                 start_index=None, # index to start the dataset from (useful for debugging or resuming training)
                 load_org_img=False, # if True, load the original images instead of the latents
                 ):



        # obtain data paths
        paths_loader = PathsLoader(df_path, data_path, load_latents=True, load_seg=load_seg,
                                   attmasks_path=attmasks_path, attmasks_shapes_list=attmasks_shapes_list)
        self.paths_list = paths_loader.get_data_by_split(split=split)

        self.split = split
        self.new_shape = new_shape
        self.n_latent_channels = n_latent_channels

        self.mode = mode
        self.to_modality_name = to_modality_name
        self.from_modality_name = from_modality_name

        # load segmentation if needed
        self.load_seg = load_seg
        self.load_attmasks = attmasks_path is not None and attmasks_shapes_list is not None
        self.attmasks_shapes_list = attmasks_shapes_list

        self.load_org_img = load_org_img

        # histogram path
        self.path_histograms = path_histograms

        # T2W is always the target modality
        self.to_modality_index = self.get_modality_index(to_modality_name)

        # From modality: for 1-to-1 mode, randomly pick one of the available modalities (not T2W)
        if from_modality_name is not None:
            self.from_modality_index = self.get_modality_index(from_modality_name)
        else:
            self.gen_from_modality = torch.Generator().manual_seed(seed)

        if start_index is not None:
            self.paths_list = self.paths_list[start_index:]
        if max_images is not None:
            self.paths_list = self.paths_list[:max_images]
            # print(f"Limiting to {max_images} images")

        # number of latent in the folder
        self.num_instances = len(self.paths_list)
        self._length = self.num_instances

        print(f"Number of {split} images: {self.num_instances}")


        # obtain images data paths
        self.load_only_latents = load_only_latents
        if not self.load_only_latents:
            paths_loader_imgs = PathsLoader(df_path, data_path, load_latents=False, load_org_img=load_org_img)
            self.paths_list_imgs = paths_loader_imgs.get_data_by_split(split=split)
            if start_index is not None:
                self.paths_list_imgs = self.paths_list_imgs[start_index:]

            self.paths_list_imgs =self.paths_list_imgs[:self.num_instances]

        if mode == "4n-to-4":
            self.gen_noisy_modality_lt = torch.Generator().manual_seed(seed)
            self.gen_noisy_modality_img = torch.Generator().manual_seed(seed)

        # Crear un índice para acceder rápidamente por s_id
        self.id_to_index = {}
        for idx, paths in enumerate(self.paths_list):
            s_id = os.path.basename(os.path.dirname(paths["t1n"]))
            self.id_to_index[s_id] = idx




    def __len__(self):
        return self._length

    def __getitem__(self, index):
        # dictionary to store the image and the prompt
        sample = {}
        # select latent path name from the list
        instance_latents = self.paths_list[index % self.num_instances]

        # load lattents
        latents_list = [np.load(instance_latents["t1n"]).squeeze(0),
                            np.load(instance_latents["t1c"]).squeeze(0),
                            np.load(instance_latents["t2w"]).squeeze(0),
                            np.load(instance_latents["t2f"]).squeeze(0)]

        s_id = os.path.basename(os.path.dirname(instance_latents["t1n"]))
        sample["s_id"] = s_id

        # change lattents shape
        if self.new_shape is not None:
            __latents_list_resize = []
            for lt in latents_list:
                __lt_ch_list = [utils.resize_center_crop_pad(lt_ch, self.new_shape)[0] for lt_ch in lt]
                __latents_list_resize.append(np.stack(__lt_ch_list, axis=0))
            latents_list = __latents_list_resize


        ## ---- TO MODALITY ---- ##
        # T2W is always the missing modality
        __to_modality_index = self.to_modality_index
        __to_modality_one_hot = torch.zeros((4,))
        __to_modality_one_hot[__to_modality_index] = 1.0


        if self.mode in ["1-to-1", "3-to-1"]:
            # create the to_modality latents
            __to_modality_latents = latents_list[__to_modality_index]
            __to_modality_latents = torch.from_numpy(__to_modality_latents)

            ## ---- FROM MODALITY ---- ##
            # create the from_modality latents tensor / one hot vector
            if self.mode == "1-to-1":
                # obtain the from_modality index
                if self.from_modality_name is not None:
                    __from_modality_index = self.from_modality_index
                else:
                    __possible_options = [i for i in range(4) if i != __to_modality_index]
                    __from_modality_index = __possible_options[torch.randint(low=0, high=3, size=(1,), generator=self.gen_from_modality).item()]

                # obtrain the from_modality latents
                __from_modality_latents = torch.from_numpy(latents_list[__from_modality_index])

                # create the from_modality one hot vector
                __from_modality_one_hot = torch.zeros((4,))
                __from_modality_one_hot[__from_modality_index] = 1.0

            elif self.mode == "3-to-1":
                # concatenate all the modalities but not the one we are going to use for the to_modality
                __from_modality_latents = np.concatenate([latents_list[i] for i in range(4) if i != __to_modality_index], axis=0)
                __from_modality_latents = torch.from_numpy(__from_modality_latents)
                # create the from_modality one hot vector
                __from_modality_one_hot = torch.ones((4,))
                __from_modality_one_hot[__to_modality_index] = 0.0

        elif self.mode in ["4-to-4", "4n-to-4", "4b-to-4"]: # 4n = 4 channels (one noised), 4b = 4 channels (one black)
            __to_modality_latents = np.concatenate(latents_list, axis=0)
            __from_modality_latents = __to_modality_latents.copy()

            __to_modality_latents = torch.from_numpy(__to_modality_latents)
            __from_modality_latents = torch.from_numpy(__from_modality_latents)
            if self.mode == "4b-to-4":
                __from_modality_latents[self.n_latent_channels*__to_modality_index:self.n_latent_channels*(__to_modality_index+1)] = 0.0
            elif self.mode == "4n-to-4":
                __noise_lt = torch.randn(latents_list[0].shape, device="cpu", generator=self.gen_noisy_modality_lt)
                __from_modality_latents[self.n_latent_channels*__to_modality_index:self.n_latent_channels*(__to_modality_index+1)] = __noise_lt

            # create the from_modality one hot vector
            __from_modality_one_hot = torch.ones((4,))
            __from_modality_one_hot[__to_modality_index] = 0.0


        if not self.load_only_latents:
            instance_img = self.paths_list_imgs[index % self.num_instances]
            images_list = [utils.load_nifti(instance_img["t1n"])[0],
                            utils.load_nifti(instance_img["t1c"])[0],
                            utils.load_nifti(instance_img["t2w"])[0],
                            utils.load_nifti(instance_img["t2f"])[0]]

            if self.load_org_img:
                images_list = [utils.robust_normalize(utils.resize_center_crop_pad(img, (256,256,160))[0]) for img in images_list]

            ## ---- TO MODALITY ---- ##
            if self.mode in ["1-to-1", "3-to-1"]:
                __to_modality_images = np.expand_dims(images_list[__to_modality_index], axis=0)
                if self.mode == "1-to-1":
                    __from_modality_images = np.expand_dims(images_list[__from_modality_index], axis=0)
                    # unsqueeze to add the channel dimension
                elif self.mode == "3-to-1":
                    __from_modality_images = np.stack([images_list[i] for i in range(4) if i != __to_modality_index], axis=0)

                __to_modality_images = torch.from_numpy(__to_modality_images)
                __from_modality_images = torch.from_numpy(__from_modality_images)

            elif self.mode in ["4-to-4", "4n-to-4", "4b-to-4"]:
                __to_modality_images = np.stack(images_list, axis=0)
                __from_modality_images = __to_modality_images.copy()

                __to_modality_images = torch.from_numpy(__to_modality_images)
                __from_modality_images = torch.from_numpy(__from_modality_images)

                if self.mode == "4b-to-4":
                    __from_modality_images[__to_modality_index] = 0.0
                elif self.mode == "4n-to-4":
                    __noise_img = torch.randn(images_list[0].shape, device="cpu", generator=self.gen_noisy_modality_img)
                    __from_modality_images[__to_modality_index] = __noise_img

        if self.load_seg:
            # load the segmentation
            seg = utils.load_nifti(instance_latents["seg"])[0]
            seg = utils.resize_center_crop_pad(seg, np.array(latents_list[0].squeeze()[0].shape)*4)[0]
            seg = np.expand_dims(seg, axis=0)
            sample["segmentation"] = torch.from_numpy(seg)

        if self.load_attmasks:
            # load the attention masks
            attmasks_paths = instance_latents["attmasks"]
            for i, attmask_path in enumerate(attmasks_paths):
                attmask = np.load(attmask_path)
                # not implemente for now, but we can resize the attention masks to the new shape
                # if self.new_shape is not None:
                #     attmask = utils.resize_center_crop_pad(attmask, self.new_shape)[0]
                attmask = np.expand_dims(attmask, axis=0)  # add channel dimension
                sample[f"attmasks_{'_'.join(str(x) for x in self.attmasks_shapes_list[i])}"] = torch.from_numpy(attmask)

        if self.path_histograms is not None:
            # load the histogram for the to modality
            s_histograms = np.load(f"{self.path_histograms}/{s_id}_hist.npy")
            __to_modality_histograms = s_histograms[self.to_modality_index]
            __to_modality_histograms = torch.from_numpy(__to_modality_histograms).float()

            if self.mode in ["1-to-1"]:
                # load the histogram for the from modality
                __from_modality_histograms = s_histograms[self.from_modality_index]
                __from_modality_histograms = torch.from_numpy(__from_modality_histograms).float().unsqueeze(0)
            else:
                # concatenate the histograms of the modalities used for the from modality
                __from_modality_histograms = [s_histograms[i] for i in range(4) if i != self.to_modality_index]
                __from_modality_histograms = np.stack(__from_modality_histograms, axis=0)
                __from_modality_histograms = torch.from_numpy(__from_modality_histograms).float()



            # if self.mode in ["1-to-1", "3-to-1"]:
            #     # load the histogram for the from modality
            #     if self.mode == "1-to-1":
            #         __from_modality_histograms = s_histograms[self.from_modality_index]
            #         __from_modality_histograms = torch.from_numpy(__from_modality_histograms).float()
            #     elif self.mode == "3-to-1":
            #         # concatenate the histograms of the modalities used for the from modality
            #         __from_modality_histograms = [s_histograms[i] for i in range(4) if i != self.to_modality_index]
            #         __from_modality_histograms = np.concatenate(__from_modality_histograms, axis=0)
            #         __from_modality_histograms = torch.from_numpy(__from_modality_histograms).float()

            # elif self.mode in ["4-to-4", "4n-to-4", "4b-to-4"]:
            #     # concatenate the histograms of all the modalities
            #     __from_modality_histograms = np.concatenate(s_histograms, axis=0)
            #     __from_modality_histograms = torch.from_numpy(__from_modality_histograms).float()
            #     if self.mode == "4b-to-4": # NOT WORKING
            #         __from_modality_histograms[self.to_modality_index] = 0.0
            #     elif self.mode == "4n-to-4":
            #         __from_modality_histograms[self.to_modality_index] = torch.randn_like(__from_modality_histograms[self.to_modality_index], generator=self.gen_noisy_modality_img)


        sample["from_modality_latents"] = __from_modality_latents
        sample["to_modality_latents"] = __to_modality_latents
        sample["from_modality_one_hot"] = __from_modality_one_hot
        sample["to_modality_one_hot"] = __to_modality_one_hot

        if not self.load_only_latents:
            sample["from_modality_images"] = __from_modality_images
            sample["to_modality_images"] = __to_modality_images

        if self.path_histograms is not None:
            sample["to_modality_histograms"] = __to_modality_histograms
            sample["from_modality_histograms"] = __from_modality_histograms

        return sample






def collate_fn(samples):
    from_modality_latents = torch.stack([example["from_modality_latents"] for example in samples])
    from_modality_latents = from_modality_latents.to(memory_format=torch.contiguous_format).float()
    to_modality_latents = torch.stack([example["to_modality_latents"] for example in samples])
    to_modality_latents = to_modality_latents.to(memory_format=torch.contiguous_format).float()
    from_modality_one_hot = torch.stack([example["from_modality_one_hot"] for example in samples])
    from_modality_one_hot = from_modality_one_hot.to(memory_format=torch.contiguous_format).float()
    to_modality_one_hot = torch.stack([example["to_modality_one_hot"] for example in samples])
    to_modality_one_hot = to_modality_one_hot.to(memory_format=torch.contiguous_format).float()

    # alsor return the subject id
    s_id = [example["s_id"] for example in samples]
    # s_id = torch.tensor(s_id)

    samples_dict = {
        "from_modality_latents": from_modality_latents,
        "to_modality_latents": to_modality_latents,
        "from_modality_one_hot": from_modality_one_hot,
        "to_modality_one_hot": to_modality_one_hot,
        "s_id": s_id
    }

    if "from_modality_images" in samples[0]:
        from_modality_images = torch.stack([example["from_modality_images"] for example in samples])
        from_modality_images = from_modality_images.to(memory_format=torch.contiguous_format).float()
        to_modality_images = torch.stack([example["to_modality_images"] for example in samples])
        to_modality_images = to_modality_images.to(memory_format=torch.contiguous_format).float()

        samples_dict["from_modality_images"] = from_modality_images
        samples_dict["to_modality_images"] = to_modality_images

    if "segmentation" in samples[0]:
        segmentation = torch.stack([example["segmentation"] for example in samples])
        segmentation = segmentation.to(memory_format=torch.contiguous_format).float()
        samples_dict["segmentation"] = segmentation

    # stack all the attention masks with the same shape
    # find all teh key containing "attmasks_"
    attmasks_keys = [key for key in samples[0].keys() if key.startswith("attmasks_")]
    for key in attmasks_keys:
        attmasks = torch.stack([example[key] for example in samples])
        attmasks = attmasks.to(memory_format=torch.contiguous_format).float()
        samples_dict[key] = attmasks

    if "to_modality_histograms" in samples[0]:
        to_modality_histogram = torch.stack([example["to_modality_histograms"] for example in samples])
        to_modality_histogram = to_modality_histogram.to(memory_format=torch.contiguous_format).float()
        samples_dict["to_modality_histograms"] = to_modality_histogram
    if "from_modality_histograms" in samples[0]:
        from_modality_histogram = torch.stack([example["from_modality_histograms"] for example in samples])
        from_modality_histogram = from_modality_histogram.to(memory_format=torch.contiguous_format).float()
        samples_dict["from_modality_histograms"] = from_modality_histogram

    return samples_dict
