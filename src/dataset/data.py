import os
import json
import random

from transformers import CLIPImageProcessor
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms.functional import crop
from src.dataset.data_utils import RandomResizeCrop, ToTensor, EnhanceText, PairCompose, mask2bbox


imagenet_templates_small = [
    "a photo of a {}",
    "a rendering of a {}",
    "a cropped photo of the {}",
    "the photo of a {}",
    "a photo of a clean {}",
    "a photo of a dirty {}",
    "a dark photo of the {}",
    "a photo of my {}",
    "a photo of the cool {}",
    "a close-up photo of a {}",
    "a bright photo of the {}",
    "a cropped photo of a {}",
    "a photo of the {}",
    "a good photo of the {}",
    "a photo of one {}",
    "a close-up photo of the {}",
    "a rendition of the {}",
    "a photo of the clean {}",
    "a rendition of a {}",
    "a photo of a nice {}",
    "a good photo of a {}",
    "a photo of the nice {}",
    "a photo of the small {}",
    "a photo of the weird {}",
    "a photo of the large {}",
    "a photo of a cool {}",
    "a photo of a small {}",
]

class MSDiffusionDataset(Dataset):
    def __init__(self, root, size=1024):
        self.video_root = os.path.join(root, "videos")
        self.mask_root = os.path.join(root, "masks")
        self.data = os.listdir(self.video_root)
        self.prompts = json.load(open(os.path.join(root, "prompts.json"), "r"))
        self.preprocessor = CLIPImageProcessor()
        self.transforms = PairCompose(
            [
                RandomResizeCrop(size=size, crop_p=0.5),
                ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        video_name = self.data[index]
        prompt = self.prompts[video_name]["prompt"]
        video_path = os.path.join(self.video_root, video_name)
        
        if "entity" in self.prompts[video_name]:
            is_single = True
            entities = self.prompts[video_name]["entity"]
            concept_image_paths = [os.path.join(video_path, "concept.png")]
        else:
            is_single = False
            entities = self.prompts[video_name]["entities"]
            concept_image_paths = [os.path.join(video_path, f"concept{i}.png") for i in range(2)]
        predicate = self.prompts[video_name].get("predicate", None)
        
        concept_images = [Image.open(image_path).convert("RGB").resize((512, 512)) for image_path in concept_image_paths]
        concept_images = self.preprocessor(concept_images, return_tensors="pt").pixel_values    # (n, 3, 224, 224)
        
        end_frame_path = os.path.join(video_path, "motion.png")
        if is_single:
            end_frame_masks_path = [end_frame_path.replace("videos", "masks")]
        else:
            end_frame_masks_path = [end_frame_path.replace("videos", "masks").replace("motion", f"motion{i}") for i in range(2)]
            
        end_frame = Image.open(end_frame_path).convert("RGB")
        original_size = end_frame.size[::-1]
        end_frame_masks = [Image.open(end_frame_mask_path).convert("L") for end_frame_mask_path in end_frame_masks_path]
        
        end_frame, kwargs = self.transforms(img=end_frame, mask=end_frame_masks)
        masks = kwargs["mask"] >= 0.5    # (n, 128, 128)
        res_h, res_w = masks.shape[1], masks.shape[2]
        bboxes = []
        for mask in masks:
            x1, y1, x2, y2 = torch.tensor(mask2bbox(mask))
            bbox = torch.tensor([x1 / res_w, y1 / res_h, x2 / res_w, y2 / res_h])
            bboxes.append(bbox)
        bboxes = torch.stack(bboxes, dim=0)     # (n, 4)
        
        ret = dict(
            images=end_frame,
            concept_images=concept_images,
            bboxes=bboxes,
            prompts=prompt,
            entities=entities,
            predicate=predicate,
            original_sizes=original_size,
            crop_top_lefts=(0, 0)
        )
        
        return ret

class RelationBoothDataset(Dataset):
    def __init__(self, root, size=1024):
        self.video_root = os.path.join(root, "videos")
        self.mask_root = os.path.join(root, "masks")
        self.data = os.listdir(self.video_root)
        self.prompts = json.load(open(os.path.join(root, "prompts.json"), "r"))
        self.preprocessor = CLIPImageProcessor()
        self.preprocessor_896 = CLIPImageProcessor(size=896, crop_size=896)
        self.transforms = PairCompose(
            [
                RandomResizeCrop(size=size, crop_p=0.5),
                ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        video_name = self.data[index]
        prompt = self.prompts[video_name]["prompt"]
        video_path = os.path.join(self.video_root, video_name)
        
        if "entity" in self.prompts[video_name]:
            is_single = True
            entities = self.prompts[video_name]["entity"]
            concept_image_paths = [os.path.join(video_path, "concept.png")]
        else:
            is_single = False
            entities = self.prompts[video_name]["entities"]
            concept_image_paths = [os.path.join(video_path, f"concept{i}.png") for i in range(2)]
        predicate = self.prompts[video_name].get("predicate", None)
        
        concept_images = [Image.open(image_path).convert("RGB").resize((512, 512)) for image_path in concept_image_paths]
        concept_images_896 = self.preprocessor_896(concept_images, return_tensors="pt").pixel_values
        concept_images = self.preprocessor(concept_images, return_tensors="pt").pixel_values    # (n, 3, 224, 224)
        
        end_frame_path = os.path.join(video_path, "motion.png")
        if is_single:
            end_frame_masks_path = [end_frame_path.replace("videos", "masks")]
        else:
            end_frame_masks_path = [end_frame_path.replace("videos", "masks").replace("motion", f"motion{i}") for i in range(2)]
            
        end_frame = Image.open(end_frame_path).convert("RGB")
        original_size = end_frame.size[::-1]
        end_frame_masks = [Image.open(end_frame_mask_path).convert("L") for end_frame_mask_path in end_frame_masks_path]
        
        end_frame, kwargs = self.transforms(img=end_frame, mask=end_frame_masks)
        masks = kwargs["mask"] >= 0.5    # (n, 128, 128)
        res_h, res_w = masks.shape[1], masks.shape[2]
        bboxes = []
        for mask in masks:
            x1, y1, x2, y2 = torch.tensor(mask2bbox(mask))
            bbox = torch.tensor([x1 / res_w, y1 / res_h, x2 / res_w, y2 / res_h])
            bboxes.append(bbox)
        bboxes = torch.stack(bboxes, dim=0)     # (n, 4)
        
        ret = dict(
            images=end_frame,
            concept_images=concept_images,
            concept_images_896=concept_images_896,
            bboxes=bboxes,
            prompts=prompt,
            entities=entities,
            predicate=predicate,
            original_sizes=original_size,
            crop_top_lefts=(0, 0)
        )
        
        return ret


class KeypointsDataset(Dataset):
    def __init__(self, root, size=1024):
        self.size = size
        self.data_root = os.path.join(root, "videos")
        self.mask_root = os.path.join(root, "masks")
        self.data = os.listdir(self.data_root)
        self.prompts = json.load(open(os.path.join(root, "prompts.json"), "r"))
        self.preprocessor = CLIPImageProcessor()
        self.transforms = PairCompose(
            [
                RandomResizeCrop(size=size, crop_p=0.5),
                ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        data = self.data[index]
        prompt: str = self.prompts[data]["prompt"]
        entities: list = self.prompts[data]["entities"] if "entities" in self.prompts[data] else self.prompts[data]["entity"]
        video_path = os.path.join(self.data_root, data)
        
        if os.path.exists(os.path.join(video_path, "concept.png")):
            is_single = True
            concept_images_path = [os.path.join(video_path, "concept.png")]
        else:
            is_single = False
            concept_images_path = [os.path.join(video_path, f"concept{i}.png") for i in range(2)]
        predicate = self.prompts[data].get("predicate", None)

        concept_images_for_vae = [Image.open(path).convert("RGB") for path in concept_images_path]
        concept_images_for_vae = [self.transforms(image)[0] for image in concept_images_for_vae]
        concept_images_for_vae = torch.stack(concept_images_for_vae)    # (n, 3, h, w)
        
        concept_images = [Image.open(concept_image_path).convert("RGB").resize((512, 512)) for concept_image_path in concept_images_path]
        concept_images = self.preprocessor(concept_images, return_tensors="pt").pixel_values  # (n, 3, 224, 224)
        
        end_frame_path = os.path.join(video_path, "motion.png")
        if is_single:
            end_frame_masks_path = [end_frame_path.replace("videos", "masks")]
        else:
            end_frame_masks_path = [end_frame_path.replace("videos", "masks").replace("motion", f"motion{i}") for i in range(2)]
            
        end_frame = Image.open(end_frame_path).convert("RGB")
        original_size = end_frame.size[::-1]
        end_frame_masks = [Image.open(end_frame_mask_path).convert("L") for end_frame_mask_path in end_frame_masks_path]
        
        end_frame, kwargs = self.transforms(img=end_frame, mask=end_frame_masks)
        masks = kwargs["mask"] >= 0.5    # (n, 128, 128)
        object_segmaps = torch.max(masks, dim=0)[0].to(torch.float32)    # (128, 128)
        res_h, res_w = masks.shape[1], masks.shape[2]
        bboxes = []
        for mask in masks:
            x1, y1, x2, y2 = torch.tensor(mask2bbox(mask))
            bbox = torch.tensor([x1 / res_w, y1 / res_h, x2 / res_w, y2 / res_h])
            bboxes.append(bbox)
        bboxes = torch.stack(bboxes, dim=0)     # (n, 4)    
        
        if is_single:
            concept_keypoints = torch.load(os.path.join(video_path, "concept_keyposes.pth"))
            motion_keypoints = torch.load(os.path.join(video_path, "motion_keyposes.pth"))
            keypoints = torch.cat([motion_keypoints, concept_keypoints])
        else:
            concept_keypoints_one = torch.load(os.path.join(video_path, "concept_keyposes_one.pth"))
            concept_keypoints_two = torch.load(os.path.join(video_path, "concept_keyposes_two.pth"))
            motion_keypoints = torch.load(os.path.join(video_path, "motion_keyposes.pth"))
            keypoints = torch.cat([motion_keypoints, concept_keypoints_one, concept_keypoints_two])
        
        
        ret = dict(
            images=end_frame,
            concept_images=concept_images,
            bboxes=bboxes,
            object_segmaps=object_segmaps,
            prompts=prompt,
            entities=entities,
            predicate=predicate,
            concept_images_for_vae=concept_images_for_vae,
            keypoints=keypoints,
            original_sizes=original_size,
            crop_top_lefts=(0, 0)
        )
        
        return ret
    
    
class DenseFeatureDataset(Dataset):
    def __init__(self, root, size=1024):
        self.root = root
        self.size = size
        self.data = [data_dir for data_dir in os.listdir(self.root) if os.path.isdir(os.path.join(self.root, data_dir))]  
        
        self.preprocessor = CLIPImageProcessor()
        self.preprocessor_896 = CLIPImageProcessor(size=896, crop_size=896)
        self.transforms = PairCompose(
            [
                RandomResizeCrop(size=size, crop_p=0.5),
                EnhanceText(),
                ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        data_dir = self.data[index]
        
        noun = [w for w in data_dir.split("_") if not w.isdigit()]
        noun = " ".join(noun)
        entities = noun
        
        data_dir_path = os.path.join(self.root, data_dir)
        # prompt_for_each_image = self.annos[data_dir]
        
        images = [image_name for image_name in os.listdir(data_dir_path) if image_name.endswith("jpg")]
        concept_image = random.choice(images)
        concept_image_path = os.path.join(data_dir_path, concept_image)
        
        target_image = random.choice(images)
        while target_image == concept_image:
            target_image = random.choice(images)
        target_image_path = os.path.join(data_dir_path, target_image)
        target_mask_path = target_image_path.replace("jpg", "png")
        
        concept_images = [Image.open(concept_image_path).convert("RGB").resize((512, 512))]
        concept_images_896 = self.preprocessor_896(concept_images, return_tensors="pt").pixel_values    # (n, 3, 896, 896)
        concept_images = self.preprocessor(concept_images, return_tensors="pt").pixel_values    # (n, 3, 224, 224)

        end_frame = Image.open(target_image_path).convert("RGB")
        original_size = end_frame.size[::-1]    # h, w
        end_frame_masks = [Image.open(target_mask_path).convert("L")]
        
        end_frame, kwargs = self.transforms(img=end_frame, mask=end_frame_masks, prompts=noun)
        masks = kwargs["mask"] >= 0.5    # (n, 128, 128)
        res_h, res_w = masks.shape[1], masks.shape[2]
        bboxes = []
        for mask in masks:
            x1, y1, x2, y2 = torch.tensor(mask2bbox(mask))
            bbox = torch.tensor([x1 / res_w, y1 / res_h, x2 / res_w, y2 / res_h])
            bboxes.append(bbox)
        bboxes = torch.stack(bboxes, dim=0)     # (n, 4)
        
        prompt = kwargs["prompts"]
        ret = dict(
            images=end_frame,
            concept_images=concept_images,
            concept_images_896=concept_images_896,
            bboxes=bboxes,
            prompts=prompt,
            entities=entities,
            original_sizes=original_size,
            crop_top_lefts=(0, 0)
        )
        
        return ret
    

class SingleObjectDataset(Dataset):
    def __init__(self, root, size=1024):
        self.root = root
        self.data = [image_name for image_name in os.listdir(self.root) if image_name.endswith(".jpg")]
        self.entity = os.path.basename(self.root).replace("_", " ")
        self.prompt = f"A photo of a {self.entity}."
        self.preprocessor = CLIPImageProcessor()
        self.preprocessor_896 = CLIPImageProcessor(size=896, crop_size=896)
        self.transforms = PairCompose(
            [
                RandomResizeCrop(size=size, crop_p=0.5),
                ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        image_name = self.data[index]
        image_path = os.path.join(self.root, image_name)
        concept_image = [Image.open(image_path).convert("RGB").resize((512, 512))]
        processed_concept_image = self.preprocessor(concept_image, return_tensors="pt").pixel_values
        concept_images_896 = self.preprocessor_896(concept_image, return_tensors="pt").pixel_values    # (n, 3, 896, 896)

        target_name = random.choice(self.data)  
        target_path = os.path.join(self.root, target_name)
        mask_path = target_path.replace(".jpg", ".png")
        
        target_image = Image.open(target_path).convert("RGB")
        original_size = target_image.size[::-1]
        target_mask = [Image.open(mask_path).convert("L")]
        
        end_frame, kwargs = self.transforms(img=target_image, mask=target_mask)
        mask = kwargs["mask"][0] >= 0.5    # (1, 128, 128)
        res = mask.shape[0]
        bbox = torch.tensor(mask2bbox(mask)).unsqueeze(0) / res    # (1, 4)
        
        ret = dict(
            images=end_frame,
            concept_images=processed_concept_image,
            concept_images_896=concept_images_896,
            bboxes=bbox,
            prompts=self.prompt,
            entities=self.entity,
            original_sizes=original_size,
            crop_top_lefts=(0, 0)
        )
        
        return ret


class MVImagenetDataset(Dataset):
    def __init__(self, root, size=1024):
        self.root = root
        self.size = size
        
        with open(os.path.join(root, "mvimgnet_category.txt"), "r") as file:
            lines = file.read().splitlines()

        idx2name = {}
        for line in lines:
            idx, name = line.split(",")
            idx2name[idx] = name.lower()
        self.idx2category = idx2name
        
        self.prompts = json.load(open(os.path.join(root, "prompts.json"), "r"))
        self.data = []
        self.data2category = {}
        for class_dir in os.listdir(self.root):
            class_dir_path = os.path.join(self.root, class_dir)
            if os.path.isdir(class_dir_path):
                category_name = self.idx2category[class_dir]
                for image_pair in os.listdir(class_dir_path):
                    image_pair_path = os.path.join(class_dir_path, image_pair, "images")
                    if len(os.listdir(image_pair_path)) == 4:
                        data = image_pair_path
                        self.data.append(data)
                        self.data2category[data] = category_name
        
        self.preprocessor = CLIPImageProcessor()
        self.preprocessor_896 = CLIPImageProcessor(size=896, crop_size=896)
        
        self.train_resize = transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR)
        self.train_mask_resize = transforms.Resize(size, interpolation=transforms.InterpolationMode.NEAREST)
        self.train_crop = transforms.CenterCrop(size)
        self.train_flip = transforms.RandomHorizontalFlip(p=1.0)
        self.train_transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
        self.text_enhance = EnhanceText()
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        data = self.data[index]
        category = self.data2category[data]
        
        images = [image_name for image_name in os.listdir(data) if image_name.endswith("jpg")]
        concept_image = random.choice(images)
        concept_image_path = os.path.join(data, concept_image)
        
        target_image = random.choice(images)
        while target_image == concept_image:
            target_image = random.choice(images)
        target_image_path = os.path.join(data, target_image)
        target_mask_path = target_image_path.replace("jpg", "png")
        
        concept_images = [Image.open(concept_image_path).convert("RGB")]
        concept_images_896 = self.preprocessor_896(concept_images, return_tensors="pt").pixel_values    # (n, 3, 896, 896)
        concept_images = self.preprocessor(concept_images, return_tensors="pt").pixel_values    # (n, 3, 224, 224)

        image = Image.open(target_image_path).convert("RGB")
        image_mask = Image.open(target_mask_path).convert("L")
        
        original_size = (image.height, image.width)
        image = self.train_resize(image)
        image_mask = self.train_mask_resize(image_mask)
        if self.train_flip is not None and random.random() < 0.5:
            # flip
            image = self.train_flip(image)
            image_mask = self.train_flip(image_mask)

        y1 = max(0, int(round((image.height - self.size) / 2.0)))
        x1 = max(0, int(round((image.width - self.size) / 2.0)))
        image = self.train_crop(image)
        image_mask = self.train_crop(image_mask)
        crop_top_left = (y1, x1)
        image = self.train_transforms(image)

        masks = ToTensor()(image_mask)
        res_h, res_w = masks.shape[1], masks.shape[2]
        bboxes = []
        for mask in masks:
            x1, y1, x2, y2 = torch.tensor(mask2bbox(mask))
            bbox = torch.tensor([x1 / res_w, y1 / res_h, x2 / res_w, y2 / res_h])
            bboxes.append(bbox)
        bboxes = torch.stack(bboxes, dim=0)     # (n, 4)

        prompt = self.prompts[target_image_path]
        ret = dict(
            images=image,
            concept_images=concept_images,
            concept_images_896=concept_images_896,
            bboxes=bboxes,
            prompts=prompt,
            entities=category,
            original_sizes=original_size,
            crop_top_lefts=crop_top_left
        )
        
        return ret


if __name__ == "__main__":
    cases = []
    root = "./data"
    for dir in os.listdir(root):
        dir_path = os.path.join(root, dir)   
        dataset = MSDiffusionDataset(root=dir_path)
        data = dataset[1]
        case = {}
        case["prompt"] = data["prompts"]
        case["bbox"] = data["bboxes"].tolist()
        cases.append(case)
    with open("cases.json", "w") as file:
        json.dump(cases, file)