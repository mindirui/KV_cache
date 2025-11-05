from torch.utils.data import Dataset
import os
import numpy as np
from torchvision import transforms as T
from PIL import Image
from dataclasses import dataclass
import random
import torch
import glob
import matplotlib.pyplot as plt
import itertools
import re

@dataclass
class DatasetConfig:
    root: str = "D:/GSOdataset"
    seed: int = 42
    samples_per_object: int = 4
    input_num : int = 3
    output_num : int = 1
    resolution : int = None 
    sample_type : str = 'random'
    aug_num : int = 1
    select_imgs : list = None

class GSO_sequence(Dataset):
    def __init__(self, config: DatasetConfig):
        self.config = config
        self.seed = config.seed
        if config.sample_type == "select_imgs":
            self.metas = []
            for i in config.select_imgs:
                meta = {
                    "path": i["obj_path"],
                    "input": i["input"],
                    "output": i["output"]
                }
                self.metas.append(meta)
        else:
            self.build_metas()
        if config.resolution:
            self.transform = T.Compose([
                T.ToTensor(),  
                T.Resize((config.resolution, config.resolution)),  # 256, 256
                T.Normalize([0.5], [0.5])
            ])


    def build_metas(self):        
        self.metas = []
        object_paths = os.listdir(self.config.root)
        # for path in object_path:
        #     obj_path = os.path.join(self.config.root, path, "model")
        #     samples = self.sample_object(self.config.samples_per_object, self.config.view_num, self.config.input_num, self.config.sample_type)
        
        for path in object_paths:
            obj_path = os.path.join(self.config.root, path, "model")
            pattern = re.compile(r"^\d{3}\.png$")
            valid_files = list(filter(pattern.match, os.listdir(obj_path)))
            if len(valid_files) < self.config.input_num + self.config.output_num:
                    raise ValueError(
                f"文件数量不足：{obj_path} 中仅找到 {len(valid_files)} 个有效文件，")
            
            samples = self.sample_object(
                self.config.samples_per_object,
                self.config.input_num,
                self.config.output_num,
                self.config.sample_type,
                valid_files
            )

            for sample in samples:
                sample_meta = {
                    "path": obj_path,
                    "input": sample["input"],
                    "output": sample["output"]
                }
                self.metas.append(sample_meta)
                

    def sample_object(self, samples_per_object, input_num, output_num, sample_type,files_list):
            # 设置随机种子
        random.seed(self.seed)
        if sample_type == "random":
            samples = []
            for _ in range(samples_per_object):
                shuffled = files_list.copy()
                random.shuffle(shuffled)

                input_files = shuffled[:input_num]
                output_files = shuffled[-output_num:]

                sample = {
                    "input": input_files,
                    "output": output_files[::-1]
                }
                samples.append(sample)
        elif sample_type == "random_aug":
            samples = []
            for _ in range(samples_per_object):
                shuffled = files_list.copy()
                random.shuffle(shuffled)

                input_files = shuffled[:input_num]
                output_files = shuffled[-output_num:]

                # 添加增补数据
                augmented_inputs = input_files.copy()
                for fname in input_files:
                    base_name = fname[:-4]  # 去掉 .png，例如 '000'
                    for i in range(self.config.aug_num):
                        aug_name = f"{base_name}_aug{i}.png"
                        augmented_inputs.append(aug_name)
                sample = {
                    "input": augmented_inputs,
                    "output": output_files
                }
                samples.append(sample)
            
            
            
        return samples

    
    def __len__(self):
        return len(self.metas)
    
    def __getitem__(self, idx):
        meta = self.metas[idx]
        base_path = meta["path"]  # 例如 "/path/to/object/model"

        input_images = []
        input_poses = []
        input_path = []

        for img_name in meta["input"]:
            img_path = os.path.join(base_path, img_name)
            img = plt.imread(img_path)
            img[img[:, :, -1] == 0.] = [1., 1., 1., 1.]
            img = Image.fromarray(np.uint8(img[:, :, :3] * 255.)).convert("RGB")
            img_tensor = self.transform(img) if hasattr(self, "transform") else torch.tensor(np.array(img)).permute(2, 0, 1).float() / 255.0
            input_images.append(img_tensor)

            pose_path = img_path.replace(".png", ".npy")
            pose = np.load(pose_path)
            pose_4x4 = np.eye(4, dtype=np.float32)
            pose_4x4[:3, :4] = pose
            input_poses.append(torch.tensor(pose_4x4, dtype=torch.float32))

            input_path.append(img_path)

        output_images = []
        output_poses = []
        output_path = []

        for img_name in meta["output"]:
            img_path = os.path.join(base_path, img_name)
            img = plt.imread(img_path)
            img[img[:, :, -1] == 0.] = [1., 1., 1., 1.]
            img = Image.fromarray(np.uint8(img[:, :, :3] * 255.)).convert("RGB")
            img_tensor = self.transform(img) if hasattr(self, "transform") else torch.tensor(np.array(img)).permute(2, 0, 1).float() / 255.0
            output_images.append(img_tensor)

            pose_path = img_path.replace(".png", ".npy")
            pose = np.load(pose_path)
            pose_4x4 = np.eye(4, dtype=np.float32)
            pose_4x4[:3, :4] = pose
            output_poses.append(torch.tensor(pose_4x4, dtype=torch.float32))

            output_path.append(img_path)

        return {
            "input_images": torch.stack(input_images),
            "input_poses": torch.stack(input_poses),
            "output_images": torch.stack(output_images),
            "output_poses": torch.stack(output_poses),
            "input_path": input_path,
            "output_path": output_path
        }
        
    def __len__(self):
        return len(self.metas)
        
def print_sample_comparison(sample1, sample2):
    print("=== Sample 1 ===")
    print("Input Paths:")
    for p in sample1["input_path"]:
        print("  -", p)
    print("Output Paths:")
    for p in sample1["output_path"]:
        print("  -", p)

    print("\n=== Sample 2 ===")
    print("Input Paths:")
    for p in sample2["input_path"]:
        print("  -", p)
    print("Output Paths:")
    for p in sample2["output_path"]:
        print("  -", p)

    print("\n=== Summary ===")
    print(f"Sample 1 input count: {len(sample1['input_path'])}, output count: {len(sample1['output_path'])}")
    print(f"Sample 2 input count: {len(sample2['input_path'])}, output count: {len(sample2['output_path'])}")

if __name__ == "__main__":
    config = DatasetConfig(root = "D:/GSO_datasets/eschernet_data", sample_type='random',input_num=3,output_num=4,samples_per_object=4,seed=42)
    dataset = GSO_sequence(config)

    for i in range(dataset.__len__()):
        sample = dataset[i]
        input_images = sample["input_images"]  # [N, C, H, W]
        output_images = sample["output_images"]
        input_paths = sample["input_path"]
        output_paths = sample["output_path"]
        total_imgs = len(input_images) + len(output_images)
        plt.figure(figsize=(3 * total_imgs, 4))

        # 显示 input 图
        for i, (img_tensor, path) in enumerate(zip(input_images, input_paths)):
            plt.subplot(2, total_imgs, i + 1)
            img_np = img_tensor.permute(1, 2, 0).cpu().numpy()
            plt.imshow(img_np)
            plt.title(f"Input {i}\n{path.split('/')[-1]}")
            plt.axis('off')

        # 显示 output 图
        for i, (img_tensor, path) in enumerate(zip(output_images, output_paths)):
            plt.subplot(2, total_imgs, total_imgs + i + 1)
            img_np = img_tensor.permute(1, 2, 0).cpu().numpy()
            plt.imshow(img_np)
            plt.title(f"Output {i}\n{path.split('/')[-1]}")
            plt.axis('off')

        plt.tight_layout()
        plt.show()

        