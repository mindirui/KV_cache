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

@dataclass
class DatasetConfig:
    root: str = "D:/GSOdataset"
    seed: int = 42
    samples_per_object: int = 4
    view_num : int = 4
    input_num : int = 3
    output_num : int = 1
    resolution : int = None 
    sample_type : str = 'random'

class GSO_sequence(Dataset):
    def __init__(self, config: DatasetConfig):
        self.config = config
        np.random.seed(self.config.seed)
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
            samples = self.sample_object(
                self.config.samples_per_object,
                self.config.view_num,
                self.config.input_num,
                self.config.sample_type
            )

            for sample in samples:
                sample_meta = {
                    "path": obj_path,
                    "input": sample["input"],
                    "output": sample["output"]
                }
                self.metas.append(sample_meta)
                

    def sample_object(self, samples_per_object, view_num, input_num, sample_type):
        azimuth_list = [0, 30, 90, 150, 210, 270, 330]
        elevation_list = [-20, 30]
        radius_list = [1.0, 1.2, 1.4, 1.6, 1.8, 2.0]

        default_radius_min = 1.5
        default_radius_max = 2.2

        valid_radius = [r for r in radius_list if default_radius_min <= r <= default_radius_max]
        metas = []
        rng = np.random.default_rng(self.config.seed)

        for _ in range(samples_per_object):
            if sample_type == 'random':
                all_combinations = list(itertools.product(elevation_list, azimuth_list, radius_list))
                selected = rng.choice(all_combinations, size=view_num, replace=False)

            elif sample_type == 'cycle':
                elev = rng.choice(elevation_list)
                radius = rng.choice(radius_list)
                azim_start_idx = rng.integers(0, len(azimuth_list))
                azim_choices = [(azimuth_list[(azim_start_idx + i) % len(azimuth_list)]) for i in range(view_num)]
                selected = [(elev, azim, radius) for azim in azim_choices]
                
            elif sample_type == 'eschernet':
                all_combinations = list(itertools.product(elevation_list, azimuth_list, valid_radius))
                selected = rng.choice(all_combinations, size=view_num, replace=False)

            else:
                raise ValueError(f"Unknown sample_type: {sample_type}")

            rng.shuffle(selected)
            input_items = selected[:input_num]
            output_items = selected[input_num:]

            def format_name(elev, azim, r):
                elev_id = int(elev == -20)
                azim_id = f"{azim / 30:.1f}"
                return f"{elev_id}_{azim_id}_{r}.png"

            metas.append({
                "input": [format_name(*tup) for tup in input_items],
                "output": [format_name(*tup) for tup in output_items]
            })

        return metas
    
    def _view_to_filename(self, elev, azim, radius):
        elev_id = int(elev == -20)  # 1 or 0
        azim_id = round(azim / 30, 1)  # 保留1位小数
        return f"{elev_id}_{azim_id}_{radius}.png"
    
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
    config = DatasetConfig(root = "D:/GSOdataset_new", sample_type='random')
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

        