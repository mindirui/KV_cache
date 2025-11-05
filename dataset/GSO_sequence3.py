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
        def construct_paths(elevation, azimuths, obj_path):
            paths = []
            for az in azimuths:
                # 拼接搜索模式，例如 "30_90_*.png"
                search_pattern = os.path.join(obj_path, f"{elevation}_{az}_*.png")
                matched_files = glob.glob(search_pattern)
                if matched_files:
                    paths.append(matched_files[0])  # 取第一个匹配的文件
                else:
                    print(f"Warning: No files matched for pattern {search_pattern}")
            return paths
        
        self.metas = []
        object_path = os.listdir(self.config.root)
        for path in object_path:
            obj_path = os.path.join(self.config.root, path, "render_mvs_25/model")
            samples = self.sample_object(self.config.samples_per_object, self.config.view_num, self.config.input_num)
            for sample in samples:
                elevation = sample["elevation"]
                azimuth_input = sample["azimuth_input"]
                azimuth_output = sample["azimuth_output"]
                input_paths = construct_paths(elevation, azimuth_input, obj_path)
                output_paths = construct_paths(elevation, azimuth_output, obj_path)
                self.metas.append({
                    "input": input_paths,
                    "output": output_paths
                })
                
    def sample_object(self,  samples_per_object, view_num, input_num, Neighboring_views = True):
        azimuth_list = [30, 90, 150, 210, 270, 330]
        elevation_list = [-20, 30]
        results = []
        for _ in range(samples_per_object):
            elevation = np.random.choice(elevation_list)

            # 连续采样，不环形
            start_idx = np.random.randint(0, len(azimuth_list) - view_num + 1)
            azimuth_seq = azimuth_list[start_idx : start_idx + view_num]

            azimuth_seq_shuffled = np.random.permutation(azimuth_seq)
            azimuth_output = sorted(azimuth_seq_shuffled[:view_num - input_num])
            azimuth_input = sorted(set(azimuth_seq) - set(azimuth_output))
            results.append({
                        "elevation": elevation,
                        "azimuth_input": azimuth_input,
                        "azimuth_output": azimuth_output,
                    })
        return results

    def __len__(self):
        return len(self.metas)
    
    def __getitem__(self, idx):
        meta = self.metas[idx]
        input_images = []
        input_poses = []
        input_poses_inv = []
        input_path = []
        for img_path in meta["input"]:
            # 加载图像
            img = plt.imread(img_path)
            img[img[:, :, -1] == 0.] = [1., 1., 1., 1.]
            img = Image.fromarray(np.uint8(img[:, :, :3] * 255.)).convert("RGB")
            img_tensor = self.transform(img) if hasattr(self, "transform") else torch.tensor(np.array(img)).permute(2, 0, 1).float() / 255.0
            input_images.append(img_tensor)

            # 加载对应的 pose
            pose_path = img_path.replace(".png", ".npy")
            pose = np.load(pose_path)
            pose_4x4 = np.eye(4, dtype=np.float32)
            pose_4x4[:3, :4] = pose
            input_poses.append(torch.tensor(pose_4x4, dtype=torch.float32))
            input_poses_inv.append(torch.tensor(np.linalg.inv(pose_4x4), dtype=torch.float32))
            
            input_path.append(img_path)
            
        output_images = []
        output_poses = []
        output_poses_inv = []
        output_path = []
        for img_path in meta["output"]:
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
            output_poses_inv.append(torch.tensor(np.linalg.inv(pose_4x4), dtype=torch.float32))
            
            output_path.append(img_path)

        return {
            "input_images": torch.stack(input_images),
            "input_poses": torch.stack(input_poses),
            "input_poses_inv": torch.stack(input_poses_inv),
            "output_images": torch.stack(output_images),
            "output_poses": torch.stack(output_poses),
            "output_poses_inv": torch.stack(output_poses_inv),
            "input_path": input_path,
            "output_path": output_path
    }

        

if __name__ == "__main__":
    config = DatasetConfig()
    dataset = GSO_sequence(config)
    sample = dataset[0]
    
    from torchvision.utils import make_grid
    import matplotlib.pyplot as plt
    grid = make_grid(sample["input_images"], nrow=4)
    plt.imshow(grid.permute(1, 2, 0))  # C, H, W → H, W, C
    plt.title("Input Images")
    plt.axis("off")
    plt.show()

    # 打印一个 pose 矩阵
    print("Example input pose:\n", sample["input_poses"][0])