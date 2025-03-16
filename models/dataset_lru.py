import torch
import torch.nn.functional as F
import cv2 as cv
import numpy as np
import os
from glob import glob
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial.transform import Slerp
import logging
from PIL import Image
import gc


# This function is borrowed from IDR: https://github.com/lioryariv/idr
def load_K_Rt_from_P(filename, P=None):
    if P is None:
        lines = open(filename).read().splitlines()
        if len(lines) == 4:
            lines = lines[1:]
        lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines)]
        P = np.asarray(lines).astype(np.float32).squeeze()

    out = cv.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    K = K / K[2, 2]
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3, 3] = (t[:3] / t[3])[:, 0]

    return intrinsics, pose


def sample_image_as_numpy(torch_img, resolution_level):
    l = resolution_level
    tx = torch.linspace(0, torch_img.shape[1] - 1, torch_img.shape[1] // l)
    ty = torch.linspace(0, torch_img.shape[0] - 1, torch_img.shape[0] // l)
    pixels_x, pixels_y = torch.meshgrid(tx, ty, indexing='ij')
    pixels_x = pixels_x.to(torch.int64).transpose(0, 1)
    pixels_y = pixels_y.to(torch.int64).transpose(0, 1)
    torch_img = torch_img[(pixels_y, pixels_x)]
    numpy_img = torch_img.detach().cpu().numpy()
    return numpy_img


# 代替の読み込み方法を追加
def safe_read_image(path):
    """OpenCVとPILの両方を試して画像を読み込む"""
    # まずOpenCVで試す
    img = cv.imread(path)
    if img is not None:
        return img
    
    # OpenCVが失敗したらPILで試す
    try:
        pil_img = Image.open(path)
        # PIL -> NumPy -> OpenCV形式に変換
        img = np.array(pil_img)
        if len(img.shape) == 2:  # グレースケール
            img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
        elif img.shape[2] == 4:  # RGBA
            img = cv.cvtColor(img, cv.COLOR_RGBA2BGR)
        elif img.shape[2] == 3:  # RGB
            img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
        return img
    except Exception as e:
        logging.warning(f"PILでも読み込めませんでした: {path}, エラー: {e}")
        return None


class Dataset:
    def __init__(self, conf):
        super(Dataset, self).__init__()
        print('Load data: Begin')
        self.device = torch.device('cuda')
        self.conf = conf

        self.data_dir = conf.get_string('data_dir')
        logging.info(f"Loading data from directory: {self.data_dir}")
        
        self.render_cameras_name = conf.get_string('render_cameras_name')
        self.object_cameras_name = conf.get_string('object_cameras_name')

        self.camera_outside_sphere = conf.get_bool('camera_outside_sphere', default=True)
        self.scale_mat_scale = conf.get_float('scale_mat_scale', default=1.1)

        # カメラデータの読み込み
        camera_path = os.path.join(self.data_dir, self.render_cameras_name)
        if not os.path.exists(camera_path):
            raise FileNotFoundError(f"Camera file not found: {camera_path}")
            
        camera_dict = np.load(camera_path)
        self.camera_dict = camera_dict
        
        # 画像パスの取得
        self.images_lis = sorted(glob(os.path.join(self.data_dir, 'image/*.png')))
        if len(self.images_lis) == 0:
            self.images_lis = sorted(glob(os.path.join(self.data_dir, 'image/*.jpg')))
            
        if len(self.images_lis) == 0:
            raise FileNotFoundError(f"No image files found in {os.path.join(self.data_dir, 'image/')}")
            
        orig_n_images = len(self.images_lis)
        logging.info(f"Found {orig_n_images} images")
        
        # マスク画像の読み込み
        self.masks_lis = sorted(glob(os.path.join(self.data_dir, 'mask/*.png')))
        if len(self.masks_lis) == 0:
            self.masks_lis = sorted(glob(os.path.join(self.data_dir, 'mask/*.jpg')))
            
        if len(self.masks_lis) == 0:
            raise FileNotFoundError(f"No mask files found in {os.path.join(self.data_dir, 'mask/')}")
        
        # データローダー方式に変更：最初の画像だけロードして寸法を取得
        sample_image_path = self.images_lis[0]
        sample_mask_path = self.masks_lis[0]
        
        if not os.path.exists(sample_image_path):
            raise FileNotFoundError(f"Sample image not found: {sample_image_path}")
            
        sample_img = safe_read_image(sample_image_path)
        if sample_img is None:
            raise IOError(f"Failed to load sample image: {sample_image_path}")
        
        if not os.path.exists(sample_mask_path):
            raise FileNotFoundError(f"Sample mask not found: {sample_mask_path}")
        
        sample_mask = safe_read_image(sample_mask_path)
        if sample_mask is None:
            raise IOError(f"Failed to load sample mask: {sample_mask_path}")
        
        # 画像のサイズを記録（後で使用）
        self.img_h, self.img_w = sample_img.shape[:2]
        logging.info(f"Image dimensions: {self.img_w}x{self.img_h}")
        
        # 問題のある画像インデックスをキャッシュするセット
        self.bad_image_indices = set()
        
        # LRU画像キャッシュ（最大20枚までメモリに保持）
        self.image_cache = {}
        self.mask_cache = {}
        self.max_cache_size = 20
        
        # 画像をテストロード（無効な画像をチェック）
        valid_images_lis = []
        valid_masks_lis = []
        
        for i, (img_path, mask_path) in enumerate(zip(self.images_lis, self.masks_lis)):
            try:
                # 存在チェックと読み込みテスト
                if os.path.exists(img_path) and os.path.exists(mask_path):
                    img = safe_read_image(img_path)
                    if img is None:
                        logging.warning(f"Failed to load image, skipping: {img_path}")
                        self.bad_image_indices.add(i)
                        continue
                        
                    msk = safe_read_image(mask_path)
                    if msk is None:
                        logging.warning(f"Failed to load mask, skipping: {mask_path}")
                        self.bad_image_indices.add(i)
                        continue
                    
                    valid_images_lis.append(img_path)
                    valid_masks_lis.append(mask_path)
                else:
                    self.bad_image_indices.add(i)
                    logging.warning(f"Image or mask file not found, skipping: {img_path} or {mask_path}")
            except Exception as e:
                self.bad_image_indices.add(i)
                logging.error(f"Error checking image at index {i}: {e}")
        
        # 有効な画像リストに更新
        self.images_lis = valid_images_lis
        self.masks_lis = valid_masks_lis
        self.n_images = len(self.images_lis)
        
        if self.n_images == 0:
            raise RuntimeError("No valid images found!")
            
        logging.info(f"Found {self.n_images} valid image-mask pairs out of {orig_n_images}")
        
        if self.n_images < orig_n_images:
            logging.warning(f"Only {self.n_images} out of {orig_n_images} images were loaded successfully. Skipped {orig_n_images - self.n_images} images.")
        
        # 初期化時に必要な属性を設定
        self.world_mats_np = []
        self.scale_mats_np = []
        
        # カメラ行列を収集（有効な画像のみ）
        for i in range(self.n_images):
            # カメラ行列のキーが存在するか確認
            img_idx = int(os.path.basename(self.images_lis[i]).split('.')[0])  # ファイル名から元のインデックスを取得
            if f'world_mat_{img_idx}' not in camera_dict or f'scale_mat_{img_idx}' not in camera_dict:
                logging.warning(f"Camera matrix for image {i} (idx: {img_idx}) not found, using first camera instead")
                # 最初のカメラ行列を代用
                if i > 0 and len(self.world_mats_np) > 0:
                    self.world_mats_np.append(self.world_mats_np[0].copy())
                    self.scale_mats_np.append(self.scale_mats_np[0].copy())
                else:
                    # 一つも有効なカメラがない場合
                    fallback_idx = 0
                    while f'world_mat_{fallback_idx}' not in camera_dict or f'scale_mat_{fallback_idx}' not in camera_dict:
                        fallback_idx += 1
                        if fallback_idx >= 1000:  # 安全対策
                            raise RuntimeError("No valid camera matrices found!")
                    
                    self.world_mats_np.append(camera_dict[f'world_mat_{fallback_idx}'].astype(np.float32))
                    self.scale_mats_np.append(camera_dict[f'scale_mat_{fallback_idx}'].astype(np.float32))
            else:
                self.world_mats_np.append(camera_dict[f'world_mat_{img_idx}'].astype(np.float32))
                self.scale_mats_np.append(camera_dict[f'scale_mat_{img_idx}'].astype(np.float32))
        
        # 有効な画像数を確認
        if self.n_images != len(self.world_mats_np):
            logging.error(f"Mismatch between number of valid images ({self.n_images}) and camera matrices ({len(self.world_mats_np)})")
            raise RuntimeError("Number of valid images doesn't match number of camera matrices!")
            
        logging.info(f"Final number of usable images: {self.n_images}")

        self.intrinsics_all = []
        self.pose_all = []

        for scale_mat, world_mat in zip(self.scale_mats_np, self.world_mats_np):
            P = world_mat @ scale_mat
            P = P[:3, :4]
            intrinsics, pose = load_K_Rt_from_P(None, P)
            self.intrinsics_all.append(torch.from_numpy(intrinsics).float())
            self.pose_all.append(torch.from_numpy(pose).float())

        self.intrinsics_all = torch.stack(self.intrinsics_all).to(self.device)   # [n_images, 4, 4]
        self.intrinsics_all_inv = torch.inverse(self.intrinsics_all)  # [n_images, 4, 4]
        self.focal = self.intrinsics_all[0][0, 0]
        self.pose_all = torch.stack(self.pose_all).to(self.device)  # [n_images, 4, 4]
        
        # サンプル画像の寸法を使用
        self.H, self.W = self.img_h, self.img_w
        self.image_pixels = self.H * self.W

        object_bbox_min = np.array([-1.01, -1.01, -1.01, 1.0])
        object_bbox_max = np.array([ 1.01,  1.01,  1.01, 1.0])
        # Object scale mat: region of interest to **extract mesh**
        object_camera_path = os.path.join(self.data_dir, self.object_cameras_name)
        if not os.path.exists(object_camera_path):
            raise FileNotFoundError(f"Object camera file not found: {object_camera_path}")
            
        object_scale_mat = np.load(object_camera_path)['scale_mat_0']
        object_bbox_min = np.linalg.inv(self.scale_mats_np[0]) @ object_scale_mat @ object_bbox_min[:, None]
        object_bbox_max = np.linalg.inv(self.scale_mats_np[0]) @ object_scale_mat @ object_bbox_max[:, None]
        self.object_bbox_min = object_bbox_min[:3, 0]
        self.object_bbox_max = object_bbox_max[:3, 0]

        print('Load data: End')
        logging.info(f"Successfully prepared dataset with {self.n_images} valid images.")
        
        # メモリクリーンアップ
        gc.collect()

    def _get_image(self, idx):
        """インデックスから画像を取得（キャッシュ機能付き）"""
        if idx in self.bad_image_indices:
            # 問題のある画像は別のランダムな画像で代替
            logging.warning(f"Attempting to load known bad image at index {idx}, using random replacement")
            valid_indices = [i for i in range(self.n_images) if i not in self.bad_image_indices]
            if not valid_indices:
                raise RuntimeError("No valid images available!")
            idx = np.random.choice(valid_indices)
            
        # キャッシュをチェック
        if idx in self.image_cache:
            return self.image_cache[idx]
        
        # キャッシュになければ読み込む
        try:
            image_path = self.images_lis[idx]
            img = safe_read_image(image_path)
            if img is None:
                self.bad_image_indices.add(idx)
                raise IOError(f"Failed to load image: {image_path}")
                
            tensor_img = torch.from_numpy(img.astype(np.float32) / 256.0)
            
            # キャッシュが最大サイズを超えたら、最も古いアイテムを削除
            if len(self.image_cache) >= self.max_cache_size:
                oldest_key = next(iter(self.image_cache))
                del self.image_cache[oldest_key]
                
            # キャッシュに追加
            self.image_cache[idx] = tensor_img
            return tensor_img
            
        except Exception as e:
            self.bad_image_indices.add(idx)
            logging.error(f"Error loading image at index {idx}: {e}")
            # エラーが発生した場合は別の画像を試す
            valid_indices = [i for i in range(self.n_images) if i not in self.bad_image_indices]
            if not valid_indices:
                raise RuntimeError("No valid images available!")
            return self._get_image(np.random.choice(valid_indices))

    def _get_mask(self, idx):
        """インデックスからマスクを取得（キャッシュ機能付き）"""
        if idx in self.bad_image_indices:
            # 問題のある画像は別のランダムな画像で代替
            logging.warning(f"Attempting to load known bad mask at index {idx}, using random replacement")
            valid_indices = [i for i in range(self.n_images) if i not in self.bad_image_indices]
            if not valid_indices:
                raise RuntimeError("No valid masks available!")
            idx = np.random.choice(valid_indices)
            
        # キャッシュをチェック
        if idx in self.mask_cache:
            return self.mask_cache[idx]
        
        # キャッシュになければ読み込む
        try:
            mask_path = self.masks_lis[idx]
            msk = safe_read_image(mask_path)
            if msk is None:
                self.bad_image_indices.add(idx)
                raise IOError(f"Failed to load mask: {mask_path}")
                
            tensor_mask = torch.from_numpy(msk.astype(np.float32) / 256.0)
            
            # キャッシュが最大サイズを超えたら、最も古いアイテムを削除
            if len(self.mask_cache) >= self.max_cache_size:
                oldest_key = next(iter(self.mask_cache))
                del self.mask_cache[oldest_key]
                
            # キャッシュに追加
            self.mask_cache[idx] = tensor_mask
            return tensor_mask
            
        except Exception as e:
            self.bad_image_indices.add(idx)
            logging.error(f"Error loading mask at index {idx}: {e}")
            # エラーが発生した場合は別の画像を試す
            valid_indices = [i for i in range(self.n_images) if i not in self.bad_image_indices]
            if not valid_indices:
                raise RuntimeError("No valid masks available!")
            return self._get_mask(np.random.choice(valid_indices))

    def clear_cache(self):
        """キャッシュをクリアしてメモリを解放"""
        self.image_cache.clear()
        self.mask_cache.clear()
        gc.collect()
        logging.info("Image and mask cache cleared")
            
    def gen_rays_at(self, img_idx, resolution_level=1):
        """
        Generate rays at world space from one camera.
        """
        if img_idx >= self.n_images:
            logging.warning(f"Image index {img_idx} out of range, using index 0 instead.")
            img_idx = 0
            
        if img_idx in self.bad_image_indices:
            # 問題のある画像の場合は別の画像を使用
            valid_indices = [i for i in range(self.n_images) if i not in self.bad_image_indices]
            if not valid_indices:
                raise RuntimeError("No valid images available!")
            img_idx = np.random.choice(valid_indices)
            
        l = resolution_level
        tx = torch.linspace(0, self.W - 1, self.W // l)
        ty = torch.linspace(0, self.H - 1, self.H // l)
        pixels_x, pixels_y = torch.meshgrid(tx, ty)
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1) # W, H, 3
        p = torch.matmul(self.intrinsics_all_inv[img_idx, None, None, :3, :3], p[:, :, :, None]).squeeze()  # W, H, 3
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)  # W, H, 3
        rays_v = torch.matmul(self.pose_all[img_idx, None, None, :3, :3], rays_v[:, :, :, None]).squeeze()  # W, H, 3
        rays_o = self.pose_all[img_idx, None, None, :3, 3].expand(rays_v.shape)  # W, H, 3
        return rays_o.transpose(0, 1), rays_v.transpose(0, 1)

    def gen_random_rays_at(self, img_idx, batch_size):
        """
        Generate random rays at world space from one camera.
        """
        if img_idx >= self.n_images:
            logging.warning(f"Image index {img_idx} out of range, using random index instead.")
            img_idx = np.random.randint(0, self.n_images)
            
        if img_idx in self.bad_image_indices:
            # 問題のある画像の場合は別の画像を使用
            valid_indices = [i for i in range(self.n_images) if i not in self.bad_image_indices]
            if not valid_indices:
                raise RuntimeError("No valid images available!")
            img_idx = np.random.choice(valid_indices)
            
        pixels_x = torch.randint(low=0, high=self.W, size=[batch_size])
        pixels_y = torch.randint(low=0, high=self.H, size=[batch_size])
        
        # 指定された画像を読み込む
        image = self._get_image(img_idx).to(self.device)
        mask = self._get_mask(img_idx).to(self.device)
        
        color = image[(pixels_y, pixels_x)]    # batch_size, 3
        mask_value = mask[(pixels_y, pixels_x)][:, :1]  # batch_size, 1 (マスクは1チャネルのみ使用)
        
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1).float()  # batch_size, 3
        p = torch.matmul(self.intrinsics_all_inv[img_idx, None, :3, :3], p[:, :, None]).squeeze() # batch_size, 3
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)    # batch_size, 3
        rays_v = torch.matmul(self.pose_all[img_idx, None, :3, :3], rays_v[:, :, None]).squeeze()  # batch_size, 3
        rays_o = self.pose_all[img_idx, None, :3, 3].expand(rays_v.shape) # batch_size, 3
        
        # 使用後はGPUメモリから削除（必要に応じて）
        del image
        del mask
        
        return torch.cat([rays_o, rays_v, color, mask_value], dim=-1)    # batch_size, 10

    def gen_rays_between(self, idx_0, idx_1, ratio, resolution_level=1):
        """
        Interpolate pose between two cameras.
        """
        # インデックスの範囲チェック
        if idx_0 >= self.n_images or idx_1 >= self.n_images:
            logging.warning(f"Image indices {idx_0}, {idx_1} out of range, using 0 and 1 instead.")
            idx_0 = 0
            idx_1 = min(1, self.n_images-1)
            
        if idx_0 in self.bad_image_indices or idx_1 in self.bad_image_indices:
            # 問題のある画像の場合は別の画像を使用
            valid_indices = [i for i in range(self.n_images) if i not in self.bad_image_indices]
            if not valid_indices or len(valid_indices) < 2:
                raise RuntimeError("Not enough valid images available!")
            idx_0, idx_1 = np.random.choice(valid_indices, 2, replace=False)
            
        l = resolution_level
        tx = torch.linspace(0, self.W - 1, self.W // l)
        ty = torch.linspace(0, self.H - 1, self.H // l)
        pixels_x, pixels_y = torch.meshgrid(tx, ty)
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1)  # W, H, 3
        p = torch.matmul(self.intrinsics_all_inv[0, None, None, :3, :3], p[:, :, :, None]).squeeze()  # W, H, 3
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)  # W, H, 3
        trans = self.pose_all[idx_0, :3, 3] * (1.0 - ratio) + self.pose_all[idx_1, :3, 3] * ratio
        pose_0 = self.pose_all[idx_0].detach().cpu().numpy()
        pose_1 = self.pose_all[idx_1].detach().cpu().numpy()
        pose_0 = np.linalg.inv(pose_0)
        pose_1 = np.linalg.inv(pose_1)
        rot_0 = pose_0[:3, :3]
        rot_1 = pose_1[:3, :3]
        rots = Rot.from_matrix(np.stack([rot_0, rot_1]))
        key_times = [0, 1]
        slerp = Slerp(key_times, rots)
        rot = slerp(ratio)
        pose = np.diag([1.0, 1.0, 1.0, 1.0])
        pose = pose.astype(np.float32)
        pose[:3, :3] = rot.as_matrix()
        pose[:3, 3] = ((1.0 - ratio) * pose_0 + ratio * pose_1)[:3, 3]
        pose = np.linalg.inv(pose)
        rot = torch.from_numpy(pose[:3, :3]).cuda()
        trans = torch.from_numpy(pose[:3, 3]).cuda()
        rays_v = torch.matmul(rot[None, None, :3, :3], rays_v[:, :, :, None]).squeeze()  # W, H, 3
        rays_o = trans[None, None, :3].expand(rays_v.shape)  # W, H, 3
        return rays_o.transpose(0, 1), rays_v.transpose(0, 1)

    def near_far_from_sphere(self, rays_o, rays_d):
        a = torch.sum(rays_d**2, dim=-1, keepdim=True)
        b = 2.0 * torch.sum(rays_o * rays_d, dim=-1, keepdim=True)
        mid = 0.5 * (-b) / a
        near = mid - 1.0
        far = mid + 1.0
        return near, far

    def image_at(self, idx, resolution_level):
        """解像度を落として画像を返す"""
        if idx >= self.n_images:
            logging.warning(f"Image index {idx} out of range, using index 0 instead.")
            idx = 0
            
        if idx in self.bad_image_indices:
            # 問題のある画像の場合は別の画像を使用
            valid_indices = [i for i in range(self.n_images) if i not in self.bad_image_indices]
            if not valid_indices:
                raise RuntimeError("No valid images available!")
            idx = np.random.choice(valid_indices)
        
        image_path = self.images_lis[idx]
        try:
            img = safe_read_image(image_path)
            if img is None:
                self.bad_image_indices.add(idx)
                # エラーが発生した場合は別の画像を試す
                valid_indices = [i for i in range(self.n_images) if i not in self.bad_image_indices]
                if not valid_indices:
                    raise RuntimeError("No valid images available!")
                return self.image_at(np.random.choice(valid_indices), resolution_level)
        except Exception as e:
            self.bad_image_indices.add(idx)
            logging.error(f"Error loading image: {image_path}, {e}")
            # エラーが発生した場合は別の画像を試す
            valid_indices = [i for i in range(self.n_images) if i not in self.bad_image_indices]
            if not valid_indices:
                # エラーが発生した場合は、黒い画像を返す
                img = np.zeros((self.H, self.W, 3), dtype=np.uint8)
            else:
                return self.image_at(np.random.choice(valid_indices), resolution_level)
            
        return (cv.resize(img, (self.W // resolution_level, self.H // resolution_level))).clip(0, 255)