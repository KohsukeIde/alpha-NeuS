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
import re


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
        
        # 利用可能なカメラインデックスをソートして保持
        self.available_camera_indices = []
        for key in camera_dict.keys():
            if key.startswith('world_mat_'):
                try:
                    idx = int(key.split('_')[-1])
                    self.available_camera_indices.append(idx)
                except:
                    pass
                    
        # カメラインデックスをソート
        self.available_camera_indices.sort()
        
        if not self.available_camera_indices:
            raise RuntimeError("No camera matrices found in the camera file!")
            
        logging.info(f"Found {len(self.available_camera_indices)} camera matrices")
        
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
        
        # 有効な画像パスとマスクパスを保持する配列
        valid_images_lis = []
        valid_masks_lis = []
        self.world_mats_np = []
        self.scale_mats_np = []
        
        # 画像をテストロードして有効なものだけを保持
        camera_idx = 0  # カメラインデックスのカウンタ
        
        for i, (img_path, mask_path) in enumerate(zip(self.images_lis, self.masks_lis)):
            try:
                # 存在チェックと読み込みテスト
                if not os.path.exists(img_path) or not os.path.exists(mask_path):
                    logging.warning(f"Image or mask file not found, excluding: {img_path} or {mask_path}")
                    continue
                    
                img = cv.imread(img_path)
                if img is None:
                    logging.warning(f"Failed to load image, excluding: {img_path}")
                    continue
                    
                msk = cv.imread(mask_path)
                if msk is None:
                    logging.warning(f"Failed to load mask, excluding: {mask_path}")
                    continue
                
                # 利用可能なカメラインデックスをチェック
                if camera_idx >= len(self.available_camera_indices):
                    logging.warning(f"No more available camera matrices, excluding: {img_path}")
                    break
                
                # 現在のカメラインデックスを取得
                current_camera_idx = self.available_camera_indices[camera_idx]
                
                # カメラ行列の存在確認
                if f'world_mat_{current_camera_idx}' not in camera_dict or f'scale_mat_{current_camera_idx}' not in camera_dict:
                    logging.warning(f"Camera matrix {current_camera_idx} not found, skipping to next")
                    camera_idx += 1
                    continue
                
                # 有効な画像・マスクパスとカメラ行列を追加
                valid_images_lis.append(img_path)
                valid_masks_lis.append(mask_path)
                self.world_mats_np.append(camera_dict[f'world_mat_{current_camera_idx}'].astype(np.float32))
                self.scale_mats_np.append(camera_dict[f'scale_mat_{current_camera_idx}'].astype(np.float32))
                
                # 次のカメラインデックスへ
                camera_idx += 1
                
            except Exception as e:
                logging.error(f"Error checking image at index {i}: {e}")
        
        # 有効な画像リストに更新
        self.images_lis = valid_images_lis
        self.masks_lis = valid_masks_lis
        self.n_images = len(self.images_lis)
        
        if self.n_images == 0:
            raise RuntimeError("No valid images found!")
            
        logging.info(f"Found {self.n_images} valid image-mask pairs out of {orig_n_images}")
        
        # 初期サンプル画像を読み込んで寸法を取得
        if self.n_images > 0:
            sample_img = cv.imread(self.images_lis[0])
            self.img_h, self.img_w = sample_img.shape[:2]
            logging.info(f"Image dimensions: {self.img_w}x{self.img_h}")
        else:
            raise RuntimeError("No valid images to process!")
        
        # LRU画像キャッシュ（最大20枚までメモリに保持）
        self.image_cache = {}
        self.mask_cache = {}
        self.max_cache_size = 100

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
        
        # 背景画像の読み込み
        self.bkgds_lis = sorted(glob(os.path.join(self.data_dir, 'background/*.png')))
        if len(self.bkgds_lis) == 0:
            self.bkgds_lis = sorted(glob(os.path.join(self.data_dir, 'background/*.jpg')))
        
        # 元のNeUSと同様に背景画像を直接ロード（GPUに直接ロード）
        self.bkgds = []
        for im_name in self.bkgds_lis:
            bkgd_img = cv.imread(im_name)
            if bkgd_img is not None:
                # テンソル変換して直接GPU上に配置
                bkgd_tensor = torch.from_numpy(bkgd_img.astype(np.float32) / 256.0).to(self.device)
                self.bkgds.append(bkgd_tensor)
            else:
                # 読み込みに失敗した場合は黒画像
                self.bkgds.append(torch.zeros((self.H, self.W, 3), device=self.device))
                logging.warning(f"Failed to load background image: {im_name}, using black image")
        
        # デバッグ出力：背景画像とカメラIDの対応関係を表示
        print(f"DEBUG: Loaded {len(self.bkgds)} background images")
        for i, bkgd_path in enumerate(self.bkgds_lis):
            print(f"DEBUG: Background[{i}] = {os.path.basename(bkgd_path)}")
            
        # 各画像ファイルとカメラIDの対応関係を表示
        print("\nDEBUG: Image-Camera ID mapping:")
        for i in range(min(20, self.n_images)):  # 最大20個まで表示
            filename = os.path.basename(self.images_lis[i])
            camera_id = int(filename[1])
            print(f"DEBUG: Image[{i}] = {filename}, Camera ID = {camera_id}")
        if self.n_images > 20:
            print(f"DEBUG: ... and {self.n_images - 20} more images")
            
        # 画像数と背景画像数の不一致警告
        unique_camera_ids = set([int(os.path.basename(fname)[1]) for fname in self.images_lis])
        if len(unique_camera_ids) != len(self.bkgds):
            print(f"WARNING: Number of unique camera IDs ({len(unique_camera_ids)}) does not match number of background images ({len(self.bkgds)})")
            print(f"DEBUG: Unique camera IDs: {sorted(list(unique_camera_ids))}")
        
        # メモリクリーンアップ
        gc.collect()

    def _get_image(self, idx):
        """インデックスから画像を取得（キャッシュ機能付き）"""
        if idx >= self.n_images:
            raise IndexError(f"Image index {idx} out of range (0-{self.n_images-1})")
            
        # キャッシュをチェック
        if idx in self.image_cache:
            return self.image_cache[idx]
        
        # キャッシュになければ読み込む
        try:
            image_path = self.images_lis[idx]
            img = cv.imread(image_path)
            if img is None:
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
            logging.error(f"Error loading image at index {idx}: {e}")
            raise  # エラーを上位に伝播

    def _get_mask(self, idx):
        """インデックスからマスクを取得（キャッシュ機能付き）"""
        if idx >= self.n_images:
            raise IndexError(f"Mask index {idx} out of range (0-{self.n_images-1})")
            
        # キャッシュをチェック
        if idx in self.mask_cache:
            return self.mask_cache[idx]
        
        # キャッシュになければ読み込む
        try:
            mask_path = self.masks_lis[idx]
            msk = cv.imread(mask_path)
            if msk is None:
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
            logging.error(f"Error loading mask at index {idx}: {e}")
            raise  # エラーを上位に伝播

    def clear_cache(self):
        """キャッシュをクリアしてメモリを解放"""
        self.image_cache.clear()
        self.mask_cache.clear()
        gc.collect()
        logging.info("All caches cleared")
            
    def gen_rays_at(self, img_idx, resolution_level=1):
        """
        Generate rays at world space from one camera.
        """
        if img_idx >= self.n_images:
            raise IndexError(f"Image index {img_idx} out of range (0-{self.n_images-1})")
            
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
            raise IndexError(f"Image index {img_idx} out of range (0-{self.n_images-1})")
        
        # カメラIDとファイル名をデバッグ出力
        camera_id = self.camera_id_at(img_idx)
        filename = os.path.basename(self.images_lis[img_idx])
        print(f"DEBUG: gen_random_rays_at img_idx={img_idx}, filename={filename}, camera_id={camera_id}")
            
        pixels_x = torch.randint(low=0, high=self.W, size=[batch_size])
        pixels_y = torch.randint(low=0, high=self.H, size=[batch_size])
        
        # 指定された画像を読み込む
        image = self._get_image(img_idx).to(self.device)
        mask = self._get_mask(img_idx).to(self.device)
        
        color = image[(pixels_y, pixels_x)]    # batch_size, 3
        mask_value = mask[(pixels_y, pixels_x)]      # batch_size, 3
        mask_value = mask_value[:, :1]  # 最初のチャンネルだけを使用
        
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1).float()  # batch_size, 3
        p = torch.matmul(self.intrinsics_all_inv[img_idx, None, :3, :3], p[:, :, None]).squeeze() # batch_size, 3
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)    # batch_size, 3
        rays_v = torch.matmul(self.pose_all[img_idx, None, :3, :3], rays_v[:, :, None]).squeeze()  # batch_size, 3
        rays_o = self.pose_all[img_idx, None, :3, 3].expand(rays_v.shape) # batch_size, 3
        
        # 使用後はGPUメモリから削除（必要に応じて）
        del image
        del mask
        
        return torch.cat([rays_o, rays_v, color, mask_value], dim=-1).cuda()    # batch_size, 10

    def gen_rays_between(self, idx_0, idx_1, ratio, resolution_level=1):
        """
        Interpolate pose between two cameras.
        """
        # インデックスの範囲チェック
        if idx_0 >= self.n_images or idx_1 >= self.n_images:
            raise IndexError(f"Image indices {idx_0}, {idx_1} out of range (0-{self.n_images-1})")
            
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
            raise IndexError(f"Image index {idx} out of range (0-{self.n_images-1})")
        
        image_path = self.images_lis[idx]
        try:
            img = cv.imread(image_path)
            if img is None:
                raise IOError(f"Failed to load image: {image_path}")
        except Exception as e:
            logging.error(f"Error loading image: {image_path}, {e}")
            raise
            
        return (cv.resize(img, (self.W // resolution_level, self.H // resolution_level))).clip(0, 255)

    def camera_id_at(self, img_idx):
        """画像インデックスから背景用のカメラIDを返す（元のNeUSと同じ方式）"""
        if img_idx >= self.n_images:
            raise IndexError(f"Image index {img_idx} out of range (0-{self.n_images-1})")
        
        # 元のNeUSの実装と同様の単純な方法
        # ファイル名の2文字目（[1]）をカメラIDとして使用
        filename = os.path.basename(self.images_lis[img_idx])
        camera_id = int(filename[1])
        
        # デバッグ出力（カメラIDの確認）
        print(f"DEBUG: img_idx={img_idx}, filename={filename}, camera_id={camera_id}")
        
        # カメラIDがbkgdsの範囲内かチェック
        if camera_id >= len(self.bkgds):
            print(f"WARNING: Camera ID {camera_id} from filename {filename} is out of range (0-{len(self.bkgds)-1})")
        
        return camera_id

    def sample_rays_at(self, img_idx, resolution_level=1):
        """
        Sample pixels from one camera.
        """
        if img_idx >= self.n_images:
            raise IndexError(f"Image index {img_idx} out of range (0-{self.n_images-1})")
            
        # 解像度レベルに応じて線形空間を作成
        l = resolution_level
        
        # 画像をロード
        try:
            img = self._get_image(img_idx).to(self.device)
        except Exception as e:
            logging.error(f"Error loading image in sample_rays_at: {e}")
            raise
        
        # 画像サイズに基づいてピクセル座標を生成
        tx = torch.linspace(0, img.shape[1] - 1, img.shape[1] // l)
        ty = torch.linspace(0, img.shape[0] - 1, img.shape[0] // l)
        pixels_x, pixels_y = torch.meshgrid(tx, ty, indexing='ij')
        pixels_x = pixels_x.to(torch.int64).transpose(0, 1)
        pixels_y = pixels_y.to(torch.int64).transpose(0, 1)

        # カメラ空間のレイを生成
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1).float().to(self.device) # H, W, 3
        p = torch.matmul(self.intrinsics_all_inv[img_idx, None, None, :3, :3], p[:, :, :, None]).squeeze(-1)  # H, W, 3
        rays_camera = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)  # H, W, 3
            
        return rays_camera, pixels_x, pixels_y

    def gen_background_pixels_at(self, camera_idx, resolution_level=1):
        """背景画像からピクセル座標を生成"""
        # 背景画像を取得
        camera_id = self.camera_id_at(camera_idx)
        
        if camera_id < len(self.bkgds):
            bkgd = self.bkgds[camera_id]
        else:
            # カメラIDに対応する背景がない場合
            bkgd = torch.zeros((self.H, self.W, 3))
            
        l = resolution_level
        tx = torch.linspace(0, bkgd.shape[1] - 1, bkgd.shape[1] // l)
        ty = torch.linspace(0, bkgd.shape[0] - 1, bkgd.shape[0] // l)
        pixels_x, pixels_y = torch.meshgrid(tx, ty, indexing='ij')
        pixels_x = pixels_x.to(torch.int64)
        pixels_y = pixels_y.to(torch.int64)
        return pixels_x.transpose(0, 1), pixels_y.transpose(0, 1)

    def bkgd_at(self, idx, resolution_level):
        """特定のインデックスの背景画像を取得 (元のNeUS方式)"""
        if idx >= self.n_images:
            raise IndexError(f"Image index {idx} out of range (0-{self.n_images-1})")
            
        camera_id = self.camera_id_at(idx)
        filename = os.path.basename(self.images_lis[idx])
        print(f"DEBUG: bkgd_at() idx={idx}, filename={filename}, camera_id={camera_id}, bkgds_len={len(self.bkgds)}")
        
        # 背景画像の範囲チェック
        if camera_id >= len(self.bkgds):
            # エラーではなく黒画像を返す
            print(f"WARNING: Camera ID {camera_id} exceeds available backgrounds, using black image")
            return np.zeros((self.H // resolution_level, self.W // resolution_level, 3))
        
        # 背景画像を取得
        bkgd = self.bkgds[camera_id]
        print(f"DEBUG: Using background image for camera_id={camera_id}, shape={bkgd.shape}")
        
        # ROIデータの処理
        if hasattr(self, 'is_roi_data') and self.is_roi_data:
            bkgd = bkgd[
                self.roi_tops[idx]: self.roi_bottoms[idx],
                self.roi_lefts[idx]: self.roi_rights[idx]
                ]
                
        # リサイズして返す
        return sample_image_as_numpy(bkgd, resolution_level)