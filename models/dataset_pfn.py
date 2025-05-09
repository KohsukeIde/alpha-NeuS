import torch
import torch.nn.functional as F
import cv2 as cv
import numpy as np
import os
import re
from glob import glob
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial.transform import Slerp
import logging
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


# ファイル名からインデックスを抽出する関数
def extract_idx_from_filename(filename):
    """ファイル名からカメラ番号と画像インデックスを抽出する"""
    # ファイル名のみを取得（パスとファイル拡張子を除去）
    basename = os.path.basename(filename)
    name_without_ext = os.path.splitext(basename)[0]
    
    # 6桁のパターン: 最初の2桁がカメラ番号、最後の4桁が画像インデックス
    if len(name_without_ext) == 6 and name_without_ext.isdigit():
        camera_id = int(name_without_ext[:2])
        return camera_id
    
    # 従来のフォールバック処理
    numbers = re.findall(r'\d+', name_without_ext)
    if len(numbers) > 0:
        return int(numbers[0])  # 最初の数字をカメラIDとして返す
    else:
        return 0  # デフォルト値


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
        
        self.available_camera_indices.sort()  # インデックスをソート
        
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
            
        # 背景画像の読み込み
        self.bkgds_lis = sorted(glob(os.path.join(self.data_dir, 'background/*.png')))
        if len(self.bkgds_lis) == 0:
            self.bkgds_lis = sorted(glob(os.path.join(self.data_dir, 'background/*.jpg')))
        
        # 有効な画像パスとマスクパスを保持する配列
        valid_images_lis = []
        valid_masks_lis = []
        self.images = []
        self.masks = []
        self.bkgds = []
        self.world_mats_np = []
        self.scale_mats_np = []
        self.file_index_to_camera_index = {}  # 背景画像用のマッピング
        
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
                
                # 有効な画像データをCPUに保持（GPUではなく）
                self.images.append(torch.from_numpy(img).cpu())
                self.masks.append(torch.from_numpy(msk).cpu())
                
                # カメラ行列を追加
                self.world_mats_np.append(camera_dict[f'world_mat_{current_camera_idx}'].astype(np.float32))
                self.scale_mats_np.append(camera_dict[f'scale_mat_{current_camera_idx}'].astype(np.float32))
                
                # 背景画像用に画像ファイル名からカメラIDを抽出してマッピング
                # この情報は背景画像のロード時にのみ使用
                extracted_camera_id = extract_idx_from_filename(img_path)
                self.file_index_to_camera_index[len(valid_images_lis) - 1] = extracted_camera_id
                
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
        
        # 画像サイズを取得
        if self.n_images > 0:
            self.H, self.W = self.images[0].shape[0], self.images[0].shape[1]
            logging.info(f"Image dimensions: {self.W}x{self.H}")
            self.image_pixels = self.H * self.W
        else:
            raise RuntimeError("No valid images found!")

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
        
        # オプショナル: メモリ解放のためCPUから完全に削除するメソッド
        self.is_images_dropped = False
        
        # 背景画像の読み込み（対応するカメラIDのみ保持）
        self.bkgds = []
        for i in range(self.n_images):
            camera_id = self.file_index_to_camera_index.get(i, 0)  # デフォルトは0
            try:
                if camera_id < len(self.bkgds_lis):
                    bkgd_path = self.bkgds_lis[camera_id]
                    if os.path.exists(bkgd_path):
                        bkgd = cv.imread(bkgd_path)
                        if bkgd is not None:
                            self.bkgds.append(torch.from_numpy(bkgd).cpu())
                            continue
                # 上記の条件に合わない場合は黒背景を追加
                self.bkgds.append(torch.zeros((self.H, self.W, 3)).cpu())
            except Exception as e:
                logging.error(f"Error loading background image for camera {camera_id}: {e}")
                self.bkgds.append(torch.zeros((self.H, self.W, 3)).cpu())
        
    def drop_images_and_masks(self):
        """メモリ節約のために画像とマスクデータをメモリから削除"""
        self.is_images_dropped = True
        i = self.images
        self.images = []
        del i

        m = self.masks
        self.masks = []
        del m
        
        b = self.bkgds
        self.bkgds = []
        del b
        
        gc.collect()

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
            
        pixels_x = torch.randint(low=0, high=self.W, size=[batch_size])
        pixels_y = torch.randint(low=0, high=self.H, size=[batch_size])
        
        # データが削除されている場合は再読み込みを試みる
        if self.is_images_dropped:
            # 画像ファイルを読み込む
            try:
                image_path = self.images_lis[img_idx]
                mask_path = self.masks_lis[img_idx]
                
                img = cv.imread(image_path)
                if img is None:
                    raise IOError(f"Failed to load image: {image_path}")
                    
                msk = cv.imread(mask_path)
                if msk is None:
                    raise IOError(f"Failed to load mask: {mask_path}")
                    
                image = torch.from_numpy(img.astype(np.float32) / 256.0).to(self.device)
                mask = torch.from_numpy(msk.astype(np.float32) / 256.0).to(self.device)
                
                color = image[(pixels_y, pixels_x)]    # batch_size, 3
                mask_value = mask[(pixels_y, pixels_x)]      # batch_size, 3
                mask_value = mask_value[:, :1]  # 最初のチャンネルだけを使用
                
                # 使用後はGPUメモリから削除
                del image
                del mask
            except Exception as e:
                logging.error(f"Error loading image for rays: {e}")
                # エラーが発生した場合は、ダミーデータで代用
                color = torch.zeros([batch_size, 3]).to(self.device)
                mask_value = torch.zeros([batch_size, 1]).to(self.device)
        else:
            # 画像からデータを取得
            image = self.images[img_idx].to(self.device) / 256.0
            mask = self.masks[img_idx].to(self.device) / 256.0
            
            color = image[(pixels_y, pixels_x)]    # batch_size, 3
            mask_value = mask[(pixels_y, pixels_x)]      # batch_size, 3
            mask_value = mask_value[:, :1]  # 最初のチャンネルだけを使用
        
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1).float()  # batch_size, 3
        p = torch.matmul(self.intrinsics_all_inv[img_idx, None, :3, :3], p[:, :, None]).squeeze() # batch_size, 3
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)    # batch_size, 3
        rays_v = torch.matmul(self.pose_all[img_idx, None, :3, :3], rays_v[:, :, None]).squeeze()  # batch_size, 3
        rays_o = self.pose_all[img_idx, None, :3, 3].expand(rays_v.shape) # batch_size, 3
        
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

    def sample_rays_at(self, img_idx, resolution_level=1):
        """
        指定されたカメラからの光線サンプリング情報を生成します
        """
        if img_idx >= self.n_images:
            raise IndexError(f"Image index {img_idx} out of range (0-{self.n_images-1})")
            
        l = resolution_level
        tx = torch.linspace(0, self.W - 1, self.W // l)
        ty = torch.linspace(0, self.H - 1, self.H // l)
        pixels_x, pixels_y = torch.meshgrid(tx, ty)
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1) # W, H, 3
        p = torch.matmul(self.intrinsics_all_inv[img_idx, None, None, :3, :3], p[:, :, :, None]).squeeze(-1)  # H, W, 3
        rays_camera = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)  # H, W, 3
            
        return rays_camera, pixels_x, pixels_y

    def image_at(self, idx, resolution_level):
        """解像度を落として画像を返す"""
        if idx >= self.n_images:
            raise IndexError(f"Image index {idx} out of range (0-{self.n_images-1})")
        
        # メモリから削除されている場合はファイルから読み込む
        if self.is_images_dropped:
            image_path = self.images_lis[idx]
            try:
                img = cv.imread(image_path)
                if img is None:
                    raise IOError(f"Failed to load image: {image_path}")
                    
                return (cv.resize(img, (self.W // resolution_level, self.H // resolution_level))).clip(0, 255)
            except Exception as e:
                logging.error(f"Error loading image: {image_path}, {e}")
                raise
        else:
            # メモリに保持している画像からサンプリング
            return sample_image_as_numpy(self.images[idx].to(self.device), resolution_level) * 256.0

    def camera_id_at(self, img_idx):
        """有効なイメージインデックスから対応するカメラIDを返す"""
        if img_idx >= self.n_images:
            raise IndexError(f"Image index {img_idx} out of range (0-{self.n_images-1})")
            
        camera_id = self.file_index_to_camera_index.get(img_idx, 0)  # デフォルトは0
        img_path = self.images_lis[img_idx] if img_idx < len(self.images_lis) else "unknown"
        logging.debug(f"camera_id_at({img_idx}): path={os.path.basename(img_path)}, camera_id={camera_id}")
        return camera_id

    def gen_background_pixels_at(self, camera_idx, resolution_level=1):
        """背景画像からピクセル座標を生成"""
        # 背景画像インデックスの確認
        if camera_idx >= self.n_images:
            raise IndexError(f"Camera index {camera_idx} out of range (0-{self.n_images-1})")
            
        # 背景画像を取得
        if self.is_images_dropped or camera_idx >= len(self.bkgds):
            # 失われたデータの場合はデフォルトのサイズを使用
            bkgd_shape = (self.H, self.W)
        else:
            # 背景画像がある場合、その形状を使用
            bkgd_shape = self.bkgds[camera_idx].shape
            
        l = resolution_level
        tx = torch.linspace(0, bkgd_shape[1] - 1, bkgd_shape[1] // l)
        ty = torch.linspace(0, bkgd_shape[0] - 1, bkgd_shape[0] // l)
        pixels_x, pixels_y = torch.meshgrid(tx, ty, indexing='ij')
        pixels_x = pixels_x.to(torch.int64)
        pixels_y = pixels_y.to(torch.int64)
        return pixels_x.transpose(0, 1), pixels_y.transpose(0, 1)

    def bkgd_at(self, idx, resolution_level=1):
        """特定のインデックスの背景画像を取得"""
        if idx >= self.n_images:
            raise IndexError(f"Image index {idx} out of range (0-{self.n_images-1})")
            
        camera_id = self.camera_id_at(idx)
        logging.debug(f"bkgd_at({idx}): Camera ID = {camera_id}")
        
        # 背景画像の取得
        if self.is_images_dropped:
            # メモリから削除されている場合はファイルから読み込む
            if camera_id < len(self.bkgds_lis):
                bkgd_path = self.bkgds_lis[camera_id]
                logging.debug(f"bkgd_at({idx}): Loading from file {os.path.basename(bkgd_path)}")
                if not os.path.exists(bkgd_path):
                    raise IOError(f"Background file does not exist: {bkgd_path}")
                    
                bkgd_img = cv.imread(bkgd_path)
                if bkgd_img is None:
                    raise IOError(f"Failed to load background image: {bkgd_path}")
                    
                logging.debug(f"bkgd_at({idx}): Original background image shape {bkgd_img.shape}")
                # 背景画像のサイズを元の画像サイズにリサイズ
                if bkgd_img.shape[0] != self.H or bkgd_img.shape[1] != self.W:
                    logging.info(f"bkgd_at({idx}): Resizing background from {bkgd_img.shape[:2]} to {(self.H, self.W)}")
                    bkgd_img = cv.resize(bkgd_img, (self.W, self.H), interpolation=cv.INTER_AREA)
                bkgd = torch.from_numpy(bkgd_img).float().to(self.device) / 256.0
            else:
                # カメラIDに対応する背景がない場合
                raise RuntimeError(f"bkgd_at({idx}): Camera ID {camera_id} exceeds background list length {len(self.bkgds_lis)}")
        elif idx < len(self.bkgds):
            # メモリに保持している背景画像を使用
            logging.debug(f"bkgd_at({idx}): Using cached background for camera ID {camera_id}")
            bkgd = self.bkgds[idx].to(self.device) / 256.0
        else:
            # カメラIDに対応する背景がない場合
            raise RuntimeError(f"bkgd_at({idx}): Image index {idx} exceeds background list length {len(self.bkgds)}")
                
        # ROIデータの場合は切り出し
        if hasattr(self, 'is_roi_data') and self.is_roi_data:
            bkgd = bkgd[
                self.roi_tops[idx]: self.roi_bottoms[idx],
                self.roi_lefts[idx]: self.roi_rights[idx]
                ]
        
        result = sample_image_as_numpy(bkgd, resolution_level)
        logging.debug(f"bkgd_at({idx}): Returning background shape {result.shape}")
        return result