import os
import time
import logging
import argparse
import numpy as np
import cv2 as cv
import trimesh
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from shutil import copyfile
from icecream import ic
from tqdm import tqdm
from pyhocon import ConfigFactory
from models.dataset_lru import Dataset
from models.fields import RenderingNetwork, SDFNetwork, SingleVarianceNetwork, NeRF, MLP
from models.renderer import NeuSRenderer


class NeusBackgroundModule(torch.nn.Module):
    def __init__(self, dataset, conf) -> None:
        super().__init__()
        self.conf = conf
        self.dataset = dataset
        self.bkgd_mode = conf.get_string("train.bkgd_mode", default="fixed")
        self.use_init_bkgd = conf.get_bool("train.use_init_bkgd", default=False)
        print(f"bkgd_mode: {self.bkgd_mode}")
        print(f"use_init_bkgd: {self.use_init_bkgd}")

        self.nerf_outside = None
        if self.bkgd_mode != "fixed":
            if self.bkgd_mode == "nerf":
                self.nerf_outside = NeRF(**self.conf["model.nerf"])
            elif self.bkgd_mode == "mlp":
                self.bg_mlp = MLP(**self.conf["model.background_network"])
            elif self.bkgd_mode == "mlps":
                bg_mlps = []
                for i in range(self.dataset.n_images):
                    bg_mlp = MLP(**self.conf["model.background_network"])
                    bg_mlps.append(bg_mlp)
                self.bg_mlps = torch.nn.ModuleList(bg_mlps)
            elif self.bkgd_mode == "tensor":
                bg_tensor = torch.zeros(
                    [self.dataset.n_images, self.dataset.H, self.dataset.W, 4]
                )
                self.register_parameter("bg_tensor", torch.nn.Parameter(bg_tensor))

    def forward(self, camera_id, pixels_x, pixels_y, init_background_rgb):
        background_rgb = None
        if self.bkgd_mode == "fixed":
            background_rgb = init_background_rgb
        else:
            if self.bkgd_mode == "mlp":
                cidx = camera_id * torch.ones_like(pixels_x[..., None])
                cidx = (
                    cidx / (self.dataset.n_images - 1) * 2 - 1
                )  # range [0, n_cameras] -> [-1, 1]
                x = pixels_x / (self.dataset.W - 1) * 2 - 1  # range [0, W] -> [-1, 1]
                y = pixels_y / (self.dataset.H - 1) * 2 - 1  # range [0, H] -> [-1, 1]
                inputs = torch.cat([cidx, x, y], -1)
                background_rgb = self.bg_mlp(inputs)
            elif self.bkgd_mode == "mlps":
                x = (
                    pixels_x[:, None] / (self.dataset.W - 1) * 2 - 1
                )  # range [0, W] -> [-1, 1]
                y = (
                    pixels_y[:, None] / (self.dataset.H - 1) * 2 - 1
                )  # range [0, H] -> [-1, 1]
                inputs = torch.cat([x, y], -1)
                background_rgb = None
                for i in range(len(self.bg_mlps)):
                    bg = self.bg_mlps[i](inputs)
                    mask = (camera_id == i).float().view(-1, 1)
                    bg = bg * mask
                    if background_rgb is None:
                        background_rgb = bg
                    else:
                        background_rgb = background_rgb + bg
            elif self.bkgd_mode == "tensor":
                background_rgb = self.bg_tensor[camera_id][(pixels_y, pixels_x)]

            if self.use_init_bkgd and background_rgb is not None:
                if background_rgb.shape[-1] == 4:
                    rgb = background_rgb[:, 0:3]
                    a = background_rgb[:, 3:4]
                    background_rgb = (1 - a) * init_background_rgb + a * rgb
        
        return background_rgb


class Runner:
    def __init__(self, conf_path, mode='train', case='CASE_NAME', is_continue=False, ckpt=None):
        self.device = torch.device('cuda')

        # Configuration
        self.conf_path = conf_path
        f = open(self.conf_path)
        conf_text = f.read()
        conf_text = conf_text.replace('CASE_NAME', case)
        f.close()

        self.conf = ConfigFactory.parse_string(conf_text)
        self.conf['dataset.data_dir'] = self.conf['dataset.data_dir'].replace('CASE_NAME', case)
        self.base_exp_dir = self.conf['general.base_exp_dir']
        os.makedirs(self.base_exp_dir, exist_ok=True)
        self.dataset = Dataset(self.conf['dataset'])
        self.iter_step = 0

        # Training parameters
        self.end_iter = self.conf.get_int('train.end_iter')
        self.save_freq = self.conf.get_int('train.save_freq')
        self.report_freq = self.conf.get_int('train.report_freq')
        self.val_freq = self.conf.get_int('train.val_freq')
        self.val_mesh_freq = self.conf.get_int('train.val_mesh_freq')
        self.batch_size = self.conf.get_int('train.batch_size')
        self.validate_resolution_level = self.conf.get_int('train.validate_resolution_level')
        self.learning_rate = self.conf.get_float('train.learning_rate')
        self.learning_rate_alpha = self.conf.get_float('train.learning_rate_alpha')
        self.use_white_bkgd = self.conf.get_bool('train.use_white_bkgd')
        self.warm_up_end = self.conf.get_float('train.warm_up_end', default=0.0)
        self.anneal_end = self.conf.get_float('train.anneal_end', default=0.0)
        
        # Background parameters
        self.bkgd_mode = self.conf.get_string('train.bkgd_mode', default='fixed')

        # Weights
        self.igr_weight = self.conf.get_float('train.igr_weight')
        self.iso_weight = self.conf.get_float('train.iso_weight', default=0.0)
        self.mask_weight = self.conf.get_float('train.mask_weight')
        self.is_continue = is_continue
        self.mode = mode
        self.model_list = []
        self.writer = None

        # Networks
        params_to_train = []
        self.nerf_outside = NeRF(**self.conf['model.nerf']).to(self.device)
        self.sdf_network = SDFNetwork(**self.conf['model.sdf_network']).to(self.device)
        self.deviation_network = SingleVarianceNetwork(**self.conf['model.variance_network']).to(self.device)
        self.color_network = RenderingNetwork(**self.conf['model.rendering_network']).to(self.device)
        
        # Background module
        self.bkgd_module = NeusBackgroundModule(self.dataset, self.conf).to(self.device)
        
        params_to_train += list(self.nerf_outside.parameters())
        params_to_train += list(self.sdf_network.parameters())
        params_to_train += list(self.deviation_network.parameters())
        params_to_train += list(self.color_network.parameters())
        
        # オプティマイザー群
        self.optimizers = {}
        
        # メインオプティマイザー
        self.optimizers["main"] = torch.optim.Adam(params_to_train, lr=self.learning_rate)
        
        # 背景モジュール用オプティマイザー
        if self.bkgd_mode != 'fixed':
            self.learning_rate_bg = self.conf.get_float('train.learning_rate_bg', default=self.learning_rate)
            self.optimizers["bg"] = torch.optim.Adam(self.bkgd_module.parameters(), lr=self.learning_rate_bg)

        # レンダラー
        self.renderer = NeuSRenderer(self.nerf_outside,
                                     self.sdf_network,
                                     self.deviation_network,
                                     self.color_network,
                                     **self.conf['model.neus_renderer'])
                                     
        # ISO損失の重みをレンダラーに設定
        if hasattr(self.renderer, 'iso_weight'):
            self.renderer.iso_weight = self.iso_weight

        # Load checkpoint
        latest_model_name = None
        if is_continue:
            if ckpt is not None:
                latest_model_name = "ckpt_{:0>6d}.pth".format(ckpt)
            else:
                model_list_raw = os.listdir(os.path.join(self.base_exp_dir, 'checkpoints'))
                model_list = []
                for model_name in model_list_raw:
                    if model_name[-3:] == 'pth' and int(model_name[5:-4]) <= self.end_iter:
                        model_list.append(model_name)
                model_list.sort()
                latest_model_name = model_list[-1]

        if latest_model_name is not None:
            logging.info('Find checkpoint: {}'.format(latest_model_name))
            self.load_checkpoint(latest_model_name)

        # Backup codes and configs for debug
        if self.mode[:5] == 'train':
            self.file_backup()

    def train(self):
        self.writer = SummaryWriter(log_dir=os.path.join(self.base_exp_dir, 'logs'))
        self.update_learning_rate()
        res_step = self.end_iter - self.iter_step
        image_perm = self.get_image_perm()
        
        # パフォーマンス計測用の変数
        total_time = 0
        step_times = []
        
        for iter_i in tqdm(range(res_step)):
            start_time = time.time()
            
            # 画像インデックスを取得
            img_idx = image_perm[self.iter_step % len(image_perm)]
            
            # ランダムレイを生成
            data = self.dataset.gen_random_rays_at(img_idx, self.batch_size)
            rays_o, rays_d = data[:, :3], data[:, 3:6]
            true_rgb = data[:, 6:9]
            mask = data[:, 9:10]
            near, far = self.dataset.near_far_from_sphere(rays_o, rays_d)

            # カメラID取得
            camera_id = self.dataset.camera_id_at(img_idx)
            
            # デバッグ出力（1000イテレーションごとに表示）
            if self.iter_step % 1000 == 0:
                print(f"DEBUG: train() img_idx={img_idx}, camera_id={camera_id}")
            
            # ランダムピクセル座標を生成
            pixels_x = torch.randint(low=0, high=self.dataset.W, size=[self.batch_size], device=self.device)
            pixels_y = torch.randint(low=0, high=self.dataset.H, size=[self.batch_size], device=self.device)
            
            # 初期背景RGBの取得
            if hasattr(self.dataset, 'bkgds'):
                idx_tensor = torch.tensor(camera_id, device=self.device)
                init_background_rgb = self.dataset.bkgds[camera_id][
                    (pixels_y.to(torch.int64), pixels_x.to(torch.int64))
                ]
            else:
                # 背景画像がない場合は白か黒を使用
                if self.use_white_bkgd:
                    init_background_rgb = torch.ones([self.batch_size, 3], device=self.device)
                else:
                    init_background_rgb = torch.zeros([self.batch_size, 3], device=self.device)
            
            # 背景RGB
            background_rgb = self.bkgd_module(camera_id, pixels_x, pixels_y, init_background_rgb)
            
            # レンダリング
            render_out = self.renderer.render(rays_o,
                                            rays_d,
                                            near,
                                            far,
                                            cos_anneal_ratio=self.get_cos_anneal_ratio(),
                                            background_rgb=background_rgb)
            
            # マスク処理
            if self.mask_weight > 0.0:
                mask = (mask > 0.5).float()
            else:
                mask = torch.ones_like(mask)
                
            mask_sum = mask.sum() + 1e-5
            
            # ロス計算
            color_fine = render_out['color_fine']
            color_error = (color_fine - true_rgb) * mask
            color_fine_loss = F.l1_loss(color_error, torch.zeros_like(color_error), reduction='sum') / mask_sum
            
            s_val = render_out['s_val']
            eikonal_loss = render_out['gradient_error']
            
            weight_sum = render_out['weight_sum']
            mask_loss = F.binary_cross_entropy(weight_sum.clip(1e-3, 1.0 - 1e-3), mask)
            
            # ISO損失
            iso_loss = torch.tensor(0.0, device=self.device)
            if 'iso_loss' in render_out:
                iso_loss = render_out['iso_loss']
                if isinstance(iso_loss, torch.Tensor):
                    if iso_loss.numel() > 1:
                        iso_loss = iso_loss.mean()
            
            # 最終的なロス
            loss = color_fine_loss + \
                   eikonal_loss * self.igr_weight + \
                   mask_loss * self.mask_weight + \
                   iso_loss * self.iso_weight
            
            # 勾配計算と更新
            for optimizer in self.optimizers.values():
                optimizer.zero_grad()
                
            loss.backward()
            
            for optimizer in self.optimizers.values():
                optimizer.step()
            
            self.iter_step += 1
            
            # ログのみ数ステップおきに行うことでパフォーマンス向上
            if self.iter_step % self.report_freq == 0:
                # モニタリングデータの計算
                psnr = 20.0 * torch.log10(1.0 / (((color_fine - true_rgb)**2 * mask).sum() / (mask_sum * 3.0)).sqrt())
                cdf_fine = render_out['cdf_fine']
                weight_max = render_out['weight_max']
                
                # 学習率をログに記録
                for key, optimizer in self.optimizers.items():
                    self.writer.add_scalar(f'lr/{key}', optimizer.param_groups[0]['lr'], self.iter_step)
                
                # TensorBoardにログを記録
                self.writer.add_scalar('Loss/loss', loss, self.iter_step)
                self.writer.add_scalar('Loss/color_loss', color_fine_loss, self.iter_step)
                self.writer.add_scalar('Loss/eikonal_loss', eikonal_loss, self.iter_step)
                if self.iso_weight > 0:
                    self.writer.add_scalar('Loss/iso_loss', iso_loss, self.iter_step)
                self.writer.add_scalar('Statistics/s_val', s_val.mean(), self.iter_step)
                self.writer.add_scalar('Statistics/cdf', (cdf_fine[:, :1] * mask).sum() / mask_sum, self.iter_step)
                self.writer.add_scalar('Statistics/weight_max', (weight_max * mask).sum() / mask_sum, self.iter_step)
                self.writer.add_scalar('Statistics/psnr', psnr, self.iter_step)
                
                # ステップの実行時間を計測
                end_time = time.time()
                step_time = end_time - start_time
                total_time += step_time
                step_times.append(step_time)
                avg_time = total_time / len(step_times)
                
                print(self.base_exp_dir)
                print(f'iter:{self.iter_step:8>d} loss = {loss.item():.6f} lr={self.optimizers["main"].param_groups[0]["lr"]:.6f} time={step_time:.3f}s avg={avg_time:.3f}s')
            
            # 定期的な処理
            if self.iter_step % self.save_freq == 0:
                self.save_checkpoint()
                
            if self.iter_step % self.val_freq == 0:
                self.validate_image()
                
            if self.iter_step % self.val_mesh_freq == 0:
                print(f"Running mesh validation at iteration {self.iter_step} (val_mesh_freq={self.val_mesh_freq})")
                # 低解像度で試してみる（メモリ問題の診断用）
                self.validate_mesh(world_space=True, resolution=64, threshold=0.0)
                
            self.update_learning_rate()
            
            if self.iter_step % len(image_perm) == 0:
                image_perm = self.get_image_perm()
        
        print('Training finished!')

    def get_image_perm(self):
        return torch.randperm(self.dataset.n_images)

    def get_cos_anneal_ratio(self):
        if self.anneal_end == 0.0:
            return 1.0
        else:
            return np.min([1.0, self.iter_step / self.anneal_end])

    def update_learning_rate(self):
        if self.iter_step < self.warm_up_end:
            learning_factor = self.iter_step / self.warm_up_end
        else:
            alpha = self.learning_rate_alpha
            progress = (self.iter_step - self.warm_up_end) / (self.end_iter - self.warm_up_end)
            learning_factor = (np.cos(np.pi * progress) + 1.0) * 0.5 * (1 - alpha) + alpha

        for key, optimizer in self.optimizers.items():
            if key == "main":
                lr = self.learning_rate * learning_factor
            elif key == "bg" and hasattr(self, "learning_rate_bg"):
                lr = self.learning_rate_bg * learning_factor
            else:
                # その他のオプティマイザーがあれば、対応する学習率を取得
                attr_name = f"learning_rate_{key}"
                lr = getattr(self, attr_name, self.learning_rate) * learning_factor
                
            for g in optimizer.param_groups:
                g['lr'] = lr

    def file_backup(self):
        dir_lis = self.conf['general.recording']
        os.makedirs(os.path.join(self.base_exp_dir, 'recording'), exist_ok=True)
        for dir_name in dir_lis:
            cur_dir = os.path.join(self.base_exp_dir, 'recording', dir_name)
            os.makedirs(cur_dir, exist_ok=True)
            files = os.listdir(dir_name)
            for f_name in files:
                if f_name[-3:] == '.py':
                    copyfile(os.path.join(dir_name, f_name), os.path.join(cur_dir, f_name))

        copyfile(self.conf_path, os.path.join(self.base_exp_dir, 'recording', 'config.conf'))

    def load_checkpoint(self, checkpoint_name):
        # PyTorch 2.6以降では、weights_only=Falseを明示的に指定
        checkpoint = torch.load(
            os.path.join(self.base_exp_dir, 'checkpoints', checkpoint_name),
            map_location=self.device,
            weights_only=False  # セキュリティ制約を緩和
        )
        
        self.nerf_outside.load_state_dict(checkpoint['nerf'])
        self.sdf_network.load_state_dict(checkpoint['sdf_network_fine'])
        self.deviation_network.load_state_dict(checkpoint['variance_network_fine'])
        self.color_network.load_state_dict(checkpoint['color_network_fine'])
        
        # Load background module if exists in checkpoint
        if 'bkgd_module' in checkpoint and self.bkgd_mode != 'fixed':
            self.bkgd_module.load_state_dict(checkpoint['bkgd_module'])
            
        # オプティマイザーの読み込み
        self.optimizers["main"].load_state_dict(checkpoint['optimizer'])
        
        if "bg" in self.optimizers and 'bg_optimizer' in checkpoint:
            self.optimizers["bg"].load_state_dict(checkpoint['bg_optimizer'])
            
        self.iter_step = checkpoint['iter_step']

        logging.info('End') 

    def save_checkpoint(self):
        checkpoint = {
            'nerf': self.nerf_outside.state_dict(),
            'sdf_network_fine': self.sdf_network.state_dict(),
            'variance_network_fine': self.deviation_network.state_dict(),
            'color_network_fine': self.color_network.state_dict(),
            'optimizer': self.optimizers["main"].state_dict(),
            'iter_step': self.iter_step,
        }
        
        # Save background module if not in fixed mode
        if self.bkgd_mode != 'fixed':
            checkpoint['bkgd_module'] = self.bkgd_module.state_dict()
            
        # 追加のオプティマイザーの保存
        if "bg" in self.optimizers:
            checkpoint['bg_optimizer'] = self.optimizers["bg"].state_dict()

        os.makedirs(os.path.join(self.base_exp_dir, 'checkpoints'), exist_ok=True)
        torch.save(checkpoint, os.path.join(self.base_exp_dir, 'checkpoints', 'ckpt_{:0>6d}.pth'.format(self.iter_step)))

    def validate_image(self, idx=-1, resolution_level=-1):
        if idx < 0:
            idx = np.random.randint(self.dataset.n_images)

        print('Validate: iter: {}, camera: {}'.format(self.iter_step, idx))

        if resolution_level < 0:
            resolution_level = self.validate_resolution_level
            
        # レイを生成
        rays_o, rays_d = self.dataset.gen_rays_at(idx, resolution_level=resolution_level)
        H, W, _ = rays_o.shape
        rays_o = rays_o.reshape(-1, 3).split(self.batch_size)
        rays_d = rays_d.reshape(-1, 3).split(self.batch_size)

        # カメラIDを取得
        camera_id = self.dataset.camera_id_at(idx)
        print(f'DEBUG: validate_image camera_id: {camera_id} for image idx: {idx}')

        # ピクセル座標を取得
        _, pixels_x, pixels_y = self.dataset.sample_rays_at(idx, resolution_level=resolution_level)
        pixels_x = pixels_x.flatten().to(self.device)
        pixels_y = pixels_y.flatten().to(self.device)
        pixels_batch_size = self.batch_size
        pixels_x_splits = [pixels_x[i:i+pixels_batch_size] for i in range(0, pixels_x.shape[0], pixels_batch_size)]
        pixels_y_splits = [pixels_y[i:i+pixels_batch_size] for i in range(0, pixels_y.shape[0], pixels_batch_size)]

        # 結果を格納する配列
        out_rgb_fine = []
        out_normal_fine = []
        out_weight_sum = []

        # バッチごとにレンダリング
        for rays_o_batch, rays_d_batch, pixels_x_batch, pixels_y_batch in zip(rays_o, rays_d, pixels_x_splits, pixels_y_splits):
            near, far = self.dataset.near_far_from_sphere(rays_o_batch, rays_d_batch)
            
            # 初期背景RGB
            if hasattr(self.dataset, 'bkgds'):
                idx_tensor = torch.tensor(camera_id, device=self.device)
                init_background_rgb = self.dataset.bkgds[camera_id][
                    (pixels_y_batch.to(torch.int64), pixels_x_batch.to(torch.int64))
                ]
            else:
                # 背景画像がない場合は白か黒を使用
                if self.use_white_bkgd:
                    init_background_rgb = torch.ones([len(pixels_x_batch), 3], device=self.device)
                else:
                    init_background_rgb = torch.zeros([len(pixels_x_batch), 3], device=self.device)
            
            # 背景RGB
            background_rgb = self.bkgd_module(camera_id, pixels_x_batch, pixels_y_batch, init_background_rgb)
            
            # レンダリング
            render_out = self.renderer.render(rays_o_batch,
                                            rays_d_batch,
                                            near,
                                            far,
                                            cos_anneal_ratio=self.get_cos_anneal_ratio(),
                                            background_rgb=background_rgb)

            def feasible(key): 
                return (key in render_out) and (render_out[key] is not None)

            # 出力結果の取得
            if feasible('color_fine'):
                out_rgb_fine.append(render_out['color_fine'].detach().cpu().numpy())
                
            if feasible('weight_sum'):
                out_weight_sum.append(render_out['weight_sum'].detach().cpu().numpy())
                
            if feasible('gradients') and feasible('weights'):
                n_samples = self.renderer.n_samples + self.renderer.n_importance
                normals = render_out['gradients'] * render_out['weights'][:, :n_samples, None]
                if feasible('inside_sphere'):
                    normals = normals * render_out['inside_sphere'][..., None]
                normals = normals.sum(dim=1).detach().cpu().numpy()
                out_normal_fine.append(normals)
            del render_out

        # 結果の処理
        img_fine = None
        if len(out_rgb_fine) > 0:
            img_fine = (np.concatenate(out_rgb_fine, axis=0).reshape([H, W, 3, -1]) * 256).clip(0, 255)
            
        # アルファ情報（前景マスク）
        alpha_fine = None
        if len(out_weight_sum) > 0:
            alpha_fine = np.concatenate(out_weight_sum, axis=0).reshape([H, W, -1])

        normal_img = None
        if len(out_normal_fine) > 0:
            normal_img = np.concatenate(out_normal_fine, axis=0)
            rot = np.linalg.inv(self.dataset.pose_all[idx, :3, :3].detach().cpu().numpy())
            normal_img = (np.matmul(rot[None, :, :], normal_img[:, :, None])
                        .reshape([H, W, 3, -1]) * 128 + 128).clip(0, 255)

        # ディレクトリ作成
        os.makedirs(os.path.join(self.base_exp_dir, 'validations_fine'), exist_ok=True)
        os.makedirs(os.path.join(self.base_exp_dir, 'normals'), exist_ok=True)

        # 結果の保存
        for i in range(img_fine.shape[-1]):
            if len(out_rgb_fine) > 0:
                # Ground Truth画像を取得
                gt_img = self.dataset.image_at(idx, resolution_level)
                
                # レンダリング結果とGround Truthを並べて保存
                cv.imwrite(os.path.join(self.base_exp_dir,
                                      'validations_fine',
                                      '{:0>8d}_{}_{}.png'.format(self.iter_step, i, idx)),
                          np.concatenate([img_fine[..., i].astype(np.uint8), gt_img]))
                
            if len(out_normal_fine) > 0:
                cv.imwrite(os.path.join(self.base_exp_dir,
                                      'normals',
                                      '{:0>8d}_{}_{}.png'.format(self.iter_step, i, idx)),
                          normal_img[..., i])

    def render_novel_image(self, idx_0, idx_1, ratio, resolution_level):
        """
        Interpolate view between two cameras.
        """
        rays_o, rays_d = self.dataset.gen_rays_between(idx_0, idx_1, ratio, resolution_level=resolution_level)
        H, W, _ = rays_o.shape
        rays_o = rays_o.reshape(-1, 3).split(self.batch_size)
        rays_d = rays_d.reshape(-1, 3).split(self.batch_size)

        out_rgb_fine = []
        for rays_o_batch, rays_d_batch in zip(rays_o, rays_d):
            near, far = self.dataset.near_far_from_sphere(rays_o_batch, rays_d_batch)
            
            # 補間ビューでは白背景または背景なしで実行
            if self.use_white_bkgd:
                background_rgb = torch.ones([1, 3], device=self.device)
            else:
                background_rgb = None

            render_out = self.renderer.render(rays_o_batch,
                                            rays_d_batch,
                                            near,
                                            far,
                                            cos_anneal_ratio=self.get_cos_anneal_ratio(),
                                            background_rgb=background_rgb)

            out_rgb_fine.append(render_out['color_fine'].detach().cpu().numpy())
            del render_out

        img_fine = (np.concatenate(out_rgb_fine, axis=0).reshape([H, W, 3]) * 256).clip(0, 255).astype(np.uint8)
        return img_fine

    def validate_mesh(self, world_space=False, resolution=64, threshold=0.0, scale=1.0):
        logging.info(f"Starting mesh validation: resolution={resolution}, threshold={threshold}, scale={scale}")
        
        # バウンディングボックスの設定
        bound_min = torch.tensor(self.dataset.object_bbox_min, dtype=torch.float32)/scale
        bound_max = torch.tensor(self.dataset.object_bbox_max, dtype=torch.float32)/scale
        logging.info(f"Mesh bounds: min={bound_min.tolist()}, max={bound_max.tolist()}")

        # SDFネットワークの状態確認
        test_points = torch.rand((10, 3), device=self.device) * (bound_max - bound_min) + bound_min
        test_sdf = self.sdf_network.sdf(test_points)
        logging.info(f"SDF test values: {test_sdf.detach().cpu().numpy()}")

        # ジオメトリ抽出
        logging.info("Extracting geometry...")
        vertices, triangles =\
            self.renderer.extract_geometry(bound_min, bound_max, resolution=resolution, threshold=threshold)
            
        logging.info(f"Extraction complete. Vertices: {vertices.shape if vertices is not None else 'None'}, "
                   f"Triangles: {triangles.shape if triangles is not None else 'None'}")
        
        # 結果の確認
        if vertices is None or triangles is None:
            logging.error("No geometry extracted! Vertices or triangles is None.")
            return
            
        if len(vertices) == 0 or len(triangles) == 0:
            logging.error(f"Empty geometry extracted! Vertices: {len(vertices)}, Triangles: {len(triangles)}")
            return
            
        logging.info(f"Geometry extraction successful. Vertices: {len(vertices)}, Triangles: {len(triangles)}")
        
        # メッシュフォルダの作成
        mesh_dir = os.path.join(self.base_exp_dir, 'meshes')
        logging.info(f"Creating mesh directory: {mesh_dir}")
        os.makedirs(mesh_dir, exist_ok=True)

        # ワールド座標系への変換
        if world_space:
            logging.info("Transforming to world space...")
            vertices = vertices * self.dataset.scale_mats_np[0][0, 0] + self.dataset.scale_mats_np[0][:3, 3][None]
            logging.info("Transformation complete")

        # メッシュの作成と保存
        logging.info("Creating mesh...")
        mesh = trimesh.Trimesh(vertices, triangles)
        mesh_path = os.path.join(mesh_dir, '{:0>8d}.ply'.format(self.iter_step))
        
        logging.info(f"Saving mesh to {mesh_path}...")
        mesh.export(mesh_path)
        logging.info(f"Mesh saved successfully")

        logging.info('Mesh validation completed')

    def validate_dcudf(self, world_space=False, resolution=64, threshold=0.0, is_cut=False, scale=1.0):
        from dcudf.mesh_extraction import dcudf
        bound_min = torch.tensor(self.dataset.object_bbox_min, dtype=torch.float32)/scale
        bound_max = torch.tensor(self.dataset.object_bbox_max, dtype=torch.float32)/scale

        extractor = dcudf(
            lambda pts: self.sdf_network.sdf(pts),
            resolution,
            threshold,
            query_func_optim=lambda pts: torch.abs(self.sdf_network.sdf(pts)),
            bound_min=bound_min,
            bound_max=bound_max,
            laplacian_weight=500.0,
            is_cut=is_cut
        )
        mesh = extractor.optimize()
        vertices = mesh.vertices
        os.makedirs(os.path.join(self.base_exp_dir, 'udf_meshes'), exist_ok=True)
        if world_space:
            vertices = vertices * self.dataset.scale_mats_np[0][0, 0] + self.dataset.scale_mats_np[0][:3, 3][None]
        mesh = trimesh.Trimesh(vertices, mesh.faces)
        mesh.export(os.path.join(self.base_exp_dir, 'udf_meshes', '{:0>8d}.ply'.format(self.iter_step)))

        logging.info('End')

    def interpolate_view(self, img_idx_0, img_idx_1):
        images = []
        n_frames = 60
        for i in range(n_frames):
            print(i)
            images.append(self.render_novel_image(img_idx_0,
                                                  img_idx_1,
                                                  np.sin(((i / n_frames) - 0.5) * np.pi) * 0.5 + 0.5,
                                                  resolution_level=4))
        for i in range(n_frames):
            images.append(images[n_frames - i - 1])

        fourcc = cv.VideoWriter_fourcc(*'mp4v')
        video_dir = os.path.join(self.base_exp_dir, 'render')
        os.makedirs(video_dir, exist_ok=True)
        h, w, _ = images[0].shape
        writer = cv.VideoWriter(os.path.join(video_dir,
                                             '{:0>8d}_{}_{}.mp4'.format(self.iter_step, img_idx_0, img_idx_1)),
                                fourcc, 30, (w, h))

        for image in images:
            writer.write(image)

        writer.release()


if __name__ == '__main__':
    print('Hello α-NeUS')

    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
    logging.basicConfig(level=logging.DEBUG, format=FORMAT)

    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, default='./confs/base.conf')
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--mcube_threshold', type=float, default=0.0)
    parser.add_argument('--is_continue', default=False, action="store_true")
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--case', type=str, default='')
    parser.add_argument('--ckpt', type=int, default=None)
    parser.add_argument('--scale', type=float, default=1.0)
    parser.add_argument('--idx', type=int, default=-1)
    parser.add_argument('--mesh_resolution', type=int, default=64, help='Resolution for mesh extraction')

    args = parser.parse_args()

    torch.cuda.set_device(args.gpu)
    runner = Runner(args.conf, args.mode, args.case, args.is_continue, args.ckpt)

    if args.mode == 'train':
        runner.train()
    elif args.mode == 'validate_image':
        runner.validate_image(idx=args.idx, resolution_level=1)
    elif args.mode == 'validate_mesh':
        print(f"Running validate_mesh with resolution={args.mesh_resolution}, threshold={args.mcube_threshold}")
        runner.validate_mesh(world_space=True, resolution=args.mesh_resolution, threshold=args.mcube_threshold, scale=args.scale)
    elif args.mode == 'validate_dcudf':
        runner.validate_dcudf(world_space=True, resolution=512, threshold=args.mcube_threshold, is_cut=False, scale=args.scale)
    elif args.mode.startswith('interpolate'):  # Interpolate views given two image indices
        _, img_idx_0, img_idx_1 = args.mode.split('_')
        img_idx_0 = int(img_idx_0)
        img_idx_1 = int(img_idx_1)
        runner.interpolate_view(img_idx_0, img_idx_1)