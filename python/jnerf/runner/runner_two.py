import os
import jittor as jt
from PIL import Image
import numpy as np
from tqdm import tqdm
from jnerf.ops.code_ops import *
from jnerf.dataset.dataset import jt_srgb_to_linear, jt_linear_to_srgb
from jnerf.utils.config import get_cfg, save_cfg
from jnerf.utils.registry import build_from_cfg,NETWORKS,SCHEDULERS,DATASETS,OPTIMS,SAMPLERS,LOSSES
from jnerf.models.losses.mse_loss import img2mse, mse2psnr
from jnerf.dataset import camera_path
import cv2

class Runner_two():
    def __init__(self):
        self.cfg = get_cfg()
        self.cfg.dataset_dir_content = self.cfg.dataset_dir_content + self.cfg.exp_name
        self.cfg.dataset_dir_style = self.cfg.dataset_dir_style + self.cfg.sty_name
        print(self.cfg.exp_name+'_and_' +self.cfg.sty_name)
        self.cfg.dataset_content.train.root_dir = self.cfg.dataset_dir_content
        self.cfg.dataset_content.test.root_dir = self.cfg.dataset_dir_content
        self.cfg.dataset_style.train.root_dir = self.cfg.dataset_dir_style
        self.cfg.dataset_style.test.root_dir = self.cfg.dataset_dir_style

        if self.cfg.fp16 and jt.flags.cuda_archs[0] < 70:
            print("Warning: Sm arch is lower than sm_70, fp16 is not supported. Automatically use fp32 instead.")
            self.cfg.fp16 = False
        if not os.path.exists(self.cfg.log_dir):
            os.makedirs(self.cfg.log_dir)
        self.exp_name           = self.cfg.exp_name
        self.sty_name           = self.cfg.sty_name
        self.dataset_content            = {}
        self.dataset_content["train"]   = build_from_cfg(self.cfg.dataset_content.train, DATASETS)
        self.cfg.dataset_content_obj    = self.dataset_content["train"]
        if self.cfg.dataset_content.val:
            self.dataset_content["val"] = build_from_cfg(self.cfg.dataset_content.val, DATASETS)
        else:
            self.dataset_content["val"] = self.dataset_content["train"]
        self.dataset_content["test"]    = None
        
        self.dataset_style            = {}
        self.dataset_style["train"]   = build_from_cfg(self.cfg.dataset_style.train, DATASETS)
        self.cfg.dataset_style_obj    = self.dataset_style["train"]
        if self.cfg.dataset_style.val:
            self.dataset_style["val"] = build_from_cfg(self.cfg.dataset_style.val, DATASETS)
        else:
            self.dataset_style["val"] = self.dataset_style["train"]
        self.dataset_style["test"]    = None  

        self.model              = build_from_cfg(self.cfg.model, NETWORKS)
        self.cfg.model_obj      = self.model

        self.sampler_content            = build_from_cfg(self.cfg.sampler, SAMPLERS, cfg_dataset_obj=self.cfg.dataset_content_obj, content_flag=1)
        self.cfg.sampler_obj_content    = self.sampler_content
        self.sampler_style            = build_from_cfg(self.cfg.sampler, SAMPLERS, cfg_dataset_obj=self.cfg.dataset_style_obj, content_flag=0, update_den_freq=17)
        self.cfg.sampler_obj_style    = self.sampler_style

        self.optimizer          = build_from_cfg(self.cfg.optim, OPTIMS, params=self.model.parameters())
        self.optimizer          = build_from_cfg(self.cfg.expdecay, OPTIMS, nested_optimizer=self.optimizer)
        self.ema_optimizer      = build_from_cfg(self.cfg.ema, OPTIMS, params=self.model.parameters())
        self.loss_func          = build_from_cfg(self.cfg.loss, LOSSES)
        self.background_color   = self.cfg.background_color
        self.tot_train_steps    = self.cfg.tot_train_steps
        self.n_rays_per_batch   = self.cfg.n_rays_per_batch
        self.using_fp16         = self.cfg.fp16
        self.save_path          = os.path.join(self.cfg.log_dir, self.exp_name + '_and_' + self.sty_name)
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        if self.cfg.ckpt_path and self.cfg.ckpt_path is not None:
            self.ckpt_path = self.cfg.ckpt_path
        else:
            self.ckpt_path = os.path.join(self.save_path, "params.pkl")
        if self.cfg.load_ckpt:
            self.load_ckpt(self.ckpt_path)
        else:
            self.start=0
        self.alpha_image=self.cfg.alpha_image

        self.cfg.m_training_step = 0
        self.val_freq = 4096
        self.image_resolutions_content = self.dataset_content["train"].resolution
        self.image_resolutions_style = self.dataset_style["train"].resolution

        self.volume_feat_content = None
        self.volume_feat_style = None
        
    def save_rgb_mlp(self):
        self.load_ckpt(self.ckpt_path)
        model = self.model
        rgb_mlp = model.rgb_mlp
        jt.save({
            'rgb_mlp': rgb_mlp.state_dict(),
        }, './models/params.pkl')        
        qqq = 1

    def train(self):
        for i in tqdm(range(self.start, self.tot_train_steps)):
            self.cfg.m_training_step = i
            if i%2 == 0:
                content_flag = 1
                img_ids, rays_o, rays_d, rgb_target = next(self.dataset_content["train"])
                pos, dir = self.sampler_content.sample(img_ids, rays_o, rays_d, is_training=True)
            else:
                content_flag = 0
                img_ids, rays_o, rays_d, rgb_target = next(self.dataset_style["train"])
                pos, dir = self.sampler_style.sample(img_ids, rays_o, rays_d, is_training=True)
            training_background_color = jt.random([rgb_target.shape[0],3]).stop_grad()

            rgb_target = (rgb_target[..., :3] * rgb_target[..., 3:] + training_background_color * (1 - rgb_target[..., 3:])).detach()

            network_outputs  = self.model(pos, dir, content_flag)

            if i%2 == 0:
                rgb = self.sampler_content.rays2rgb(network_outputs, training_background_color)
            else:
                rgb = self.sampler_style.rays2rgb(network_outputs, training_background_color)

            loss = self.loss_func(rgb, rgb_target)
            self.optimizer.step(loss)

            self.ema_optimizer.ema_step()
            if self.using_fp16:
                self.model.set_fp16()

            if i>0 and i%self.val_freq==0:
                psnr=mse2psnr(self.val_img(i, content_flag=1))
                print("STEP={} | C_LOSS={} | VAL PSNR={}".format(i,loss.mean().item(), psnr))
                psnr=mse2psnr(self.val_img(i, content_flag=0))
                print("STEP={} | S_LOSS={} | VAL PSNR={}".format(i,loss.mean().item(), psnr))
        self.save_ckpt(os.path.join(self.save_path, "params.pkl"))
        self.test()
        self.test_style_two(False)
    
    def test(self, load_ckpt=False):
        if load_ckpt:
            assert os.path.exists(self.ckpt_path), "ckpt file does not exist: "+self.ckpt_path
            self.load_ckpt(self.ckpt_path)
        if self.dataset_content["test"] is None:
            self.dataset_content["test"] = build_from_cfg(self.cfg.dataset_content.test, DATASETS)
        if self.dataset_style["test"] is None:
            self.dataset_style["test"] = build_from_cfg(self.cfg.dataset_style.test, DATASETS)            
        if not os.path.exists(os.path.join(self.save_path, "test")):
            os.makedirs(os.path.join(self.save_path, "test"))
        mse_list=self.render_test(save_path=os.path.join(self.save_path, "test"),content_flag=1)
        mse_list=self.render_test(save_path=os.path.join(self.save_path, "test"),content_flag=0)
        '''if self.dataset["test"].have_img:
            tot_psnr=0
            for mse in mse_list:
                tot_psnr += mse2psnr(mse)
            print("TOTAL TEST PSNR===={}".format(tot_psnr/len(mse_list)))'''

    def calc_mean_std(self, feat, eps=1e-5):
        feat = feat.permute(0,2,1)
        # eps is a small value added to the variance to avoid divide-by-zero.
        size = feat.size()
        N, C = size[:2]
        feat_var = feat.view(N, C, -1).var(dim=2) + eps
        feat_std = feat_var.sqrt().view(N, C, 1)
        feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1)
        feat_mean = feat_mean.permute(0,2,1)
        feat_std = feat_std.permute(0,2,1)
        return feat_mean, feat_std

    def adaptive_instance_normalization(self, content_feat, volume_feat_content, volume_feat_style):
        
        content_feat = jt.unsqueeze(content_feat,0)
        volume_feat_content = jt.unsqueeze(volume_feat_content,0)
        volume_feat_style = jt.unsqueeze(volume_feat_style,0)
        size = content_feat.size()
        volume_mean_content, volume_std_content = self.calc_mean_std(volume_feat_content)
        volume_mean_style, volume_std_style = self.calc_mean_std(volume_feat_style)


        normalized_feat = (content_feat - volume_mean_content.expand(
                size)) / volume_std_content.expand(size)

        normalized_feat = normalized_feat * volume_std_style.expand(size) + volume_mean_style.expand(size) 
        return normalized_feat[0,:,:] 
    def volume_test(self, model, scale=1):

        o_min = (0-0.5)*scale + 0.5
        o_max = (1-0.5)*scale + 0.5
        ten = jt.linspace(o_min, o_max, 128)

        pos_input = jt.stack(jt.meshgrid(ten, ten, ten), -1).reshape(-1, 3)
        pos_input_content = model.pos_encoder_content(pos_input)
        pos_input_style = model.pos_encoder_style(pos_input)

        density_content = model.density_mlp_content(pos_input_content)
        density_style = model.density_mlp_style(pos_input_style)
        density_content = density_content[density_content[..., 0]>0.01]
        density_style = density_style[density_style[..., 0]>0.01]
        #rgb_feature = jt.concat([density, dir_input], -1)
        return density_content, density_style
    def model_style(self, pos_input, dir_input, content_flag=1):
        dir_input = self.model.dir_encoder(dir_input)
        if content_flag:
            pos_input = self.model.pos_encoder_content(pos_input)
            density = self.model.density_mlp_content(pos_input)
            normalized_feat = self.adaptive_instance_normalization(density, self.volume_feat_content, self.volume_feat_style)
        else:
            pos_input = self.model.pos_encoder_style(pos_input)
            density = self.model.density_mlp_style(pos_input)
            normalized_feat = self.adaptive_instance_normalization(density, self.volume_feat_style, self.volume_feat_content)           
        rgb_feature = jt.concat([normalized_feat.float16(), dir_input], -1)
        rgb = self.model.rgb_mlp(rgb_feature)
        outputs = jt.concat([rgb, density[..., :1]], -1)
        return outputs

    def render_img_style(self, dataset_mode="train", img_id=None, dataset_temp=None, content_flag=1):
        if content_flag:
            W, H = self.image_resolutions_content
        else:
            W, H = self.image_resolutions_style
        H = int(H)
        W = int(W)
        if img_id is None:
            img_id = np.random.randint(0, dataset_temp[dataset_mode].n_images, [1])[0]
            img_ids = jt.zeros([H*W], 'int32')+img_id
        else:
            img_ids = jt.zeros([H*W], 'int32')+img_id
        rays_o_total, rays_d_total, rays_pix_total = dataset_temp[dataset_mode].generate_rays_total_test(
            img_ids, W, H)
        rays_pix_total = rays_pix_total.unsqueeze(-1)
        pixel = 0
        imgs = np.empty([H*W+self.n_rays_per_batch, 3])
        alphas = np.empty([H*W+self.n_rays_per_batch, 1])
        for pixel in range(0, W*H, self.n_rays_per_batch):
            end = pixel+self.n_rays_per_batch
            rays_o = rays_o_total[pixel:end]
            rays_d = rays_d_total[pixel:end]
            if end > H*W:
                rays_o = jt.concat(
                    [rays_o, jt.ones([end-H*W]+rays_o.shape[1:], rays_o.dtype)], dim=0)
                rays_d = jt.concat(
                    [rays_d, jt.ones([end-H*W]+rays_d.shape[1:], rays_d.dtype)], dim=0)
            if content_flag:
                pos, dir = self.sampler_content.sample(img_ids, rays_o, rays_d)

                #network_outputs = self.model(pos, dir, content_flag)
                network_outputs = self.model_style(pos, dir, content_flag)
                if network_outputs.shape[0] > 0:
                    temp = 1
                rgb,alpha = self.sampler_content.rays2rgb(network_outputs, inference=True)
            else:
                pos, dir = self.sampler_style.sample(img_ids, rays_o, rays_d)
                network_outputs = self.model(pos, dir, content_flag)
                network_outputs = self.model_style(pos, dir, content_flag)
                rgb,alpha = self.sampler_style.rays2rgb(network_outputs, inference=True)

            imgs[pixel:end] = rgb.numpy()
            alphas[pixel:end] = alpha.numpy()
        imgs = imgs[:H*W].reshape(H, W, 3)
        alphas = alphas[:H*W].reshape(H, W, 1)
        imgs_tar=jt.array(dataset_temp[dataset_mode].image_data[img_id]).reshape(H, W, 4)
        imgs_tar = imgs_tar[..., :3] * imgs_tar[..., 3:] + jt.array(self.background_color) * (1 - imgs_tar[..., 3:])
        imgs_tar = imgs_tar.detach().numpy()
        if not self.alpha_image:
            imgs = imgs + np.array(self.background_color)*(1-alphas)
            alphas = None
        jt.gc()
        return imgs, alphas, imgs_tar
    
    def test_style_two(self, load_ckpt=False):
        if load_ckpt:
            assert os.path.exists(self.ckpt_path), "ckpt file does not exist: "+self.ckpt_path
            self.load_ckpt(self.ckpt_path)
        if self.dataset_content["test"] is None:
            self.dataset_content["test"] = build_from_cfg(self.cfg.dataset_content.test, DATASETS)
        if self.dataset_style["test"] is None:
            self.dataset_style["test"] = build_from_cfg(self.cfg.dataset_style.test, DATASETS)      
        if not os.path.exists(os.path.join(self.save_path, "test_style")):
            os.makedirs(os.path.join(self.save_path, "test_style"))               
        self.test_style(style_path =os.path.join(self.save_path, "test_style"), content_flag=1)
        self.test_style(style_path =os.path.join(self.save_path, "test_style"), content_flag=0)

    def test_style(self, style_path, content_flag=1):
        self.volume_feat_content, self.volume_feat_style = self.volume_test(self.model)
        if content_flag:
            dataset_temp = self.dataset_content
        else:
            dataset_temp = self.dataset_style
        nums = 0
        #for img_i in tqdm(range(0,dataset_temp["train"].n_images,1)):
        for img_i in tqdm(range(0,dataset_temp["test"].n_images,1)):
            #if nums > 5:
            #    break
            nums = nums + 1
            with jt.no_grad():
                imgs=[]
                alphas=[]
                for i in range(1):
                    #simg, alpha, img_tar = self.render_img_style(dataset_mode="train", img_id=img_i, dataset_temp=dataset_temp,content_flag=content_flag)
                    simg, alpha, img_tar = self.render_img_style(dataset_mode="test", img_id=img_i, dataset_temp=dataset_temp,content_flag=content_flag)
                    imgs.append(simg)
                    alphas.append(alpha)
                img = np.stack(imgs, axis=0).mean(0)
                if self.alpha_image:
                    alpha = np.stack(alphas, axis=0).mean(0)
                if content_flag:
                    self.save_img(style_path+f"/{self.exp_name}_2_{self.sty_name}_r_{img_i}.png", img, alpha)
                else:
                    self.save_img(style_path+f"/{self.sty_name}_2_{self.exp_name}_r_{img_i}.png", img, alpha)                    



    def render(self, load_ckpt=True, save_path=None):
        if load_ckpt:
            assert os.path.exists(self.ckpt_path), "ckpt file does not exist: "+self.ckpt_path
            self.load_ckpt(self.ckpt_path)
        if save_path is None or save_path=="":
            save_path = os.path.join(self.save_path, "demo.mp4")
        else:
            assert save_path.endswith(".mp4"), "suffix of save_path need to be .mp4"
        print("rendering video with specified camera path")
        fps = 28
        W, H = self.image_resolutions
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        videowriter = cv2.VideoWriter(save_path, fourcc, fps, (W, H))
        cam_path = camera_path.path_spherical()
        with jt.no_grad():
            for pose in tqdm(cam_path):
                img = self.render_img_with_pose(pose)
                img = (img*255+0.5).clip(0, 255).astype('uint8')
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                videowriter.write(img)
            videowriter.release()
        
    def save_ckpt(self, path):
        jt.save({
            'global_step': self.cfg.m_training_step,
            'model': self.model.state_dict(),
            'sampler_content': self.sampler_content.state_dict(),
            'sampler_style': self.sampler_style.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'nested_optimizer': self.optimizer._nested_optimizer.state_dict(),
            'ema_optimizer': self.ema_optimizer.state_dict(),
        }, path)

    def load_ckpt(self, path):
        print("Loading ckpt from:",path)
        ckpt = jt.load(path)
        self.start = ckpt['global_step']
        self.model.load_state_dict(ckpt['model'])
        if self.using_fp16:
            self.model.set_fp16()
        self.sampler_content.load_state_dict(ckpt['sampler_content'])
        self.sampler_style.load_state_dict(ckpt['sampler_style'])
        self.optimizer.load_state_dict(ckpt['optimizer'])
        nested=ckpt['nested_optimizer']['defaults']['param_groups'][0]
        for pg in self.optimizer._nested_optimizer.param_groups:
            for i in range(len(pg["params"])):
                pg["values"][i]=jt.array(nested["values"][i])
                pg["m"][i]=jt.array(nested["m"][i])
        ema=ckpt['ema_optimizer']['defaults']['param_groups'][0]
        for pg in self.ema_optimizer.param_groups:
            for i in range(len(pg["params"])):
                pg["values"][i]=jt.array(ema["values"][i])
        self.ema_optimizer.steps=ckpt['ema_optimizer']['defaults']['steps']
        
    def val_img(self, iter, content_flag):
        if content_flag:
            dataset_temp = self.dataset_content
        else:
            dataset_temp = self.dataset_style
        with jt.no_grad():
            img, _, img_tar= self.render_img(dataset_mode="val", dataset_temp=dataset_temp,content_flag=content_flag)
            if content_flag:
                self.save_img(self.save_path+f"/img{iter}_c.png", img)
                self.save_img(self.save_path+f"/target{iter}_c.png", img_tar)
            else:
                self.save_img(self.save_path+f"/img{iter}_s.png", img)
                self.save_img(self.save_path+f"/target{iter}_s.png", img_tar)           
            return img2mse(
                jt.array(img), 
                jt.array(img_tar)).item()
    
    def render_test(self, save_img=True, save_path=None, content_flag=1):
        if save_path is None:
            save_path = self.save_path
        mse_list = []
        print("rendering testset...")
        nums = 0
        if content_flag:
            dataset_temp = self.dataset_content
        else:
            dataset_temp = self.dataset_style
        for img_i in tqdm(range(0,dataset_temp["test"].n_images,1)):
            #if nums > 5:
            #    break
            #nums = nums + 1
            with jt.no_grad():
                imgs=[]
                alphas=[]
                for i in range(1):
                    simg, alpha,img_tar = self.render_img(dataset_mode="test", img_id=img_i, dataset_temp=dataset_temp,content_flag=content_flag)
                    imgs.append(simg)
                    alphas.append(alpha)
                img = np.stack(imgs, axis=0).mean(0)
                if self.alpha_image:
                    alpha = np.stack(alphas, axis=0).mean(0)
                if save_img:
                    if content_flag:
                        self.save_img(save_path+f"/{self.exp_name}_r_{img_i}.png", img, alpha)
                        if dataset_temp["test"].have_img:
                            self.save_img(save_path+f"/{self.exp_name}_gt_{img_i}.png", img_tar)
                    else:
                        self.save_img(save_path+f"/{self.sty_name}_r_{img_i}.png", img, alpha)
                        if dataset_temp["test"].have_img:
                            self.save_img(save_path+f"/{self.sty_name}_gt_{img_i}.png", img_tar)                        
                mse_list.append(img2mse(
                jt.array(img), 
                jt.array(img_tar)).item())
        return mse_list

    def save_img(self, path, img, alpha=None):
        if alpha is not None:
            img = np.concatenate([img, alpha], axis=-1)
        if isinstance(img, np.ndarray):
            ndarr = (img*255+0.5).clip(0, 255).astype('uint8')
        elif isinstance(img, jt.Var):
            ndarr = (img*255+0.5).clamp(0, 255).uint8().numpy()
        im = Image.fromarray(ndarr)
        im.save(path)

    def render_img(self, dataset_mode="train", img_id=None, dataset_temp=None, content_flag=1):
        if content_flag:
            W, H = self.image_resolutions_content
        else:
            W, H = self.image_resolutions_style
        H = int(H)
        W = int(W)
        if img_id is None:
            img_id = np.random.randint(0, dataset_temp[dataset_mode].n_images, [1])[0]
            img_ids = jt.zeros([H*W], 'int32')+img_id
        else:
            img_ids = jt.zeros([H*W], 'int32')+img_id
        rays_o_total, rays_d_total, rays_pix_total = dataset_temp[dataset_mode].generate_rays_total_test(
            img_ids, W, H)
        rays_pix_total = rays_pix_total.unsqueeze(-1)
        pixel = 0
        imgs = np.empty([H*W+self.n_rays_per_batch, 3])
        alphas = np.empty([H*W+self.n_rays_per_batch, 1])
        for pixel in range(0, W*H, self.n_rays_per_batch):
            end = pixel+self.n_rays_per_batch
            rays_o = rays_o_total[pixel:end]
            rays_d = rays_d_total[pixel:end]
            if end > H*W:
                rays_o = jt.concat(
                    [rays_o, jt.ones([end-H*W]+rays_o.shape[1:], rays_o.dtype)], dim=0)
                rays_d = jt.concat(
                    [rays_d, jt.ones([end-H*W]+rays_d.shape[1:], rays_d.dtype)], dim=0)
            if content_flag:
                pos, dir = self.sampler_content.sample(img_ids, rays_o, rays_d)
                network_outputs = self.model(pos, dir, content_flag)
                rgb,alpha = self.sampler_content.rays2rgb(network_outputs, inference=True)
            else:
                pos, dir = self.sampler_style.sample(img_ids, rays_o, rays_d)
                network_outputs = self.model(pos, dir, content_flag)
                rgb,alpha = self.sampler_style.rays2rgb(network_outputs, inference=True)

            imgs[pixel:end] = rgb.numpy()
            alphas[pixel:end] = alpha.numpy()
        imgs = imgs[:H*W].reshape(H, W, 3)
        alphas = alphas[:H*W].reshape(H, W, 1)
        imgs_tar=jt.array(dataset_temp[dataset_mode].image_data[img_id]).reshape(H, W, 4)
        imgs_tar = imgs_tar[..., :3] * imgs_tar[..., 3:] + jt.array(self.background_color) * (1 - imgs_tar[..., 3:])
        imgs_tar = imgs_tar.detach().numpy()
        if not self.alpha_image:
            imgs = imgs + np.array(self.background_color)*(1-alphas)
            alphas = None
        jt.gc()
        return imgs, alphas, imgs_tar

    def render_img_with_pose(self, pose):
        W, H = self.image_resolutions
        H = int(H)
        W = int(W)
        fake_img_ids = jt.zeros([H*W], 'int32')
        rays_o_total, rays_d_total = self.dataset["train"].generate_rays_with_pose(pose, W, H)
        img = np.empty([H*W+self.n_rays_per_batch, 3])
        alpha = np.empty([H*W+self.n_rays_per_batch, 1])
        for pixel in range(0, W*H, self.n_rays_per_batch):
            end = pixel+self.n_rays_per_batch
            rays_o = rays_o_total[pixel:end]
            rays_d = rays_d_total[pixel:end]
            if end > H*W:
                rays_o = jt.concat(
                    [rays_o, jt.ones([end-H*W]+rays_o.shape[1:], rays_o.dtype)], dim=0)
                rays_d = jt.concat(
                    [rays_d, jt.ones([end-H*W]+rays_d.shape[1:], rays_d.dtype)], dim=0)
            pos, dir = self.sampler.sample(fake_img_ids, rays_o, rays_d)
            network_outputs = self.model(pos, dir)
            rgb,a = self.sampler.rays2rgb(network_outputs, inference=True)
            img[pixel:end] = rgb.numpy()
            alpha[pixel:end] = a.numpy()
        img = img[:H*W].reshape(H, W, 3)
        alpha = alpha[:H*W].reshape(H, W, 1)
        if not self.alpha_image:
            img = img + np.array(self.background_color)*(1 - alpha)
        return img
