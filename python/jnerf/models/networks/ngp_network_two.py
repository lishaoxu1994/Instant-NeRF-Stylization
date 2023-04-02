from turtle import pos, position
import jittor as jt
from jittor import nn, init
import os
from jnerf.utils.config import get_cfg
from jnerf.utils.registry import build_from_cfg, NETWORKS, ENCODERS
from jnerf.ops.code_ops.fully_fused_mlp import FullyFusedMlp_weight

class FMLP(nn.Module):
    def __init__(self, weight_shapes, weights=None):
        super(FMLP, self).__init__()
        if weights == None:                   
            assert len(weight_shapes) > 2
            self.output_shape1 = weight_shapes[-1]
            dweights = []
            for i in range(len(weight_shapes) - 1):
                dweights.append(init.invariant_uniform((weight_shapes[i], weight_shapes[i+1]), "float16").float16())
        else:
            assert len(weights) >= 2
            self.output_shape1 = weights[-1].shape[-1]
            dweights = weights
        self.func = FullyFusedMlp_weight(dweights)
        con_weights = []
        for i in range(len(dweights)):
            if i == len(dweights) - 1:
                if dweights[i].shape[1] < 16: 
                    dweights[i] = jt.concat([dweights[i], jt.zeros((dweights[i].shape[0], 16 - dweights[i].shape[1]))], -1).float16()
            con_weights.append(dweights[i].transpose(1,0).reshape(-1))
        jt_con_weights = jt.concat(con_weights, -1)
        self.con_weights = jt_con_weights

    def execute(self, x):
        if x.shape[0] == 0:
            return jt.empty([0, self.output_shape1]).float16()
        ret = self.func(x, self.con_weights)
        if self.output_shape1 != ret.shape[1]:
            ret = ret[:,:self.output_shape1]
        return ret

@NETWORKS.register_module()
class NGPNetworks_two(nn.Module):
    def __init__(self, use_fully=True, density_hidden_layer=1, density_n_neurons=64, rgb_hidden_layer=2, rgb_n_neurons=64):
        super(NGPNetworks_two, self).__init__()
        self.use_fully = use_fully
        self.cfg = get_cfg()
        self.using_fp16 = self.cfg.fp16
        self.pos_encoder_content = build_from_cfg(self.cfg.encoder.pos_encoder, ENCODERS, cfg_aabb_scale=self.cfg.dataset_content_obj.aabb_scale)
        self.pos_encoder_style = build_from_cfg(self.cfg.encoder.pos_encoder, ENCODERS, cfg_aabb_scale=self.cfg.dataset_style_obj.aabb_scale)
        self.dir_encoder = build_from_cfg(self.cfg.encoder.dir_encoder, ENCODERS)

        if self.use_fully and jt.flags.cuda_archs[0] >= 75 and self.using_fp16:
            assert self.pos_encoder_content.out_dim%16==0
            assert self.pos_encoder_style.out_dim%16==0
            assert self.dir_encoder.out_dim%16==0
            self.density_mlp_content = FMLP([self.pos_encoder_content.out_dim, density_n_neurons, 16])
            self.density_mlp_style = FMLP([self.pos_encoder_style.out_dim, density_n_neurons, 16])
            self.rgb_mlp = FMLP([self.dir_encoder.out_dim+16, rgb_n_neurons, rgb_n_neurons, 3])
        else:
            if self.use_fully and not (jt.flags.cuda_archs[0] >= 75):
                print("Warning: Sm arch is lower than sm_75, FFMLPs is not supported. Automatically use original MLPs instead.")
            elif self.use_fully and not self.using_fp16:
                print("Warning: FFMLPs only support float16. Automatically use original MLPs instead.")
            self.density_mlp = nn.Sequential(
                nn.Linear(self.pos_encoder.out_dim, density_n_neurons, bias=False), 
                nn.ReLU(), 
                nn.Linear(density_n_neurons, 16, bias=False))
            self.rgb_mlp = nn.Sequential(nn.Linear(self.dir_encoder.out_dim+16, rgb_n_neurons, bias=False),
                            nn.ReLU(),
                            nn.Linear(rgb_n_neurons, rgb_n_neurons, bias=False),
                            nn.ReLU(),
                            nn.Linear(rgb_n_neurons, 3, bias=False))
        self.set_fp16()

    def style_execute(self, pos_input, dir_input, model_s, gan):  
        if self.using_fp16:
            with jt.flag_scope(auto_mixed_precision_level=5):
                return self.style_execute_(pos_input, dir_input, model_s, gan)
        else:
            return self.style_execute_(pos_input, dir_input, model_s, gan)
    
    def style_execute_(self, pos_input, dir_input, model_s, gan):  
        dir_input = self.dir_encoder(dir_input)
        pos_input_t = self.pos_encoder(pos_input)
        density = self.density_mlp(pos_input_t)

        if density.shape[0] > 0 :
            tttt = 1
        density = density.float32()
        density = gan(density)
        #rgb = rgb.float16()
        rgb = jt.concat([density, dir_input], -1)
        rgb = model_s.rgb_mlp(rgb)

        #rgb = self.rgb_mlp(rgb)

        outputs = jt.concat([rgb, density[..., :1]], -1)  # batchsize 4: rgbd
        return outputs

    def execute(self, pos_input, dir_input, content_flag):  
        if self.using_fp16:
            with jt.flag_scope(auto_mixed_precision_level=5):
                return self.execute_(pos_input, dir_input, content_flag)
        else:
            return self.execute_(pos_input, dir_input, content_flag)

    def execute_(self, pos_input, dir_input, content_flag):  
        dir_input = self.dir_encoder(dir_input)
        if content_flag:
            pos_input_content = self.pos_encoder_content(pos_input)
            density_content = self.density_mlp_content(pos_input_content)
            rgb_content = jt.concat([density_content, dir_input], -1)     
            rgb_content = self.rgb_mlp(rgb_content)
            outputs= jt.concat([rgb_content, density_content[..., :1]], -1)  # batchsize 4: rgbd            
        else:
            pos_input_style = self.pos_encoder_style(pos_input)
            density_style = self.density_mlp_style(pos_input_style)
            rgb_style = jt.concat([density_style, dir_input], -1)     
            rgb_style = self.rgb_mlp(rgb_style)
            outputs= jt.concat([rgb_style, density_style[..., :1]], -1)  # batchsize 4: rgbd        
        return outputs

    def density(self, pos_input, content_flag):  # batchsize,3
        if content_flag:
            density = self.pos_encoder_content(pos_input)
            density = self.density_mlp_content(density)[:,:1]
        else:
            density = self.pos_encoder_style(pos_input)
            density = self.density_mlp_style(density)[:,:1]            
        return density

    def set_fp16(self):
        if self.using_fp16:
            self.density_mlp_content.float16()
            self.density_mlp_style.float16()
            self.rgb_mlp.float16()
            self.pos_encoder_content.float16()
            self.pos_encoder_style.float16()
            self.dir_encoder.float16()