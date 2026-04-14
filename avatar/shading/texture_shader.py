import torch
from torch import nn, Tensor
from torchvision.transforms.functional import resize

from avatar.shading import DeferredPBRShader
from avatar.texture_mip import sample_mip_texture

""" DeferredPBRShader with textures for material attributes.  """
class TexturedDeferredPBRShader(DeferredPBRShader):
    def __init__(self, args, device: torch.device, num_seq: int):
        super().__init__(args, device, num_seq)

        self.texture_mat_res = max(args.texture_res_albedo, args.texture_res_r_spec)
        self.texture_nrm_res = args.texture_res_normal

        # Initialize textures
        self._material_alb = self._init_albedo.unsqueeze(0).unsqueeze(0).repeat(args.texture_res_albedo, args.texture_res_albedo, 1)
        self._material_alb = nn.Parameter(self._material_alb)

        self._material_r_spec = torch.stack([self._init_roughness, self._init_spec], dim=-1).unsqueeze(0).unsqueeze(0).repeat(args.texture_res_r_spec, args.texture_res_r_spec, 1)
        self._material_r_spec = nn.Parameter(self._material_r_spec)

        self.texture_normals = torch.zeros((self.texture_nrm_res, self.texture_nrm_res, 2), dtype=torch.float, device=device) 
        self.texture_normals = torch.nn.Parameter(self.texture_normals)

        self.override_texture_albedo = None
        self.override_texture_r = None
        self.override_texture_spec = None
        self.override_texture_normals = None
        self.override_texture_normals_intensity = 1.0
        self.override_texture_mask = None
        self.enable_normal_map = True
        self.resize_tex_res = None

        def reset_textures():
            with torch.no_grad():
                self._material_alb.fill_(self._init_albedo[0])
                self._material_r_spec[...,0].fill_(self._init_roughness)
                self._material_r_spec[...,1].fill_(self._init_spec)
                self.texture_normals.fill_(0)
        self.reset_textures = reset_textures

        self.texture_mip = args.texture_mip
        self.texture_mip_normals = args.texture_mip_normals
        self.texture_mip_levels = args.texture_mip_levels
        self.texture_mip_max_iter = args.texture_mip_max_iter

    def get_textures(self, seq_idx: Tensor, train_iter: int) -> Tensor:
        mat_texture = self.material(seq_idx, train_iter) # (t,t,3+2+3)
        nrm_texture = self._get_texture_normals_activated(train_iter)
        return mat_texture, nrm_texture

    def material(self, seq_idx: Tensor, train_iter: int) -> Tensor:
        tex_albedo = self._material_alb
        tex_r_spec = self._material_r_spec

        # Resize textures if needed (if albedo, roughness and specular are learned at different resolutions)
        if tex_albedo.shape[0] != self.texture_mat_res or tex_albedo.shape[1] != self.texture_mat_res:
            tex_albedo = resize(tex_albedo.permute(2,0,1), [self.texture_mat_res, self.texture_mat_res]).permute(1,2,0)
        if tex_r_spec.shape[0] != self.texture_mat_res or tex_r_spec.shape[1] != self.texture_mat_res:
            tex_r_spec = resize(tex_r_spec.permute(2,0,1), [self.texture_mat_res, self.texture_mat_res]).permute(1,2,0)
        
        material = torch.cat((tex_albedo, tex_r_spec), dim=-1) # (t,t,3+2)

        # Optionally concatenate the PCA albedo
        if self.parametric_albedo is not None:
            param_albedo_tex = self.parametric_albedo() # (r,r,3)
            param_albedo_tex = resize(param_albedo_tex.permute(2,0,1), [self.texture_mat_res, self.texture_mat_res]).permute(1,2,0) # (t,t,3)
            material = torch.cat((material, param_albedo_tex), dim=-1) # (t,t,3+2+3)
        
        tex = self._activate_material(material)

        # Experimental progressive mipmapping during training
        if self.texture_mip:
            tex = self._texture_mipmap(tex, train_iter)

        # Overrides (not used during training)
        if self.override_texture_albedo is not None or self.override_texture_r is not None or self.override_texture_spec is not None:
            new_tex_a, new_tex_r, new_tex_spec = self.override_texture_albedo, self.override_texture_r, self.override_texture_spec
            max_res = max(new_tex_a.shape[0] if new_tex_a is not None else 0, new_tex_r.shape[0] if new_tex_r is not None else 0, new_tex_spec.shape[0] if new_tex_spec is not None else 0)
            # Resize everything to the max res
            res = lambda x: resize((x if x.ndim == 3 else x.unsqueeze(-1)).permute(2,0,1), [max_res, max_res]).permute(1,2,0)
            mask = torch.ones((max_res, max_res, 1), dtype=torch.float, device=tex.device) if self.override_texture_mask is None else res(self.override_texture_mask)[..., 0:1]
            tex = res(tex)
            if new_tex_a is not None:
                new_tex_a = res(new_tex_a)
                new_tex_a, alpha = new_tex_a[..., 0:3], new_tex_a[..., 3:4] if new_tex_a.shape[-1] == 4 else torch.ones_like(new_tex_a[..., 0:1])
                alpha = alpha * mask
                tex[:,:,0:3] = new_tex_a * alpha + tex[:,:,0:3] * (1-alpha)
            if new_tex_r is not None:
                new_tex_r = res(new_tex_r)
                new_tex_r, alpha = new_tex_r[..., 0:1], new_tex_r[..., 3:4] if new_tex_r.shape[-1] == 4 else torch.ones_like(new_tex_r[..., 0:1])
                alpha = alpha * mask
                tex[:,:,3:4] = new_tex_r * alpha + tex[:,:,3:4] * (1-alpha)
            if new_tex_spec is not None:
                new_tex_spec = res(new_tex_spec)
                new_tex_spec, alpha = new_tex_spec[..., 0:1], new_tex_spec[..., 3:4] if new_tex_spec.shape[-1] == 4 else torch.ones_like(new_tex_spec[..., 0:1])
                alpha = alpha * mask
                tex[:,:,4:5] = new_tex_spec * alpha + tex[:,:,4:5] * (1-alpha)

        if self.resize_tex_res is not None:
            tex = resize(tex.permute(2,0,1), [self.resize_tex_res, self.resize_tex_res]).permute(1,2,0)

        return tex

    def _get_texture_normals_activated(self, train_iter: int) -> Tensor:
        if not self.enable_normal_map:
            return torch.tensor([0,0,1], dtype=torch.float, device=self.device).view(1,1,3).repeat(self.texture_nrm_res, self.texture_nrm_res, 1)

        x = self.texture_normals
        # x: (Tn, Tn, 2)
        Tn = x.shape[0]
        assert x.shape == (Tn, Tn, 2), f"Expected normal texture to be of shape ({Tn}, {Tn}, 2), got {x.shape}"

        # Reshape to (size*size, 2)
        x = x.reshape(-1, 2)  
        # Only use x,y components in [-1,1]
        xy = torch.tanh(x)
        # Constrain magnitude of xy to ≤1
        xy_squared_norm = torch.sum(xy**2, dim=1, keepdim=True).clamp(max=0.99)    
        # Derive z component (always positive in tangent space)
        # The magnitude of xy defines the inclination of the normal vector
        z = torch.sqrt(1 - xy_squared_norm)
        # Combine components
        normals = torch.cat([xy, z], dim=-1)
        normals = normals.reshape(Tn, Tn, 3)

        if self.texture_mip_normals:
            normals = self._texture_mipmap(normals, train_iter)

        if self.override_texture_normals is not None:
            new_tex = self.override_texture_normals
            res = lambda x: resize(x.permute(2,0,1), [new_tex.shape[0], new_tex.shape[1]]).permute(1,2,0)
            new_tex, alpha = new_tex[..., 0:3], new_tex[..., 3:4] if new_tex.shape[-1] == 4 else torch.ones_like(new_tex[..., 0:1])
            mask = torch.ones((new_tex.shape[0], new_tex.shape[1], 1), dtype=torch.float, device=normals.device) if self.override_texture_mask is None else res(self.override_texture_mask)[..., 0:1]
            alpha = alpha * mask * self.override_texture_normals_intensity
            new_tex = new_tex * 2 - 1  # Map from [0,1] to [-1,1]
            # Resize the original texture to match the override
            resized_tex = res(normals)
            # Blend both and normalize
            # normals = torch.nn.functional.normalize(new_tex * alpha + resized_tex * (1-alpha), dim=-1)
            normals = torch.nn.functional.normalize(new_tex * alpha + resized_tex, dim=-1)

        if self.resize_tex_res is not None:
            normals = resize(normals.permute(2,0,1), [self.resize_tex_res, self.resize_tex_res]).permute(1,2,0)

        return normals

    def _texture_mipmap(self, tex: Tensor, train_iter: int) -> Tensor:
        max_mip_level = self.texture_mip_levels
        # Compute the texture mip level based on the training iteration
        mip_step = max_mip_level / self.texture_mip_max_iter
        mip_level = max_mip_level - mip_step * train_iter
        mip_level = max(0, min(max_mip_level, mip_level))
        return sample_mip_texture(tex, max_mip_level, mip_level)
