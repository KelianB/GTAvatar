import os
import sys
import logging
from typing import List, Callable
from time import time
from pathlib import Path
from datetime import datetime
from math import pi

import torch
from torch import Tensor
from torchvision.transforms.functional import resize, InterpolationMode
from PyQt5.QtWidgets import QMainWindow, QApplication, QWidget, QSlider, QLabel, QVBoxLayout, QHBoxLayout, QDesktopWidget, QScrollArea, QComboBox, QLayout, QGridLayout, QTabWidget, QCheckBox, QPushButton, QLineEdit, QShortcut, QFileDialog, QFrame
from PyQt5.QtGui import QPixmap, QImage, QFont
from PyQt5.QtCore import Qt, pyqtSignal, QTimer

from dataset import dataset_util
from avatar import Avatar, RenderSettings, create_parser, parse_args
from avatar.environment_light import load_envmap
from avatar.rendering.gaussian_renderer_2dgs_textured import discard_hw_textures
from utils.logging import setup_logging
from utils.math import gaussian_kernel, apply_featurewise_conv1d
from utils.visualization import save_img, convert_uint, srgb_to_linear
from utils.geometry import AABB
from utils.tqdm import tqdm


CAM_ORBIT_SPEED = 0.01
CAM_Z_SPEED = 0.005
CTRL_Z_MULTIPLIER = 0.1 # scales CAM_Z_SPEED when ctrl is pressed
CTRL_ORBIT_MULTIPLIER = 0.1 # scales CAM_ORBIT_SPEED when ctrl is pressed

SCREENSHOTS_DIR = Path.home() / "Pictures" / "Screenshots"
ENVMAPS_DIR = Path("assets/envmaps")

# If non-zero, poll the filesystem for changes to the loaded texture.
# This allows using external tools to edit the textures and see the results immediately.
LIVE_REFRESH_TEXTURE_INTERVAL = 0.5 # seconds (float)

# Hardware acceleration of textures (this should be enabled by default, but can be disabled for debugging or if it causes issues on some systems)
HARDWARE_TEXTURES = True

RENDER_MODES = [
    {
        "name": "RGB",
        "fn": lambda view, output, vis: vis["render"],
        "vis": ["render"],
    },
    {
        "name": "Normals (splatted)",
        "fn": lambda view, output, vis: vis["normals"],
        "vis": ["normals"],
    },
    {
        "name": "Normals (from depth)",
        "fn": lambda view, output, vis: vis["normals_depth"],
        "vis": ["normals_depth"],
    },
    {
        "name": "Normals (mesh, smooth)",
        "fn": lambda view, output, vis: vis["mesh_normals"],
        "vis": ["mesh_normals"],
    },
    {
        "name": "Normals (mesh, flat)",
        "fn": lambda view, output, vis: vis["mesh_normals_flat"],
        "vis": ["mesh_normals_flat"],
    },
    # {
    #     "name": "Normals (GT)",
    #     "fn": lambda view, output, vis: view["normals"],
    # },
    {
        "name": "Depth",
        "fn": lambda view, output, vis: vis["depth"],
        "vis": ["depth"],
    },
    {
        "name": "Sphere",
        "fn": lambda view, output, vis: vis["shading_sphere"],
        "vis": ["shading_sphere"],
    },
    {
        "name": "Albedo",
        "fn": lambda view, output, vis: vis["material"][..., :3],
        "vis": ["material"],
    },
    {
        "name": "Roughness",
        "fn": lambda view, output, vis: vis["material"][..., (3,)].repeat(1,1,1,3),
        "vis": ["material"],
    },
    {
        "name": "Specular reflectance",
        "fn": lambda view, output, vis: vis["material"][..., (4,)].repeat(1,1,1,3),
        "vis": ["material"],
    },
    {
        "name": "Triangle overlay",
        "fn": lambda view, output, vis: torch.lerp(vis["mesh_normals_flat"], output.rast_buffers.material[..., :3], output.rast_buffers.rend_alpha * 0.8),
        # "fn": lambda view, output, vis: torch.lerp(vis["mesh_normals_flat"], output.render, output.rast_buffers.rend_alpha * 0.8),
        "vis": ["mesh_normals_flat"],
    },
    {
        "name": "GT",
        "fn": lambda view, output, vis: view["img"],
        "vis": [],
    },
]

# Max source frames to load when generating videos.
# The video will be limited to using this many frames from the source dataset to avoid using too much memory.
VIDEO_MAX_SRC_FRAMES = 1500

display02f = lambda v: f"{v:.02f}"


def main(args, avatar):
    avatar.resume()
    iter_str = "" if args.detached else f" epoch {args.resume_epoch}" if args.resume_epoch else f" iter {args.resume_iter:04d}"

    device, dataset_train, dataset_test = avatar.device, avatar.dataset_train, avatar.dataset_test
    shader = avatar.shader

    default_render_mode = 0
    default_pose = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=torch.float, device=device)
    default_expr = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=torch.float, device=device)
    default_expr = torch.cat((default_expr, torch.zeros((40), dtype=torch.float, device=device)))

    pose_labels = ["global.x", "global.y", "global.z", "neck.x", "neck.y", "neck.z", "jaw.x", "jaw.y", "jaw.z", "eyeR.x", "eyeR.y", "eyeR.z", "eyeL.x", "eyeL.y", "eyeL.z", "l_eyelid", "r_eyelid", "trans.x", "trans.y", "trans.z"]
    assert len(default_pose) == len(pose_labels)
    n_pose_params = len(pose_labels)
    n_exp = 50
    expr_labels = [str(i) for i in range(n_exp)]
    assert len(default_expr) == len(expr_labels) == n_exp

    base_view = dataset_train.collate([dataset_train[0]])
    base_view = dataset_util.to_device_recursive(base_view, device)
    cam = base_view["camera"][0].clone()

    envmaps = Path(ENVMAPS_DIR).glob("*.hdr")
    envmaps = sorted(list(envmaps))

    def load_tex(file_path, tex_type):
        try:
            tex = dataset_util.load_img(file_path).to(avatar.device) # (H,W,3|4)
            if tex.ndim == 2:
                tex = tex.unsqueeze(-1)
            if tex.shape[-1] == 3:
                tex = torch.cat((tex, torch.ones_like(tex[..., :1])), dim=-1) # add alpha

            if tex_type == "albedo":
                tex = avatar.inverse_albedo_display_transform(tex)

            tiling = [0.25, 0.5, 1, 2, 4][round(w.custom_material_tiling.value)]
            if tiling is not None:
                if tiling > 1:
                    # tile the texture n times (from square to square)
                    tex = tex.repeat(tiling, tiling, 1)
                elif tiling < 1:
                    tex = tex[:int(tex.shape[0]*tiling), :int(tex.shape[1]*tiling), :]
            return tex
        except FileNotFoundError:
            return None
    def on_custom_texture_change():
        discard_hw_textures()
        w.update_render()
    custom_tex = CustomTextures(LIVE_REFRESH_TEXTURE_INTERVAL, load_tex, on_custom_texture_change)

    logging.info("Setting up UI")

    expr_amplitude = 3
    pose_amplitude = 1.5
 
    def widget(layout, parent):
        wid = QWidget(parent)
        wid.setLayout(layout)
        return wid
    def layout_generator(Class):
        def fn(*children):
            l = Class()
            for child in children:
                if isinstance(child, QWidget): l.addWidget(child)
                if isinstance(child, QLayout): l.addLayout(child)
            return l
        return fn
    hbox = layout_generator(QHBoxLayout)
    vbox = layout_generator(QVBoxLayout)
    line = lambda: QFrame(frameShape=QFrame.HLine, frameShadow=QFrame.Sunken, styleSheet="color: gray;")

    w = None

    class MainWindow(QMainWindow):
        def __init__(self):
            super(MainWindow, self).__init__()
            self.enable_triangle_selection = False
            self.mesh_tri = None
            self.selected_tri_idx = None
            self.last_mouse_pos = None
            self.view = base_view
            self.decimation_seed = 0
            self.preset_mask = None
            self.envmap = None
            self.free_cam = True
            self.background_color = "white"
            self.gaussians_mask = None

            # Center the orbit camera at the center of the mesh in the first view
            base_verts = avatar.deformer.get_mesh_verts(base_view["flame_pose"], base_view["flame_expression"], avatar.shape_param, avatar.gaussians, avatar.flame_scale)
            aabb = AABB(base_verts.view(-1, 3))
            self.control_cam = OrbitCamera(aabb.center, distance=aabb.longest_extent * 3, min_distance=0.001, device=device)

            self.setWindowTitle(f"Visualization ({args.run_name}{iter_str})")

            # Keyboard shortcuts
            def set_shortcut(key, fn):
                shortcut = QShortcut(key, self)
                shortcut.activated.connect(fn)

            def toggle_free_cam():
                self.free_cam = not self.free_cam
                self.update_render()
            set_shortcut("f", toggle_free_cam)

            def take_screenshot():
                img = self.grab_screenshot(no_background=True)
                timestamp = datetime.today().strftime("%Y-%m-%d %H-%M-%S")
                SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)
                save_img(SCREENSHOTS_DIR / f"{timestamp}.png", img)
            set_shortcut("s", take_screenshot)

            mode_rgb_i = next(i for i, m in enumerate(RENDER_MODES) if m["name"] == "RGB")
            mode_gt_i = next(i for i, m in enumerate(RENDER_MODES) if m["name"] == "GT")
            def toggle_render_gt():
                i = self.select_mode_box.currentIndex()
                self.select_mode_box.setCurrentIndex(mode_rgb_i if i == mode_gt_i else mode_gt_i)
                self.update_render()
            set_shortcut("g", toggle_render_gt)               

            target_interval = 1/10
            anim_interval: SetInterval = None
            def togglePlay():
                nonlocal anim_interval
                if anim_interval is None:
                    is_train = self.slider_test_idx.value == -1
                    slider = self.slider_train_idx if is_train else self.slider_test_idx
                    max_v = len(dataset_train) if is_train else len(dataset_test)
                    t = time()
                    def _next_frame():
                        nonlocal t
                        v = (int(slider.value) + 1) % max_v
                        # Set the frame
                        slider.set_value_silently(v)
                        self._set_view(v, is_train=is_train)
                        self.update_render()
                        # Set the timer for the next frame, adjusting for how long this frame took to render
                        elapsed = time() - t
                        new_interval = max(1, (target_interval - elapsed) * 1000)
                        anim_interval.start(int(new_interval))
                        t = time()
                    anim_interval = SetInterval(target_interval, _next_frame)
                else:
                    anim_interval.stop()
                    anim_interval = None
            set_shortcut("p", togglePlay)

            # self.select_seq_box = QComboBox(self)
            # for i in range(len(args.train_dirs)):
            #     self.select_seq_box.addItem(f"Sequence #{i}", i)
            # self.select_seq_box.activated.connect(self.update_render)

            def _populate_envmaps(indices):
                select_light_box.addItem("Training", 0)
                for i in indices:
                    select_light_box.addItem(str(envmaps[i].stem), i+1)
            def on_light_selection_change():
                idx = select_light_box.currentData()
                if idx == 0:
                    self.envmap = None
                else:                    
                    fp = envmaps[idx-1]
                    logging.info(f"Loading environment map at \"{fp}\".")
                    self.envmap = load_envmap(fp, device=device)
                self.update_render()
            def on_envmap_search_change():
                search = envmap_search_bar.text().lower().strip()
                select_light_box.clear()
                _populate_envmaps([i for i, envmap_file in enumerate(envmaps) if search in str(envmap_file.stem).lower()])
            
            select_light_box = QComboBox(self)
            select_light_box.activated.connect(on_light_selection_change)
            _populate_envmaps(range(len(envmaps)))
            envmap_search_bar = QLineEdit(self)
            envmap_search_bar.setPlaceholderText("Search envmaps...")
            envmap_search_bar.textChanged.connect(on_envmap_search_change)

            layout_shading = QGridLayout()
            i = 0
            layout_shading.addWidget(QLabel("Env: ", self), i, 0)
            layout_shading.addWidget(select_light_box, i, 1)
            layout_shading.addWidget(envmap_search_bar, i, 2, 1, -1)
            i += 1
            self.env_rotate = SliderControl(w, "Rotation", min=-pi, max=pi, step=0.01, default_value=0, on_change=self.update_render, display=display02f)
            layout_shading.addWidget(self.env_rotate.label, i, 0)
            layout_shading.addWidget(self.env_rotate.slider, i, 1)
            layout_shading.addWidget(self.env_rotate.text, i, 2)  

            i += 1
            self.diffuse_scale = SliderControl(w, "Diffuse scale", min=0, max=5, step=0.01, default_value=1, on_change=self.update_render, display=display02f)
            layout_shading.addWidget(self.diffuse_scale.label, i, 0)
            layout_shading.addWidget(self.diffuse_scale.slider, i, 1)
            layout_shading.addWidget(self.diffuse_scale.text, i, 2)

            i += 1
            self.specular_scale = SliderControl(w, "Specular scale", min=0, max=5, step=0.01, default_value=1, on_change=self.update_render, display=display02f)
            layout_shading.addWidget(self.specular_scale.label, i, 0)
            layout_shading.addWidget(self.specular_scale.slider, i, 1)
            layout_shading.addWidget(self.specular_scale.text, i, 2)

            i += 1
            self.brightness_scale = SliderControl(w, "Brightness scale", min=0, max=5, step=0.01, default_value=1.0, display=display02f, on_change=self.update_render)
            layout_shading.addWidget(self.brightness_scale.label, i, 0)
            layout_shading.addWidget(self.brightness_scale.slider, i, 1)
            layout_shading.addWidget(self.brightness_scale.text, i, 2)

            i += 1
            self.roughness_scale = SliderControl(w, "Roughness scale", min=0, max=5, step=0.01, default_value=1, display=display02f, on_change=self.update_render)
            layout_shading.addWidget(self.roughness_scale.label, i, 0)
            layout_shading.addWidget(self.roughness_scale.slider, i, 1)
            layout_shading.addWidget(self.roughness_scale.text, i, 2)

            i += 1
            self.control_metallic = SliderControl(w, "Metallic: ", min=0, max=1, step=0.01, default_value=0, on_change=self.update_render, checkbox=True, display=display02f)
            layout_shading.addWidget(self.control_metallic.label, i, 0)
            layout_shading.addWidget(self.control_metallic.slider, i, 1)
            layout_shading.addWidget(self.control_metallic.text, i, 2)
            layout_shading.addWidget(self.control_metallic.checkbox, i, 3)

            i += 1
            self.shading_normals_mode_box = QComboBox(self)
            self.shading_normals_mode_box.addItem("splatted", 0)
            self.shading_normals_mode_box.addItem("from depth", 1)
            self.shading_normals_mode_box.addItem("mesh (smooth)", 2)
            self.shading_normals_mode_box.addItem("mesh (flat)", 3)
            self.shading_normals_mode_box.activated.connect(self.update_render)
            layout_shading.addWidget(QLabel("Shading normals: ", self), i, 0)
            layout_shading.addWidget(self.shading_normals_mode_box, i, 1)

            i += 1
            self.box_envlight_background = QCheckBox()
            self.box_envlight_background.setChecked(True)
            self.box_envlight_background.stateChanged.connect(self.update_render)
            layout_shading.addWidget(QLabel("Env. background: ", self), i, 0)
            layout_shading.addWidget(self.box_envlight_background, i, 1)


            self.select_mode_box = QComboBox(self)
            for k, mode in enumerate(RENDER_MODES):
                self.select_mode_box.addItem(mode["name"], k)
            self.select_mode_box.setCurrentIndex(default_render_mode)
            this = self
            def update_mode():
                img, _, _ = this.update_render()
                size = img.shape[0:2]
                this.set_window(size)
            self.select_mode_box.activated.connect(update_mode)

            def _set_view(item, is_train: bool):
                dataset = dataset_train if is_train else dataset_test

                if item is None:
                    # reset to defaults
                    view = self.view
                    view["flame_pose"][0] = default_pose.clone()
                    view["flame_expression"][0] = default_expr.clone()
                elif type(item) == int or type(item) == float:
                    # Pick the view at the given index
                    if item < 0 or item > len(dataset):
                        raise ValueError(f"View index {item} out of range for dataset of length {len(dataset)}")
                    view = dataset.collate([dataset[int(item)]])
                    view = dataset_util.to_device_recursive(view, device)
                    view["flame_pose"], view["flame_expression"] = avatar.compute_flame_attrs(view, is_train=is_train)
                    self.view = view
                else:
                    self.view = item
                self.set_pose_expr_display(self.view["flame_pose"][0], self.view["flame_expression"][0])
            self._set_view = _set_view

            def _slider_train_change(v: float):
                v = int(v)
                self.slider_test_idx.set_value_silently(-1)
                _set_view(None if v == -1 else v, is_train=True)
                self.update_render()
            self.slider_train_idx = SliderControl(self, "train #", min=-1, max=len(dataset_train)-1, step=1, default_value=-1, on_change=_slider_train_change, display=lambda v: "" if v == -1 else f"{int(v):04d}")
            self.slider_train_idx.text.setMaximumWidth(120)

            def _slider_test_change(v: float):
                v = int(v)
                self.slider_train_idx.set_value_silently(-1)
                _set_view(None if v == -1 else v, is_train=False)
                self.update_render()
            self.slider_test_idx = SliderControl(self, "test #", min=-1, max=len(dataset_test)-1, step=1, default_value=-1, on_change=_slider_test_change, display=lambda v: "" if v == -1 else f"{int(v):04d}")
            self.slider_test_idx.text.setMaximumWidth(120)

            layout_expr = QGridLayout()
            self.expr_controls: List[SliderControl] = []
            def expr_onchange(i):
                def _fn(v: float):
                    self.view["flame_expression"][0, i] = v
                    self.update_render()
                return _fn
            for i in range(n_exp):
                control = SliderControl(self, f"{expr_labels[i]}: ", min=-expr_amplitude, max=expr_amplitude, step=0.001, default_value=0, on_change=expr_onchange(i), display=lambda v: f"{v:.03f}")
                self.expr_controls.append(control)
                layout_expr.addWidget(control.label, i, 0)
                layout_expr.addWidget(control.slider, i, 1)
                layout_expr.addWidget(control.text, i, 2)

            layout_pose = QGridLayout()
            self.pose_controls: List[SliderControl] = []
            def pose_onchange(i):
                def _fn(v: float):
                    self.view["flame_pose"][0, i] = v
                    self.update_render()
                return _fn
            for i in range(n_pose_params):
                control = SliderControl(self, f"{pose_labels[i]}: ", min=-pose_amplitude, max=pose_amplitude, step=0.001, default_value=0, on_change=pose_onchange(i), display=lambda v: f"{v:.03f}")
                self.pose_controls.append(control)
                layout_pose.addWidget(control.label, i, 0)
                layout_pose.addWidget(control.slider, i, 1)
                layout_pose.addWidget(control.text, i, 2)
  
            self.screenshot_btn = QPushButton("Screenshot", self)
            self.screenshot_btn.setMaximumWidth(120)
            self.screenshot_btn.clicked.connect(take_screenshot)

            def set_pose_expr_display(pose: Tensor, expr: Tensor):
                # Set the UI values for pose and expression without triggering re-renders
                for p, control in zip(pose, self.pose_controls):
                    control.set_value_silently(p.item())
                for e, control in zip(expr, self.expr_controls):
                    control.set_value_silently(e.item())
            self.set_pose_expr_display = set_pose_expr_display

            layout_texturing = QGridLayout()

            i = 0

            self.box_custom_material = QCheckBox(self)
            self.box_custom_material.setChecked(True)
            def box_custom_material_changed():
                discard_hw_textures()
                self.update_render()
            self.box_custom_material.stateChanged.connect(box_custom_material_changed)
            btn_custom_material = QPushButton("Load texture", self)
            def on_custom_material():
                dir = QFileDialog.getExistingDirectory(self, "Select custom material directory", "./assets/textures")
                if dir:
                    dir = Path(dir)
                    btn_custom_material.setText(dir.stem)
                    custom_tex.reset()
                    custom_tex.set_dir(dir)
                    custom_tex.poll_for_changes()
            btn_custom_material.clicked.connect(on_custom_material)
            layout_texturing.addWidget(QLabel("Custom material: ", self), i, 0)
            layout_texturing.addWidget(btn_custom_material, i, 1)
            layout_texturing.addWidget(self.box_custom_material, i, 2)

            i += 1
            self.preset_material_mask = QComboBox(self)
            self.preset_material_mask.addItem("None", "none")
            self.preset_material_mask.addItem("Left", "left")
            self.preset_material_mask.addItem("Right", "right")
            def customMaterialMaskChanged():
                option = self.preset_material_mask.currentData()
                if option == "none":
                    self.preset_mask = None
                else:
                    # the mask will be resized automatically later
                    new_mask = torch.zeros([64, 64, 1], dtype=torch.float, device=device)
                    if option == "left":
                        new_mask[:, :32] = 1
                    elif option == "right":
                        new_mask[:, 32:] = 1
                    self.preset_mask = new_mask
                discard_hw_textures()
                self.update_render()

            self.preset_material_mask.currentIndexChanged.connect(customMaterialMaskChanged)
            layout_texturing.addWidget(QLabel("Custom mask: ", self), i, 0)
            layout_texturing.addWidget(self.preset_material_mask, i, 1)

            i += 1
            self.custom_material_tiling = SliderControl(w, "Tiling: ", min=0, max=4, step=1, default_value=2, on_change=lambda v: custom_tex.poll_for_changes(force=True), textbox=False, display=lambda x: ["0.25x", "0.5x", "1x", "2x", "4x"][round(x)])
            layout_texturing.addWidget(self.custom_material_tiling.label, i, 0)
            layout_texturing.addWidget(self.custom_material_tiling.slider, i, 1)
            layout_texturing.addWidget(self.custom_material_tiling.text, i, 2)

            i += 1
            def custom_nrm_intensity_changed(v):
                shader.override_texture_normals_intensity = v
                discard_hw_textures()
                self.update_render()
            custom_nrm_intensity = SliderControl(w, "Normal map intensity: ", min=0, max=4, step=0.01, default_value=1, on_change=custom_nrm_intensity_changed, display=display02f)
            layout_texturing.addWidget(custom_nrm_intensity.label, i, 0)
            layout_texturing.addWidget(custom_nrm_intensity.slider, i, 1)
            layout_texturing.addWidget(custom_nrm_intensity.text, i, 2)

            i += 1
            layout_texturing.addWidget(QLabel("Enable normal map: ", self), i, 0)
            self.box_enable_normal_map = QCheckBox()
            self.box_enable_normal_map.setChecked(True)
            def _onchange_enable_normal_map():
                discard_hw_textures()
                self.update_render()
            self.box_enable_normal_map.stateChanged.connect(_onchange_enable_normal_map)
            layout_texturing.addWidget(self.box_enable_normal_map, i, 1)

            i += 1
            self.text_resize_tex = QLineEdit(self)
            def textResizeTexChanged():
                try:
                    shader.resize_tex_res = int(self.text_resize_tex.text().strip())
                    discard_hw_textures()
                    self.update_render()
                except Exception as e:
                    print(e)
            self.text_resize_tex.textChanged.connect(textResizeTexChanged)
            layout_texturing.addWidget(QLabel("Resize texture: ", self), i, 0)
            layout_texturing.addWidget(self.text_resize_tex, i, 1)

            layout_other = QGridLayout()

            # def decr_decimation_seed():
            #     self.decimation_seed -= 1
            #     self.update_render()
            # def incr_decimation_seed():
            #     self.decimation_seed += 1
            #     self.update_render()
            # self.btn_decimation_seed_minus = QPushButton("-", self)
            # self.btn_decimation_seed_minus.clicked.connect(decr_decimation_seed)
            # self.btn_decimation_seed_plus = QPushButton("+", self)
            # self.btn_decimation_seed_plus.clicked.connect(incr_decimation_seed)
            self.box_randcol = QCheckBox()
            self.box_randcol.setChecked(False)
            self.box_randcol.stateChanged.connect(self.update_render)
            self.box_enable_d = QCheckBox()
            self.box_enable_d.setChecked(True)
            self.box_enable_d.stateChanged.connect(self.update_render)

            self.box_tri_select = QCheckBox()
            self.box_tri_select.setChecked(self.enable_triangle_selection)
            self.box_tri_select.stateChanged.connect(self.update_render)

            i = 0
            layout_other.addWidget(QLabel("[Camera]", self), i, 0)
            layout_other.addWidget(line(), i, 1, 1, -1)

            i += 1
            self.control_fov = SliderControl(w, "Free cam FoV: ", min=1, max=150, step=0.1, default_value=cam.FoVx * 180 / torch.pi, on_change=self.update_render, display=display02f)    
            layout_other.addWidget(self.control_fov.label, i, 0)
            layout_other.addWidget(self.control_fov.slider, i, 1)
            layout_other.addWidget(self.control_fov.text, i, 2)

            i += 1
            layout_other.addWidget(QLabel("[Gaussians]", self), i, 0)
            layout_other.addWidget(line(), i, 1, 1, -1)

            i += 1
            self.control_decimation = SliderControl(w, "Decimation: ", min=0, max=1, step=0.01, default_value=0, on_change=self.update_render, display=display02f)
            layout_other.addWidget(self.control_decimation.label, i, 0)
            layout_other.addWidget(self.control_decimation.slider, i, 1)
            layout_other.addWidget(self.control_decimation.text, i, 2)
            # layout_other.addWidget(QLabel("Seed", self), i, 3)
            # layout_other.addWidget(self.btn_decimation_seed_minus, i, 4)
            # layout_other.addWidget(self.btn_decimation_seed_plus, i, 5)
            i += 1
            self.control_opacity = SliderControl(w, "Set opacity: ", min=0, max=1, step=0.01, default_value=1, on_change=self.update_render, display=display02f, checkbox=True)
            layout_other.addWidget(self.control_opacity.label, i, 0)
            layout_other.addWidget(self.control_opacity.slider, i, 1)
            layout_other.addWidget(self.control_opacity.text, i, 2)
            layout_other.addWidget(self.control_opacity.checkbox, i, 3)
            i += 1
            self.control_opacity_culling = SliderControl(w, "Cull opacity below: ", min=0, max=1, step=0.01, default_value=0, on_change=self.update_render, display=display02f, checkbox=True)
            layout_other.addWidget(self.control_opacity_culling.label, i, 0)
            layout_other.addWidget(self.control_opacity_culling.slider, i, 1)
            layout_other.addWidget(self.control_opacity_culling.text, i, 2)
            layout_other.addWidget(self.control_opacity_culling.checkbox, i, 3)
            i += 1
            self.control_scale_multiplier = SliderControl(w, "Scale multiplier: ", min=0, max=20, step=0.1, default_value=1, on_change=self.update_render, display=display02f, checkbox=True)
            layout_other.addWidget(self.control_scale_multiplier.label, i, 0)
            layout_other.addWidget(self.control_scale_multiplier.slider, i, 1)
            layout_other.addWidget(self.control_scale_multiplier.text, i, 2)
            layout_other.addWidget(self.control_scale_multiplier.checkbox, i, 3)
            i += 1
            layout_other.addWidget(QLabel("Random colors: ", self), i, 0)
            layout_other.addWidget(self.box_randcol, i, 1)
            i += 1
            layout_other.addWidget(QLabel("Enable displacements: ", self), i, 0)
            layout_other.addWidget(self.box_enable_d, i, 1)
            i += 1
            layout_other.addWidget(QLabel("Triangle selection: ", self), i, 0)
            layout_other.addWidget(self.box_tri_select, i, 1)

            i += 1
            layout_other.addWidget(QLabel("[Video]", self), i, 0)
            layout_other.addWidget(line(), i, 1, 1, -1)

            i += 1
            self.video_anim_choice = QComboBox(self)
            self.video_anim_choice.addItem("Static", "static")
            self.video_anim_choice.addItem("Train", "train")
            self.video_anim_choice.addItem("Test", "test")
            layout_other.addWidget(QLabel("Animation: ", self), i, 0)
            layout_other.addWidget(self.video_anim_choice, i, 1)

            i += 1
            self.box_video_filtering = QCheckBox(self)
            self.box_video_filtering.setChecked(False)
            layout_other.addWidget(QLabel("Temporal filtering: ", self), i, 0)
            layout_other.addWidget(self.box_video_filtering, i, 1)

            i += 1
            self.box_video_rotate_light = QCheckBox(self)
            self.box_video_rotate_light.setChecked(False)
            layout_other.addWidget(QLabel("Rotate light: ", self), i, 0)
            layout_other.addWidget(self.box_video_rotate_light, i, 1)

            i += 1
            btn_generate = QPushButton("Generate video", self)
            btn_generate.clicked.connect(self.generate_video)
            layout_other.addWidget(btn_generate, i, 0)

            self.img_display = ClickableImage(self)
            def on_click_img(x,y,button,shift,ctrl):
                # Triangle picking
                if button == Qt.RightButton and self.enable_triangle_selection and self.mesh_tri is not None:
                    mesh_tri = self.mesh_tri.squeeze(0)
                    if 0 <= y < mesh_tri.shape[0] and 0 <= x < mesh_tri.shape[1]:
                        tri_idx = mesh_tri[y,x,0]
                        if tri_idx == 0: self.selected_tri_idx = None # can't select triangle #0... that's ok
                        elif ctrl: self.selected_tri_idx = [tri_idx] + ([] if self.selected_tri_idx is None else self.selected_tri_idx)
                        else: self.selected_tri_idx = [tri_idx]
                        self.update_render()
                if button == Qt.LeftButton:
                    self.last_mouse_pos = x, y
            def on_release():
                self.last_mouse_pos = None
            def on_mouse_move(x, y, shift_pressed, ctrl_pressed):
                if self.last_mouse_pos is None or not self.free_cam:
                    return
                last_x, last_y = self.last_mouse_pos
                dx, dy = x - last_x, y - last_y
                mul = CAM_ORBIT_SPEED * (CTRL_ORBIT_MULTIPLIER if ctrl_pressed else 1)
                if shift_pressed:
                    self.control_cam.move_center(dx * mul * 0.2, dy * mul * 0.2)
                else:
                    self.control_cam.set_yaw(self.control_cam.yaw + dx * mul)
                    self.control_cam.set_pitch(self.control_cam.pitch + dy * mul)
                
                self.update_render()
                self.last_mouse_pos = x, y
            def on_wheel_img(d, shift_pressed, ctrl_pressed):
                if not self.free_cam:
                    return
                mul = CAM_Z_SPEED * (CTRL_Z_MULTIPLIER if ctrl_pressed else 1)
                self.control_cam.set_distance(self.control_cam.distance - d * mul)
                self.update_render()
            self.img_display.mouseMoved.connect(on_mouse_move)
            self.img_display.mousePressed.connect(on_click_img)
            self.img_display.mouseReleased.connect(on_release)
            self.img_display.wheelScrolled.connect(on_wheel_img)

            tabWidget = QTabWidget(self)

            w_column_1 = 200
            layouts = [layout_expr, layout_pose, layout_shading, layout_texturing, layout_other]
            for layout in layouts:
                assert isinstance(layout, QGridLayout) 
                # layout.setColumnMinimumWidth(1, WIDTH_SLIDERS)
                for i in range(layout.count()):
                    item = layout.itemAt(i)
                    _, col, _, _ = layout.getItemPosition(i)
                    wid = item.widget()
                    if col == 1 and not isinstance(wid, QFrame):
                        wid.setMinimumWidth(w_column_1)
                        wid.setMaximumWidth(w_column_1)


            exp_scroll_area = QScrollArea(self)
            exp_scroll_area.setWidget(widget(layout_expr, self))
            tabWidget.addTab(exp_scroll_area, "Expression")
            
            pose_scroll_area = QScrollArea(self)
            pose_scroll_area.setWidget(widget(layout_pose, self))
            tabWidget.addTab(pose_scroll_area, "Pose")

            light_scroll_area = QScrollArea(self)
            light_scroll_area.setWidget(widget(layout_shading, self))
            tabWidget.addTab(light_scroll_area, "Shading")

            texturing_scroll_area = QScrollArea(self)
            texturing_scroll_area.setWidget(widget(layout_texturing, self))
            tabWidget.addTab(texturing_scroll_area, "Texture")

            other_scroll_area = QScrollArea(self)
            other_scroll_area.setWidget(widget(layout_other, self))
            tabWidget.addTab(other_scroll_area, "Other")

            root_layout = hbox(
                self.img_display,
                vbox(
                    # self.select_seq_box,
                    hbox(self.select_mode_box, self.screenshot_btn),
                    hbox(self.slider_train_idx.label, self.slider_train_idx.slider, self.slider_train_idx.text),
                    hbox(self.slider_test_idx.label, self.slider_test_idx.slider, self.slider_test_idx.text),
                    hbox(QLabel("[F] Toggle free cam | [S] Screenshot"),),
                    hbox(QLabel("[G] Toggle GT/Render | [P] Play/Pause animation"),),
                    tabWidget
                )
            )

            self.setCentralWidget(widget(root_layout, self))

            # Set the initial view (triggers a render)
            self.slider_train_idx.value = 1

        def set_window(self, img_size: int):
            w = img_size[1] + 600
            h = img_size[0]
            if not hasattr(self, "_w") or w != self._w:
                self.setGeometry(0, 0, w, h)
                self._w = w
                centerPoint = QDesktopWidget().availableGeometry().center()
                qtRectangle = self.frameGeometry()
                qtRectangle.moveCenter(centerPoint)
                self.move(qtRectangle.topLeft())

        def env_light_matrix(self):
            rot = torch.tensor([0, self.env_rotate.value, 0], dtype=torch.float, device=device)
            from flame.lbs import batch_rodrigues
            mat = torch.eye(4, dtype=torch.float, device=device).unsqueeze(0)
            mat[:, :3, :3] = batch_rodrigues(rot.unsqueeze(0), dtype=rot.dtype)  
            return mat

        @torch.no_grad()
        def update_render(self, *_, display=True):
            view = self.view
            # view["seq_idx"][0] = self.select_seq_box.currentData()

            env_rot = self.env_light_matrix()
            use_env_background = self.box_envlight_background.isChecked()

            shader.enable_normal_map = self.box_enable_normal_map.isChecked()

            if self.box_custom_material.isChecked():
                shader.override_texture_albedo = custom_tex.albedo
                shader.override_texture_r = custom_tex.roughness
                shader.override_texture_spec = custom_tex.spec
                shader.override_texture_normals = custom_tex.normals
                shader.override_texture_mask = None if custom_tex.mask is None and self.preset_mask is None else (custom_tex.mask if custom_tex.mask is not None else 1.0) * (self.preset_mask if self.preset_mask is not None else 1.0)
            else:
                shader.override_texture_albedo, shader.override_texture_r, shader.override_texture_spec, shader.override_texture_normals, shader.override_texture_mask = None, None, None, None, None

            self.enable_triangle_selection = self.box_tri_select.isChecked()
            if self.enable_triangle_selection and self.selected_tri_idx is not None:
                g_triangle_idx, _ = avatar.gaussians.get_binding()
                g_mask = (g_triangle_idx == self.selected_tri_idx[0])
                for tri_idx in self.selected_tri_idx[1:]:
                    g_mask = torch.logical_or(g_mask, g_triangle_idx == tri_idx)
                gaussians_mask = g_mask.view(-1)
            else: 
                gaussians_mask = None
            
            if self.gaussians_mask is not None:
                gaussians_mask = torch.logical_and(gaussians_mask, self.gaussians_mask) if gaussians_mask is not None else self.gaussians_mask
    
            render_settings = RenderSettings(
                decimation_ratio = self.control_decimation.value,
                decimation_seed = self.decimation_seed,
                override_opacity = self.control_opacity.value if self.control_opacity.checked else -1,
                scaling_multiplier = self.control_scale_multiplier.value if self.control_scale_multiplier.checked else 1.0,
                opacity_culling = self.control_opacity_culling.value if self.control_opacity_culling.checked else 0,
                gaussians_mask = gaussians_mask,
                random_colors =  self.box_randcol.isChecked(),
                hw_textures = HARDWARE_TEXTURES,
                metallic = self.control_metallic.value if self.control_metallic.checked else 0.0,
                diffuse_scale = self.diffuse_scale.value,
                specular_scale = self.specular_scale.value,
                roughness_scale = self.roughness_scale.value,
                brightness_scale = self.brightness_scale.value,
                shading_normals = ["splatted", "depth", "mesh", "mesh_flat"][self.shading_normals_mode_box.currentData()],
                background_color = "black" if self.envmap is not None and use_env_background else self.background_color,
                use_env_background = use_env_background,
            )
    
            render_mode = RENDER_MODES[self.select_mode_box.currentData()]

            # Update camera
            if self.free_cam:
                cam.world_view_transform = self.control_cam.world_to_cam.to(device).transpose(0,1)
                cam.world_view_transform_inv = cam.world_view_transform.inverse()
                cam.full_proj_transform = cam.world_view_transform @ cam.projection_matrix
                cam.set_fov(self.control_fov.value * torch.pi / 180)
                # Replace the view's camera with our camera
                view = {**view, "camera": [cam]}
            view["camera"][0].set_resolution(args.resolution, args.resolution)

            avatar.deformer.displacements_scale = 1.0 if self.box_enable_d.isChecked() else 0.0

            output, get_vis = avatar.run(view, env_light=self.envmap, env_rot=env_rot, render_settings=render_settings)
            vis = get_vis(*render_mode["vis"]) if "vis" in render_mode and len(render_mode["vis"]) > 0 else dict()
            img = render_mode["fn"](view, output, vis).squeeze(0)
        
            if display:
                self.img_display.set_image(img)

                if self.enable_triangle_selection:
                    mesh_tri = avatar.get_rasterized_mesh(view)["pix_to_tri"]
                    # mesh_tri is a (1,H,W,1) image where each pixel contains the triangle index covering it (0 if none)
                    self.mesh_tri = resize(mesh_tri.permute(0,3,1,2), [img.shape[0], img.shape[1]], InterpolationMode.NEAREST).permute(0,2,3,1)

            return img, output, get_vis
        
        def grab_screenshot(self, no_background=True):
            img, output, _ = self.update_render(display=False)
            if no_background:
                alpha = output.rast_buffers["rend_alpha"]
                if self.envmap is not None and self.box_envlight_background.isChecked():
                    alpha = torch.ones_like(alpha) # keep the env light background
                img =  torch.cat((img, alpha.squeeze(0)), dim=-1).squeeze(0)
            return img

        def generate_video(self):
            anim_mode = self.video_anim_choice.currentData()
            do_anim = anim_mode != "static"
            do_rotate_light = self.box_video_rotate_light.isChecked()
            output_fps = 60

            assert do_anim or do_rotate_light

            if do_anim:
                is_train = anim_mode == "train"
                dataset = dataset_train if is_train else dataset_test
                filtering = self.box_video_filtering.isChecked()
                views = prepare_video_views(avatar, dataset, output_fps, filtering=filtering, end=VIDEO_MAX_SRC_FRAMES)
                total_frames = len(views)

            if do_rotate_light:
                light_rot_duration = 6 # time for the light to make one rotation
                light_steps = light_rot_duration * output_fps
                if not do_anim:
                    # If not animating, do exactly one rotation of the env light
                    total_frames = light_rot_duration * output_fps

            timestamp = datetime.today().strftime("%Y-%m-%d_%H-%M-%S")
            video_dir = avatar.experiment_dir / f"interact_video_{timestamp}"
            video_dir.mkdir(parents=True, exist_ok=True)

            for i in tqdm(range(total_frames), desc="Generating video frames"):
                if do_anim:
                    self._set_view(views[i], is_train=is_train)
                if do_rotate_light:
                    angle = -torch.pi + (i % light_steps) * (2*torch.pi / light_steps)
                    self.env_rotate.set_value_silently(angle)
                img = self.grab_screenshot(no_background=False)
                save_img(video_dir / f"{i:04d}.png", img)
            
            # os.system(f"/usr/bin/ffmpeg -y -framerate {output_fps} -pattern_type glob -i '{video_dir / '*.png'}' -c:v libx264 -pix_fmt yuv420p '{video_dir / 'video.mp4'}'")
            os.system(f"/usr/bin/ffmpeg -y -framerate {output_fps} -f image2 -i '{video_dir / '%04d.png'}' -c:v libx264 -crf 10 '{video_dir / 'video.avi'}'")

            # Update the render so the final frame of the video is displayed (since we were skipping the display during generation)
            self.update_render()

        def closeEvent(self, event):
            if custom_tex.poll_interval is not None:
                custom_tex.poll_interval.stop()

    app = QApplication(sys.argv)
    app.setFont(QFont("Consolas", 11))

    w = MainWindow()

    img, _, _ = w.update_render()
    w.set_window(img.shape[0:2])
    w.show()
    app.exec_()



@torch.no_grad()
def prepare_video_views(avatar: Avatar, dataset, target_fps: int = None, filtering: bool = False, start=None, end=None):
    device = avatar.device
    source_fps = avatar.args.source_fps
    target_fps = target_fps or source_fps

    start = start or 0
    end = end or len(dataset)
    subset = torch.utils.data.Subset(dataset, range(start, min(end, len(dataset))))

    # Preload the dataset
    all_views = [dataset_util.to_device_recursive(dataset.collate([x]), device) for x in tqdm(subset, desc="Loading all views")]

    if avatar.args.detached:
        poses, exprs = [v["flame_pose"] for v in all_views], [v["flame_expression"] for v in all_views]
    else:
        poses, exprs = zip(*[avatar.compute_flame_attrs(v, is_train=(dataset == avatar.dataset_train)) for v in tqdm(all_views, desc="Computing FLAME attributes")])
    
    poses, exprs = torch.cat(poses), torch.cat(exprs) # shape (n_views, n_params)
        
    if source_fps != target_fps:
        # Subsample poses and expressions to match the target FPS
        frame_interval = source_fps / target_fps
        i_before = torch.arange(0, len(subset), frame_interval).floor().long()
        i_after = (i_before + 1).clamp(max=len(subset)-1)
        alpha = torch.arange(0, len(subset), frame_interval).unsqueeze(1).to(device) % 1
        poses = poses[i_before] * (1 - alpha) + poses[i_after] * alpha
        exprs = exprs[i_before] * (1 - alpha) + exprs[i_after] * alpha
        all_views = [all_views[i] for i in i_before]

    if filtering:
        # Apply temporal filtering to FLAME poses and expressions
        conv_weights = gaussian_kernel(ksize=9, sigma=2).to(device)
        poses = apply_featurewise_conv1d(poses, conv_weights, pad_mode="replicate")
        exprs = apply_featurewise_conv1d(exprs, conv_weights, pad_mode="replicate")

    return [
        {**view, "flame_pose": pose.unsqueeze(0), "flame_expression": expr.unsqueeze(0)}
        for view, pose, expr in zip(all_views, poses, exprs)
    ]

class CustomTextures():
    def __init__(self, polling_interval: float, load_tex_fn: Callable[[str, str], Tensor], on_change: Callable[[], None]):
        self.dir = None
        self.load_tex_fn = load_tex_fn
        self.on_change = on_change
        self.textures = dict({
            "albedo": {"tensor": None, "last_mtime": None, "match": ["albedo.", "color."]},
            "roughness": {"tensor": None, "last_mtime": None, "match": ["roughness."]},
            "spec": {"tensor": None, "last_mtime": None, "match": ["spec.", "f0.", "specular."]},
            "normals": {"tensor": None, "last_mtime": None, "match": ["normals.", "normal.", "normalgl."]},
            "mask": {"tensor": None, "last_mtime": None, "match": ["mask."]},
        })

        if polling_interval > 0:
            self.poll_interval = SetInterval(polling_interval, self.poll_for_changes)

    def reset(self):
        self.dir = None
        for v in self.textures.values():
            v["tensor"] = None
            v["last_mtime"] = None

    def set_dir(self, dir: Path):
        self.dir = dir
    
    def _load(self, key: str, fp: Path):
        tex = self.load_tex_fn(fp, key)
        self.textures[key]["tensor"] = tex
        self.textures[key]["last_mtime"] = os.path.getmtime(fp)
        # logging.info(f"Loaded custom texture {key} from {fp}")

    def poll_for_changes(self, force=False):
        if self.dir is None:
            return False
        if not self.dir.exists():
            self.reset()
            return True

        files = os.listdir(self.dir)
        def _find(queries):
            for q in queries:
                for f in files:
                    if q.lower() in f.lower() and f.lower().endswith(('.png', '.jpg', '.jpeg')):
                        return self.dir / f
            return None

        changed = False
        for k, v in self.textures.items():
            fp = _find(v["match"])
            if fp is not None:
                # Check if the texture file has been modified
                if v["last_mtime"] != os.path.getmtime(fp) or force:
                    self._load(k, fp)
                    changed = True
        if changed:
            self.on_change()
        return changed

    @property
    def albedo(self): return self.textures["albedo"]["tensor"]
    @property
    def roughness(self): return self.textures["roughness"]["tensor"]
    @property
    def spec(self): return self.textures["spec"]["tensor"]
    @property
    def normals(self): return self.textures["normals"]["tensor"]
    @property
    def mask(self): return self.textures["mask"]["tensor"]

class SetInterval(QTimer):
    def __init__(self, interval, action) :
        super().__init__()
        self.timeout.connect(action)
        self.start(round(interval * 1000)) # QTimer works with milliseconds

    def stop(self):
        super().stop()

class FloatSlider(QSlider):
    def __init__(self, *args, **kargs):
        super(QSlider, self).__init__( *args, **kargs)
        self._min = 0
        self._max = 99
        self.interval = 1

    def setValue(self, value):
        index = round((value - self._min) / self.interval)
        return super(FloatSlider, self).setValue(index)

    def setRange(self, min, max, step):
        self.setMinimum(min)
        self.setMaximum(max)
        self.setInterval(step)

    def value(self):
        return super(FloatSlider, self).value() * self.interval + self._min

    def setMinimum(self, value):
        self._min = value
        self._range_adjusted()

    def setMaximum(self, value):
        self._max = value
        self._range_adjusted()

    def setInterval(self, value):
        # To avoid division by zero
        if not value:
            raise ValueError('Interval of zero specified')
        self.interval = value
        self._range_adjusted()

    def _range_adjusted(self):
        number_of_steps = int((self._max - self._min) / self.interval)
        super(FloatSlider, self).setMaximum(number_of_steps)

class SliderControl:
    def __init__(self, parent: QWidget, label: str, min=0, max=1, step=0.01, default_value=0, on_change=None, checkbox=False, textbox=True, display=None):
        display = display or (lambda x: str(x))
        on_change = on_change or (lambda v: None)

        self.display = display
        self._v = default_value
        self.on_change = on_change
        self.label = QLabel(label, parent)
        self.slider = FloatSlider(Qt.Horizontal, parent)
        self.slider.setRange(min, max, step)
        self.slider.setValue(default_value)                        

        self.textbox = textbox
        if textbox:
            self.text = QLineEdit(display(default_value), parent)
            self.text.setMaximumWidth(60)
        else:
            self.text = QLabel(display(default_value), parent)

        self._checked = False
        if checkbox:
            self.checkbox = QCheckBox()
            self.checkbox.setChecked(self._checked)
            def on_change_checkbox():
                self._checked = self.checkbox.isChecked()
                self.slider.setEnabled(self._checked)
                self.text.setEnabled(self._checked)
                on_change(self._v)
            self.slider.setEnabled(self._checked)
            self.text.setEnabled(self._checked)
            self.checkbox.stateChanged.connect(on_change_checkbox)

        def on_change_slider():
            self._v = self.slider.value()
            self.text.blockSignals(True)
            self.text.setText(self.display(self._v))
            self.text.blockSignals(False)
            on_change(self._v)
        self.slider.valueChanged.connect(on_change_slider)

        if textbox:
            def on_change_text():
                try:
                    self._v = float(self.text.text())
                    self.slider.blockSignals(True)
                    self.slider.setValue(self._v)
                    self.slider.blockSignals(False)
                    on_change(self._v)
                except Exception as e:
                    print(e)
            self.text.textChanged.connect(on_change_text)
    
    @property
    def value(self):
        return self._v
    
    @value.setter
    def value(self, v: float):
        self.set_value_silently(v)
        self.on_change(v)
    
    def set_value_silently(self, v: float):
        self.slider.blockSignals(True)
        self.slider.setValue(v)
        self.slider.blockSignals(False)
        self.text.blockSignals(True)
        self.text.setText(self.display(v))
        self.text.blockSignals(False)
        self._v = v
        
    @property
    def checked(self):
        return self._checked

class ClickableImage(QLabel):
    mousePressed = pyqtSignal(int, int, int, bool, bool)
    mouseReleased = pyqtSignal()
    mouseMoved = pyqtSignal(int, int, bool, bool)
    wheelScrolled = pyqtSignal(int, bool, bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        # self.setAlignment(Qt.AlignCenter)
        self.setAlignment(Qt.AlignBottom)

    def set_image(self, img_tensor):
        img = convert_uint(img_tensor)
        height, width, _ = img.shape
        bytesPerLine = 3 * width
        qImg = QImage(img.data, width, height, bytesPerLine, QImage.Format_RGB888)
        self.setPixmap(QPixmap.fromImage(qImg))

    def _convert_raw_pos(self, x, y):
        label_width = self.width()
        label_height = self.height()
        pixmap_width = self.pixmap().width()
        pixmap_height = self.pixmap().height()
        
        # Compute top-left corner of pixmap inside the QLabel
        # This only works for bottom-center alignment
        offset_x = (label_width - pixmap_width) // 2
        offset_y = (label_height - pixmap_height)

        return x - offset_x, y - offset_y

    def mousePressEvent(self, event):
        # Get click position inside QLabel
        x = event.pos().x()
        y = event.pos().y()
        img_x, img_y = self._convert_raw_pos(x, y)
        is_shift_pressed = event.modifiers() & Qt.ShiftModifier
        is_ctrl_pressed = event.modifiers() & Qt.ControlModifier
        self.mousePressed.emit(img_x, img_y, event.button(), is_shift_pressed, is_ctrl_pressed)

    def mouseReleaseEvent(self, event):
        self.mouseReleased.emit()

    def mouseMoveEvent(self, event):
        img_x, img_y = self._convert_raw_pos(event.pos().x(), event.pos().y())
        is_shift_pressed = event.modifiers() & Qt.ShiftModifier
        is_ctrl_pressed = event.modifiers() & Qt.ControlModifier
        self.mouseMoved.emit(img_x, img_y, is_shift_pressed, is_ctrl_pressed)

    def wheelEvent(self, event):
        """ Detect mouse wheel scrolling. (positive delta -> scroll up / negative delta -> scroll down) """
        delta = event.angleDelta().y()  # y() gives vertical scroll amount
        is_shift_pressed = event.modifiers() & Qt.ShiftModifier
        is_ctrl_pressed = event.modifiers() & Qt.ControlModifier
        self.wheelScrolled.emit(delta, is_shift_pressed, is_ctrl_pressed)

class OrbitCamera:
    def __init__(self, center, distance=1, min_distance=1e-5, yaw=0, pitch=0, device=None, dtype=torch.float32):
        """
        Initialize the orbit controller.
        
        Args:
            center (list/tuple/torch.Tensor): The point to orbit around (3D).
            device (str): "cpu" or "cuda".
            dtype (torch.dtype): torch data type.
        """
        if device is None:
            device = center.device if torch.is_tensor(center) else "cpu"
        self.device = device
        self.dtype = dtype
        self.center = center.clone().to(device) if torch.is_tensor(center) else torch.tensor(center, device=device, dtype=dtype)
        self.min_distance = min_distance
        self.yaw = yaw
        self.pitch = pitch
        self.distance = distance
        self._world_to_cam, self._cam_to_world = None, None

    def move_center(self, dx, dy, dz=0):
        self.center += (self.cam_to_world[:3,:3] @ torch.tensor([-dx, -dy, dz], dtype=torch.float, device=self.device))
        self._invalidate_transform()

    def set_center(self, xyz: Tensor):
        self.center = xyz.to(self.device).to(self.dtype)
        self._invalidate_transform()

    def set_yaw(self, yaw):
        """
        Set the camera yaw angle.
        Args:
            yaw (float): Rotation around the Y axis (horizontal), in radians.
        """
        self.yaw = yaw
        self._invalidate_transform()

    def set_pitch(self, pitch):
        """
        Set the camera pitch angle.
        Args:
            pitch (float): Rotation around the X axis (vertical), in radians.
        """
        # clamp to avoid flipping
        pitch = max(-torch.pi/2 + 0.01, min(torch.pi/2 - 0.01, pitch))
        self.pitch = pitch
        self._invalidate_transform()

    def set_distance(self, d):
        """
        Set the distance to the center.
        Args:            
            distance (float): Distance from the orbit center.
        """
        self.distance = max(d, self.min_distance)
        self._invalidate_transform()

    @property
    def world_to_cam(self):
        if self._world_to_cam is None:
            self._world_to_cam, self._cam_to_world = self._compute_transform()
        return self._world_to_cam

    @property
    def cam_to_world(self):
        if self._cam_to_world is None:
            self._world_to_cam, self._cam_to_world = self._compute_transform()
        return self._cam_to_world

    def _invalidate_transform(self):
        self._world_to_cam, self._cam_to_world = None, None

    def _compute_transform(self):
        device = self.device
        dtype = self.dtype
        normalize = lambda x: torch.nn.functional.normalize(x, dim=-1)

        yaw, pitch, distance = self.yaw, self.pitch, self.distance
        yaw = yaw.to(device) if torch.is_tensor(yaw) else torch.tensor(yaw, device=device, dtype=dtype)
        pitch = pitch.to(device) if torch.is_tensor(pitch) else torch.tensor(pitch, device=device, dtype=dtype)
        distance = distance.to(device) if torch.is_tensor(distance) else torch.tensor(distance, device=device, dtype=dtype)

        # Clamp pitch to avoid flipping
        pitch = torch.clamp(pitch, -torch.pi/2 + 1e-5, torch.pi/2 - 1e-5)

        # Spherical to Cartesian position
        x = distance * torch.cos(pitch) * torch.sin(-yaw)
        y = distance * torch.sin(pitch)
        z = distance * torch.cos(pitch) * torch.cos(-yaw)
        position = self.center + torch.stack([x, y, z])

        # Build Look-At transform
        up = torch.tensor([0.0, 1.0, 0.0], device=device, dtype=dtype)
        forward = normalize(self.center - position)
        right = normalize(torch.cross(up, -forward, dim=-1))
        up_corrected = normalize(torch.cross(forward, right, dim=-1))

        # Camera-to-world (local axes as rows)
        cam_to_world = torch.eye(4, device=device, dtype=dtype)
        cam_to_world[0, :3] = right
        cam_to_world[1, :3] = up_corrected
        cam_to_world[2, :3] = forward
        cam_to_world[:3, 3] = position

        # World-to-camera (view matrix)
        world_to_cam = torch.eye(4, device=device, dtype=dtype)
        rot = cam_to_world[:3, :3]#.T
        trans = -rot @ position
        world_to_cam[:3, :3] = rot
        world_to_cam[:3, 3] = trans

        return world_to_cam, cam_to_world

    def clone(self):
        return OrbitCamera(self.center, self.distance, self.min_distance, self.yaw, self.pitch, self.device, self.dtype)


if __name__ == "__main__":
    parser = create_parser()
    arg = parser.add_argument
    arg("--resolution", type=int, default=512, help="Render resolution for the interactive viewer.")
    args = parse_args(parser)

    setup_logging()

    avatar = Avatar(args)
    avatar.init_modules()
    main(args, avatar)