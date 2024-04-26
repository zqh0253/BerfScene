import gradio as gr
from models import build_model
from PIL import Image
import numpy as np
import torchvision
import math
import ninja
import torch
from tqdm import trange
import imageio
import requests
import argparse
import imageio
import datetime
from scipy.spatial.transform import Rotation  

from gradio_draggable import Draggable

checkpoint = 'clevr.pth'
state = torch.load(checkpoint, map_location='cpu')
G = build_model(**state['model_kwargs_init']['generator_smooth'])
o0, o1 = G.load_state_dict(state['models']['generator_smooth'], strict=False)
G.eval().cuda()
G.backbone.synthesis.input.x_offset =0
G.backbone.synthesis.input.y_offset =0
G_kwargs= dict(noise_mode='const',
                fused_modulate=False,
                impl='cuda',
                fp16_res=None)

COLOR_NAME_LIST = ['cyan', 'green', 'purple', 'red', 'yellow', 'gray', 'purple', 'blue']
SHAPE_NAME_LIST = ['cube', 'sphere', 'cylinder']
MATERIAL_NAME_LIST = ['rubber', 'metal']

canvas_x = 800
canvas_y = 200
batch_size = 1
code = torch.randn(1, G.z_dim).cuda()
to_pil = torchvision.transforms.ToPILImage()

RT = torch.tensor([[ -1.0000,   0.0000,   0.0000,   0.0000,   0.0000,   0.5000,  -0.8660,
          10.3923,   0.0000,  -0.8660,  -0.5000,   6.0000,   0.0000,   0.0000,
           0.0000,   1.0000, 262.5000,   0.0000,  32.0000,   0.0000, 262.5000,
          32.0000,   0.0000,   0.0000,   1.0000]], device='cuda')

obj_dict = {}

# init
fake_bevs = torch.zeros([1, 14, 256, 256], device='cuda').float()
_ = G(code, RT, fake_bevs)


def trans(x, y, z, length):
    w = h = length
    x = 0.5 * w - 128 + 256 - (x/9 + .5) * 256
    y = 0.5 * h - 128 + (y/9 + .5) * 256
    z = z / 9 * 256
    return x, y, z

def objs_to_canvas(lst, length=256, scale = 2.6):
    objs = []
    for each in lst:
        x, y, obj_id = each['x'], each['y'], each['id']

        if obj_id not in obj_dict:
            color = np.random.choice(COLOR_NAME_LIST)
            shape = 'cube'
            material = 'rubber'
            rot = 0
            obj_dict[obj_id] = [color, shape, material, rot]

        color, shape, material, rot = obj_dict[obj_id]
        x = -x / canvas_x * 16
        y = y / canvas_y * 2
        y *= 2
        x += 1.0
        y -= 1.5
        z = 0.35
        objs.append([x, y, z, shape, color, material, rot])

    h, w = length, int(length *scale)
    nc = 14
    canvas = np.zeros([h, w, nc])
    xx = np.ones([h,w]).cumsum(0)
    yy = np.ones([h,w]).cumsum(1)
    
    for x, y, z, shape, color, material, rot in objs:
        y, x, z = trans(x, y, z, length)
        
        feat = [0] * nc
        feat[0] = 1
        feat[COLOR_NAME_LIST.index(color) + 1] = 1
        feat[SHAPE_NAME_LIST.index(shape) + 1 + len(COLOR_NAME_LIST)] = 1
        feat[MATERIAL_NAME_LIST.index(material) + 1 + len(COLOR_NAME_LIST) + len(SHAPE_NAME_LIST)] = 1
        feat = np.array(feat)
        rot_sin = np.sin(rot / 180 * np.pi)
        rot_cos = np.cos(rot / 180 * np.pi)

        if shape == 'cube':
            mask = (np.abs(+rot_cos * (xx-x) + rot_sin * (yy-y)) <= z) * \
                   (np.abs(-rot_sin * (xx-x) + rot_cos * (yy-y)) <= z)
        else:
            mask = ((xx-x)**2 + (y-yy)**2) ** 0.5 <= z
        canvas[mask] = feat
    canvas = np.transpose(canvas, [2, 0, 1]).astype(np.float32)
    return canvas
 
@torch.no_grad()
def predict_local_view(lst):
    canvas = torch.tensor(objs_to_canvas(lst)).cuda()[None]
    bevs = canvas[..., 0: 0+256]
    gen = G(code, RT, bevs)
    rgb = gen['gen_output']['image'][0] * .5 + .5
    return to_pil(rgb)

@torch.no_grad()
def predict_local_view_video(lst):
    canvas = torch.tensor(objs_to_canvas(lst)).cuda()[None]
    bevs = canvas[..., 0: 0+256]
    RT_array = np.array(RT[0].cpu())
    rot = RT_array[:16].reshape(4,4)
    trans = RT_array[16:]
    rot_new = rot.copy()
    r = Rotation.from_matrix(rot[:3, :3])
    angles = r.as_euler("zyx",degrees=True)
    v_mean, h_mean = angles[1], angles[2]

    video_path = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + '.mp4'
 
    writer = imageio.get_writer(video_path, fps=25)
    for t in np.linspace(0, 1, 50):
        angles[1] = 0.5 * np.cos(t * 2 * math.pi) + v_mean
        angles[2] = 1 * np.sin(t * 2 * math.pi) + h_mean
        r = Rotation.from_euler("zyx",angles,degrees=True)
        rot_new[:3,:3] = r.as_matrix()
        new_RT = torch.tensor(np.concatenate([rot_new.flatten(), trans])[None]).cuda().float()
        gen = G(code, new_RT, bevs)
        rgb = gen['gen_output']['image'][0] * .5 + .5
        writer.append_data(np.array(to_pil(rgb)))
    writer.close()
    return video_path

@torch.no_grad()
def predict_global_view(lst):
    canvas = torch.tensor(objs_to_canvas(lst)).cuda()[None]
    length = canvas.shape[-1]
    lines = []
    for i in trange(0, length - 256, 10):
        bevs = canvas[..., i: i+256]
        gen = G(code, RT, bevs)
        start = 128 if i > 0 else 0
        lines.append(gen['gen_output']['image'][0, ..., start:128+32])
    rgb = torch.cat(lines, 2) * .5 + .5
    return to_pil(rgb)

with gr.Blocks() as demo:
    gr.Markdown(
            """
            # BerfScene: Bev-conditioned Equivariant Radiance Fields for Infinite 3D Scene Generation
            Qihang Zhang, Yinghao Xu, Yujun Shen, Bo Dai, Bolei Zhou*, Ceyuan Yang* (*Corresponding Author)<br>
            [Arxiv Report](https://arxiv.org/abs/2312.02136) | [Project Page](https://zqh0253.github.io/BerfScene/) | [Github](https://github.com/zqh0253/BerfScene)
            """
        )

    gr.Markdown(
        """
        ### Quick Start
        1. Drag and place objects in the canvas.
        2. Click `Add object` to insert object into the canvas.
        3. Click `Reset` to clean the canvas.
        4. Click `Get local view` to synthesize local 3D scenes.
        5. Click `Get global view` to synthesize global 3D scenes.
        """
    )

    with gr.Row():
        with gr.Column():
            
            drag = Draggable()
            with gr.Row():
                submit_btn_local = gr.Button("Get local view", variant='primary')
                submit_btn_global = gr.Button("Get global view", variant='primary')
            
        with gr.Column():
            with gr.Row():
                single_view_image = gr.Image(label='single view', interactive=False)
                single_view_video = gr.Video(label='mutli-view', interactive=False, autoplay=True)

            global_view_image = gr.Image(label='global view', interactive=False)


    submit_btn_local.click(fn=predict_local_view, inputs=drag, outputs=single_view_image)
    submit_btn_local.click(fn=predict_local_view_video, inputs=drag, outputs=single_view_video)
    submit_btn_global.click(fn=predict_global_view, inputs=drag, outputs=global_view_image)


parser = argparse.ArgumentParser()
parser.add_argument('--port', type=int, help='The port number', default=7860)
args = parser.parse_args()

demo.queue()
demo.launch(server_name='0.0.0.0', server_port=args.port, debug=True, show_error=True)
