import random
import gradio as gr
from functools import partial
from .gradio_custommodel3d import CustomModel3D
from .gradio_customgs import CustomGS


def create_interface_img_to_3d(segmentation_api, freesplatter_api, model='Zero123++ v1.2'):
    default_views = {
        'Zero123++ v1.1': ['Input', 'V2', 'V3', 'V5'],
        'Zero123++ v1.2': ['V1', 'V2', 'V3', 'V5', 'V6'],
        'Hunyuan3D Std': ['V1', 'V2', 'V4', 'V6']
    }
    views_info = {
        'Zero123++ v1.1': 'View poses (azimuth, elevation): V1(30, 30), V2(90, -20), V3(150, 30), V4(-150, -20), V5(-90, 30), V6(-30, -20)',
        'Zero123++ v1.2': 'View poses (azimuth, elevation): V1(30, 20), V2(90, -10), V3(150, 20), V4(-150, -10), V5(-90, 20), V6(-30, -10)',
        'Hunyuan3D Std': 'View poses (azimuth, elevation): V1(0, 0), V2(60, 0), V3(120, 0), V4(180, 0), V5(-120, 0), V6(-60, 0)',
    }

    var_dict = dict()
    with gr.Blocks(analytics_enabled=False) as interface:
        with gr.Row():
            with gr.Column(scale=1):
                with gr.Row():
                    var_dict['in_image'] = gr.Image(
                        label='Input image',
                        type='pil', 
                        image_mode='RGBA', 
                    )
                    var_dict['fg_image'] = gr.Image(
                        label='Segmented foreground', 
                        type='pil', 
                        interactive=False, 
                        image_mode='RGBA',
                    )
                
                with gr.Accordion("Diffusion settings", open=True):
                    with gr.Row():
                        var_dict['do_rembg'] = gr.Checkbox(
                            label='Remove background', 
                            value=True, 
                            container=False,
                        )
                    with gr.Row():
                        with gr.Column():
                            var_dict['seed'] = gr.Number(
                                label='Random seed', 
                                value=42, 
                                min_width=100, 
                                precision=0, 
                                minimum=0, 
                                maximum=2 ** 31,
                                elem_classes=['force-hide-container'],
                            )
                            var_dict['random_seed'] = gr.Button(
                                '\U0001f3b2\ufe0f Try your luck!', 
                                elem_classes=['tool'],
                            )
                        with gr.Column():
                            var_dict['diffusion_steps'] = gr.Slider(
                                label="Sampling steps",
                                minimum=15,
                                maximum=75,
                                value=30,
                                step=5,
                            )
                            var_dict['guidance_scale'] = gr.Slider(
                                label="Guidance scale",
                                minimum=1,
                                maximum=10,
                                value=4,
                                step=1,
                            )

                with gr.Accordion("Reconstruction settings", open=True):
                    with gr.Row():
                        var_dict['view_indices'] = gr.CheckboxGroup(
                            choices=['Input', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6'],
                            value=default_views[model],
                            type='index',
                            label='Views used for reconstruction',
                            info='Using input image is only recommended for Zero123++ v1.1',
                        )
                    with gr.Row():
                        var_dict['gs_type'] = gr.Radio(
                            choices=['2DGS', '3DGS'],
                            value='2DGS', 
                            type='value',
                            label='Gaussian splatting type', 
                            info='2DGS often leads to better mesh geometry'
                        )
                        var_dict['mesh_reduction'] = gr.Slider(
                            label="Mesh simplification ratio",
                            info='Larger ratio leads to less faces',
                            minimum=0.8,
                            maximum=0.95,
                            value=0.95,
                            step=0.05,
                        )
                with gr.Row(equal_height=False):
                    var_dict['run_btn'] = gr.Button('Generate', variant='primary', scale=2)
                with gr.Row(visible=False):
                    var_dict['model'] = gr.Textbox(value=model, label='Model')

                gr.Examples(
                    examples='examples/img_to_3d',
                    inputs=var_dict['in_image'],
                    cache_examples=False,
                    label='Examples (click one of the images below to start)',
                    examples_per_page=21,
                )

            with gr.Column(scale=1):
                var_dict['out_multiview'] = gr.Image(
                    label='Generated views', 
                    interactive=False, 
                    image_mode='RGBA',
                )
                var_dict['out_pose'] = gr.Plot(
                    label='Estimated poses', 
                )
                var_dict['out_gs_vis'] = CustomGS(
                    label='Output GS', 
                    interactive=False, 
                    height=320,
                )
                var_dict['out_video'] = gr.Video(
                    label='Output video', 
                    interactive=False, 
                    autoplay=True, 
                    height=320,
                )
                var_dict['out_mesh'] = CustomModel3D(
                    label='Output mesh', 
                    interactive=False, 
                    height=400,
                )

        var_dict['run_btn'].click(
            fn=segmentation_api, 
            inputs=var_dict['in_image'],
            outputs=var_dict['fg_image'], 
            concurrency_id='default_group',
            api_name='run_segmentation',
        ).success(
            fn=partial(freesplatter_api, cache_dir=interface.GRADIO_CACHE),
            inputs=[var_dict['fg_image'], 
                    var_dict['model'], 
                    var_dict['diffusion_steps'], 
                    var_dict['guidance_scale'], 
                    var_dict['seed'], 
                    var_dict['view_indices'],
                    var_dict['gs_type'], 
                    var_dict['mesh_reduction']],
            outputs=[var_dict['out_multiview'], var_dict['out_gs_vis'], var_dict['out_video'], var_dict['out_mesh'], var_dict['out_pose']], 
            concurrency_id='default_group',
            api_name='run_image_to_3d',
        )

        var_dict['random_seed'].click(
            fn=lambda: random.randint(0, 2 ** 31),
            outputs=var_dict['seed'],
            show_progress=False,
            api_name=False,
        )

    return interface, var_dict
