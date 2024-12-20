import os
import glob
import gradio as gr
from functools import partial
from PIL import Image
from .gradio_custommodel3d import CustomModel3D
from .gradio_customgs import CustomGS


def create_interface_views_to_3d(freesplatter_api):
    example_root = 'examples/views_to_3d'
    examples = []
    for dir in os.listdir(example_root):
        sample_dir = os.path.join(example_root, dir)
        input_files = sorted(glob.glob(f'{sample_dir}/*.png')+glob.glob(f'{sample_dir}/*.jpg'))
        examples.append(input_files)

    var_dict = dict()
    with gr.Blocks(analytics_enabled=False) as interface:
        with gr.Row():
            with gr.Column(scale=1):
                with gr.Row():
                    var_dict['image_files'] = gr.Gallery(
                        label='Input images',
                        type="filepath",
                        file_types=['image'],
                        show_label=False,
                        columns=[5],
                        rows=[2],
                        object_fit='contain',
                        height="auto",
                    )
                
                with gr.Accordion("Reconstruction settings", open=True):
                    with gr.Row():
                        var_dict['do_rembg'] = gr.Checkbox(
                            label='Remove background', 
                            value=False, 
                            container=False,
                        )
                    with gr.Row():
                        var_dict['gs_type'] = gr.Radio(
                            choices=['2DGS', '3DGS'],
                            value='2DGS', 
                            type='value',
                            label='Gaussian splatting type', 
                            info='2DGS often leads to better mesh geometry',
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
                    var_dict['run_btn'] = gr.Button('Reconstruct', variant='primary', scale=2)
                
                with gr.Row():
                    var_dict['out_multiview'] = gr.Image(
                        label='Input views', 
                        interactive=False,
                        visible=False,
                    )

                snapshot_1 = gr.Image(None, visible=False, image_mode='RGBA')
                snapshot_2 = gr.Image(None, visible=False, image_mode='RGBA')
                snapshot_3 = gr.Image(None, visible=False, image_mode='RGBA')
                snapshot_4 = gr.Image(None, visible=False, image_mode='RGBA')
                def set_gallery_images(*images):
                    return list(images)

                gr.Examples(
                    examples=examples,
                    fn=set_gallery_images,
                    inputs=[snapshot_1, snapshot_2, snapshot_3, snapshot_4],
                    outputs=[var_dict['image_files']],
                    run_on_click=True,
                    cache_examples=False,
                    label='Examples (click one of the rows below to start)',
                    examples_per_page=5,
                )

            with gr.Column(scale=1):
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
            fn=partial(freesplatter_api, cache_dir=interface.GRADIO_CACHE),
            inputs=[var_dict['image_files'], 
                    var_dict['do_rembg'], 
                    var_dict['gs_type'], 
                    var_dict['mesh_reduction']],
            outputs=[var_dict['out_multiview'], var_dict['out_gs_vis'], var_dict['out_video'], var_dict['out_mesh'], var_dict['out_pose']], 
            concurrency_id='default_group',
            api_name='run_views_to_3d',
        )

    return interface, var_dict
