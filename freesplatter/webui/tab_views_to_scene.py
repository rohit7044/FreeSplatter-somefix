import os
import glob
import gradio as gr
from functools import partial
from PIL import Image
from .gradio_custommodel3d import CustomModel3D
from .gradio_customgs import CustomGS


def create_interface_views_to_scene(freesplatter_api):
    example_root = 'examples/views_to_scene'
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
                    var_dict['in_image_1'] = gr.Image(
                        label='Input image 1',
                        type='pil', 
                        image_mode='RGB', 
                    )
                    var_dict['in_image_2'] = gr.Image(
                        label='Input image 2',
                        type='pil', 
                        image_mode='RGB', 
                    )
                
                with gr.Row(equal_height=False):
                    var_dict['run_btn'] = gr.Button('Reconstruct', variant='primary', scale=2)
                
                with gr.Row():
                    var_dict['out_multiview'] = gr.Image(
                        label='Input views', 
                        interactive=False,
                        visible=False,
                    )

                gr.Examples(
                    examples=examples,
                    inputs=[var_dict['in_image_1'], var_dict['in_image_2']],
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

        var_dict['run_btn'].click(
            fn=partial(freesplatter_api, cache_dir=interface.GRADIO_CACHE),
            inputs=[var_dict['in_image_1'], 
                    var_dict['in_image_2']],
            outputs=[var_dict['out_multiview'], var_dict['out_gs_vis'], var_dict['out_video'], var_dict['out_pose']], 
            concurrency_id='default_group',
            api_name='run_views_to_3d',
        )

    return interface, var_dict
