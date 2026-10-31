# Prism conditioning and sparse-mode settings. Sampling is the stock KSampler.
import math
import numpy as np
from PIL import Image
import torch
import comfy.model_base
import comfy.utils
import comfy.latent_formats
import comfy.nested_tensor


def prepare_reference_image(image, width, height, mode):
    """Resize the first reference image with the selected upstream-compatible method."""
    if mode == 'legacy_center_crop':
        return comfy.utils.common_upscale(image[:1,:,:,:3].movedim(-1,1), width,height,'bilinear','center').movedim(1,-1).cpu()
    if mode != 'official_lanczos':
        raise ValueError('Unknown Prism reference resize mode')
    pixels=(image[0,:,:,:3].cpu().clamp(0,1)*255).round().to(torch.uint8).numpy()
    pixels=np.asarray(Image.fromarray(pixels).resize((width,height),resample=Image.Resampling.LANCZOS)).copy()
    return torch.from_numpy(pixels).float().unsqueeze(0)/255.0


def latent_statistics(device, dtype):
    """Return Wan 2.1 normalization statistics on the requested device and dtype."""
    layout=comfy.latent_formats.Wan21()
    return layout.latents_mean.to(device=device,dtype=dtype), layout.latents_std.to(device=device,dtype=dtype)


def prepare_av_reference(video_vae,image,width,height,frames,fps,reference_resize):
    """Encode the image reference and construct stock-sampler video/audio latents."""
    if video_vae.latent_channels != 16 or video_vae.downscale_index_formula != (4,8,8):
        raise ValueError('Use the 16-channel Wan 2.1 video VAE.')
    if fps <= 0:
        raise ValueError('fps must be positive')
    width,height = math.ceil(width/16)*16,math.ceil(height/16)*16
    frames = max(1,math.ceil((frames-1)/4)*4+1)
    first = prepare_reference_image(image,width,height,reference_resize)
    pixels = torch.full((frames,height,width,3),0.5,dtype=torch.float32)
    pixels[:1] = first
    raw = video_vae.encode(pixels).float().cpu()
    mean,std = latent_statistics(raw.device,raw.dtype)
    reference = (raw-mean)/std
    mask = torch.zeros((1,4,reference.shape[2],reference.shape[3],reference.shape[4]))
    mask[:,:,0] = 1
    reference = torch.cat((mask,reference),dim=1)
    # Empty raw video latent is the mean, not all zeros: the native guider
    # normalizes it before sampling. Audio zeros represent the clean latent.
    # At sigma=1 stock KSampler generates both noise streams in their native
    # shapes, then packs them. The model handles the audio coordinate scale.
    video = mean.expand_as(raw).clone()
    samples = int(48000*frames/fps)
    audio = torch.zeros((1,128,math.ceil(samples/960)),dtype=torch.float32)
    latent = {'samples':comfy.nested_tensor.NestedTensor((video,audio)),
              'sample_rate':48000,'num_samples':samples}
    return reference,latent

class PrismAttention:
    @classmethod
    def INPUT_TYPES(cls):
        """Expose explicit attention mode and block-selection controls."""
        return {'required': {
            'model': ('MODEL',),
            'attention': (['dense','prism_sparse','prism_sparse_tail_safe'], {'default':'prism_sparse_tail_safe'}),
            'sparsity': ('FLOAT', {'default':0.75,'min':0.0,'max':0.99}),
            'cdf_threshold': ('FLOAT', {'default':0.2,'min':0.01,'max':1.0}),
        }}
    RETURN_TYPES = ('MODEL',)
    FUNCTION = 'configure'
    CATEGORY = 'model/conditioning/prism'

    def configure(self,model,attention,sparsity,cdf_threshold):
        """Clone the model patcher and set Prism attention options without mutating weights."""
        if not isinstance(model.model,comfy.model_base.Prism):
            raise ValueError('Prism attention settings require a Prism model.')
        if attention not in ('dense','prism_sparse','prism_sparse_tail_safe'):
            raise ValueError('Unknown Prism attention mode')
        configured = model.clone()
        options = configured.model_options['transformer_options'] = configured.model_options.get('transformer_options',{}).copy()
        options.update(prism_attention=attention,prism_sparsity=sparsity,prism_cdf_threshold=cdf_threshold)
        return (configured,)

def attach_av_conditions(video_conditions,audio_conditions,reference,fps):
    """Attach audio context and reference metadata while preserving conditioning schedules."""
    if len(audio_conditions) not in (1,len(video_conditions)):
        raise ValueError('Audio conditioning must have one global entry or match the video conditioning entries.')
    result = []
    for i,(context,metadata) in enumerate(video_conditions):
        audio_context = audio_conditions[0 if len(audio_conditions)==1 else i][0]
        if context.ndim!=3 or audio_context.ndim!=3 or context.shape[-1]!=4096 or audio_context.shape[-1]!=4096:
            raise ValueError('Prism requires UMT5-XXL conditioning with 4096 embedding dimensions; use CLIPLoader type=wan.')
        # Preserve native conditioning weights, scheduling and all other metadata.
        options = metadata.copy()
        options.update(prism_reference=reference,prism_audio_context=audio_context,prism_fps=fps)
        result.append([context,options])
    return result

class PrismPrepareAV:
    @classmethod
    def INPUT_TYPES(cls):
        """Declare image-to-video inputs and optional independent audio conditioning."""
        return {'required': {
            'positive': ('CONDITIONING',), 'negative': ('CONDITIONING',),
            'video_vae': ('VAE',), 'image': ('IMAGE',),
            'width': ('INT', {'default':848,'min':16,'max':4096,'step':16}),
            'height': ('INT', {'default':480,'min':16,'max':4096,'step':16}),
            'frames': ('INT', {'default':81,'min':1,'max':1025,'step':4}),
            'fps': ('FLOAT', {'default':24.0,'min':1.0,'max':120.0}),
            'reference_resize': (['official_lanczos','legacy_center_crop'], {'default':'official_lanczos'}),
        }, 'optional': {'audio_positive': ('CONDITIONING',)}}
    RETURN_TYPES = ('CONDITIONING','CONDITIONING','LATENT')
    RETURN_NAMES = ('positive','negative','av_latent')
    FUNCTION = 'prepare'
    CATEGORY = 'model/conditioning/prism'

    def prepare(self,positive,negative,video_vae,image,width,height,frames,fps,reference_resize,audio_positive=None):
        """Return joint AV conditioning and a latent with the original audio sample count."""
        reference,latent = prepare_av_reference(video_vae,image,width,height,frames,fps,reference_resize)
        audio_positive = positive if audio_positive is None else audio_positive
        return (attach_av_conditions(positive,audio_positive,reference,fps),
                attach_av_conditions(negative,negative,reference,fps),latent)

NODE_CLASS_MAPPINGS = {'PrismPrepareAV':PrismPrepareAV, 'PrismAttention':PrismAttention}
NODE_DISPLAY_NAME_MAPPINGS = {'PrismPrepareAV':'Prism Prepare AV', 'PrismAttention':'Prism Sparse Attention Settings'}
