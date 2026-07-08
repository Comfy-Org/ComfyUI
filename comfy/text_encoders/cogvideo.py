import comfy.text_encoders.sd3_clip
from comfy import sd1_clip


class CogVideoXT5Tokenizer(comfy.text_encoders.sd3_clip.T5XXLTokenizer):
    """Inner T5 tokenizer for CogVideoX.

    CogVideoX was trained with T5 embeddings padded to 226 tokens (not 77 like SD3).
    Used both directly by supported_models.CogVideoX_T2V.clip_target (paired with
    the raw T5XXLModel) and by the CogVideoXTokenizer outer wrapper below.
    """
    def __init__(self, embedding_directory=None, tokenizer_data={}):
        super().__init__(embedding_directory=embedding_directory, tokenizer_data=tokenizer_data, min_length=226)


class CogVideoXTokenizer(sd1_clip.SD1Tokenizer):
    """Outer tokenizer wrapper for CLIPLoader (type="cogvideox")."""
    def __init__(self, embedding_directory=None, tokenizer_data={}):
        super().__init__(embedding_directory=embedding_directory, tokenizer_data=tokenizer_data,
                         clip_name="t5xxl", tokenizer=CogVideoXT5Tokenizer)


class CogVideoXT5XXL(sd1_clip.SD1ClipModel):
    """Outer T5XXL model wrapper for CLIPLoader (type="cogvideox").

    Wraps the raw T5XXL model in the SD1ClipModel interface so that CLIP.__init__
    (which reads self.dtypes) works correctly. The inner model is the standard
    sd3_clip.T5XXLModel (no attention_mask change needed for CogVideoX).
    """
    def __init__(self, device="cpu", dtype=None, model_options={}):
        super().__init__(device=device, dtype=dtype, name="t5xxl",
                         clip_model=comfy.text_encoders.sd3_clip.T5XXLModel,
                         model_options=model_options)


def cogvideo_te(dtype_t5=None, t5_quantization_metadata=None):
    """Factory that returns a CogVideoXT5XXL class configured with the detected
    T5 dtype and optional quantization metadata, for use in load_text_encoder_state_dicts.
    """
    class CogVideoXTEModel_(CogVideoXT5XXL):
        def __init__(self, device="cpu", dtype=None, model_options={}):
            if t5_quantization_metadata is not None:
                model_options = model_options.copy()
                model_options["t5xxl_quantization_metadata"] = t5_quantization_metadata
            if dtype_t5 is not None:
                dtype = dtype_t5
            super().__init__(device=device, dtype=dtype, model_options=model_options)
    return CogVideoXTEModel_
