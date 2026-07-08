import json
import comfy.memory_management
import comfy.supported_models
import comfy.supported_models_base
import comfy.utils
import math
import logging
import torch

def count_blocks(state_dict_keys, prefix_string):
    count = 0
    while True:
        c = False
        for k in state_dict_keys:
            if k.startswith(prefix_string.format(count)):
                c = True
                break
        if c == False:
            break
        count += 1
    return count

def any_suffix_in(keys, prefix, main, suffix_list=[]):
    for x in suffix_list:
        if "{}{}{}".format(prefix, main, x) in keys:
            return True
    return False

def calculate_transformer_depth(prefix, state_dict_keys, state_dict):
    context_dim = None
    use_linear_in_transformer = False

    transformer_prefix = prefix + "1.transformer_blocks."
    transformer_keys = sorted(list(filter(lambda a: a.startswith(transformer_prefix), state_dict_keys)))
    if len(transformer_keys) > 0:
        last_transformer_depth = count_blocks(state_dict_keys, transformer_prefix + '{}')
        context_dim = state_dict['{}0.attn2.to_k.weight'.format(transformer_prefix)].shape[1]
        use_linear_in_transformer = len(state_dict['{}1.proj_in.weight'.format(prefix)].shape) == 2
        time_stack = '{}1.time_stack.0.attn1.to_q.weight'.format(prefix) in state_dict or '{}1.time_mix_blocks.0.attn1.to_q.weight'.format(prefix) in state_dict
        time_stack_cross = '{}1.time_stack.0.attn2.to_q.weight'.format(prefix) in state_dict or '{}1.time_mix_blocks.0.attn2.to_q.weight'.format(prefix) in state_dict
        return last_transformer_depth, context_dim, use_linear_in_transformer, time_stack, time_stack_cross
    return None

def detect_unet_config(state_dict, key_prefix, metadata=None):
    state_dict_keys = list(state_dict.keys())

    if '{}joint_blocks.0.context_block.attn.qkv.weight'.format(key_prefix) in state_dict_keys: #mmdit model
        unet_config = {}
        unet_config["in_channels"] = state_dict['{}x_embedder.proj.weight'.format(key_prefix)].shape[1]
        patch_size = state_dict['{}x_embedder.proj.weight'.format(key_prefix)].shape[2]
        unet_config["patch_size"] = patch_size
        final_layer = '{}final_layer.linear.weight'.format(key_prefix)
        if final_layer in state_dict:
            unet_config["out_channels"] = state_dict[final_layer].shape[0] // (patch_size * patch_size)

        unet_config["depth"] = state_dict['{}x_embedder.proj.weight'.format(key_prefix)].shape[0] // 64
        unet_config["input_size"] = None
        y_key = '{}y_embedder.mlp.0.weight'.format(key_prefix)
        if y_key in state_dict_keys:
            unet_config["adm_in_channels"] = state_dict[y_key].shape[1]

        context_key = '{}context_embedder.weight'.format(key_prefix)
        if context_key in state_dict_keys:
            in_features = state_dict[context_key].shape[1]
            out_features = state_dict[context_key].shape[0]
            unet_config["context_embedder_config"] = {"target": "torch.nn.Linear", "params": {"in_features": in_features, "out_features": out_features}}
        num_patches_key = '{}pos_embed'.format(key_prefix)
        if num_patches_key in state_dict_keys:
            num_patches = state_dict[num_patches_key].shape[1]
            unet_config["num_patches"] = num_patches
            unet_config["pos_embed_max_size"] = round(math.sqrt(num_patches))

        rms_qk = '{}joint_blocks.0.context_block.attn.ln_q.weight'.format(key_prefix)
        if rms_qk in state_dict_keys:
            unet_config["qk_norm"] = "rms"

        unet_config["pos_embed_scaling_factor"] = None #unused for inference
        context_processor = '{}context_processor.layers.0.attn.qkv.weight'.format(key_prefix)
        if context_processor in state_dict_keys:
            unet_config["context_processor_layers"] = count_blocks(state_dict_keys, '{}context_processor.layers.'.format(key_prefix) + '{}.')
        unet_config["x_block_self_attn_layers"] = []
        for key in state_dict_keys:
            if key.startswith('{}joint_blocks.'.format(key_prefix)) and key.endswith('.x_block.attn2.qkv.weight'):
                layer = key[len('{}joint_blocks.'.format(key_prefix)):-len('.x_block.attn2.qkv.weight')]
                unet_config["x_block_self_attn_layers"].append(int(layer))
        return unet_config

    if '{}clf.1.weight'.format(key_prefix) in state_dict_keys: #stable cascade
        unet_config = {}
        text_mapper_name = '{}clip_txt_mapper.weight'.format(key_prefix)
        if text_mapper_name in state_dict_keys:
            unet_config['stable_cascade_stage'] = 'c'
            w = state_dict[text_mapper_name]
            if w.shape[0] == 1536: #stage c lite
                unet_config['c_cond'] = 1536
                unet_config['c_hidden'] = [1536, 1536]
                unet_config['nhead'] = [24, 24]
                unet_config['blocks'] = [[4, 12], [12, 4]]
            elif w.shape[0] == 2048: #stage c full
                unet_config['c_cond'] = 2048
        elif '{}clip_mapper.weight'.format(key_prefix) in state_dict_keys:
            unet_config['stable_cascade_stage'] = 'b'
            w = state_dict['{}down_blocks.1.0.channelwise.0.weight'.format(key_prefix)]
            if w.shape[-1] == 640:
                unet_config['c_hidden'] = [320, 640, 1280, 1280]
                unet_config['nhead'] = [-1, -1, 20, 20]
                unet_config['blocks'] = [[2, 6, 28, 6], [6, 28, 6, 2]]
                unet_config['block_repeat'] = [[1, 1, 1, 1], [3, 3, 2, 2]]
            elif w.shape[-1] == 576: #stage b lite
                unet_config['c_hidden'] = [320, 576, 1152, 1152]
                unet_config['nhead'] = [-1, 9, 18, 18]
                unet_config['blocks'] = [[2, 4, 14, 4], [4, 14, 4, 2]]
                unet_config['block_repeat'] = [[1, 1, 1, 1], [2, 2, 2, 2]]
        return unet_config

    if '{}transformer.rotary_pos_emb.inv_freq'.format(key_prefix) in state_dict_keys: #stable audio dit
        unet_config = {}
        unet_config["audio_model"] = "dit1.0"
        unet_config["global_cond_dim"] = state_dict['{}to_global_embed.0.weight'.format(key_prefix)].shape[1]
        cond_embed = state_dict['{}to_cond_embed.0.weight'.format(key_prefix)]
        unet_config["project_cond_tokens"] = cond_embed.shape[0] != cond_embed.shape[1]
        unet_config["embed_dim"] = state_dict['{}to_timestep_embed.0.weight'.format(key_prefix)].shape[0]
        mem_tokens = state_dict.get('{}transformer.memory_tokens'.format(key_prefix), None)
        to_qkv = state_dict.get('{}transformer.layers.0.self_attn.to_qkv.weight'.format(key_prefix), None)
        differential = False
        if to_qkv is not None:
            if to_qkv.shape[0] == to_qkv.shape[1] * 5:
                differential = True
        if mem_tokens is not None:
            unet_config["num_memory_tokens"] = mem_tokens.shape[0]
        if '{}transformer.layers.0.self_attn.q_norm.weight'.format(key_prefix) in state_dict:
            unet_config["attn_kwargs"] = {"qk_norm": "ln", "feat_scale": True}
        rms_norm = state_dict.get('{}transformer.layers.0.self_attn.q_norm.gamma'.format(key_prefix), None)
        if rms_norm is not None:
            unet_config["attn_kwargs"] = {"qk_norm": "rms", "differential": differential}
            unet_config["norm_type"] = "rms_norm"
            unet_config["num_heads"] = unet_config["embed_dim"] // rms_norm.shape[0]

        if '{}timestep_features.weight'.format(key_prefix) in state_dict:
            unet_config["timestep_features_type"] = "learned"
        else:
            unet_config["timestep_features_type"] = "expo"

        io_channels = state_dict['{}postprocess_conv.weight'.format(key_prefix)].shape[0]
        unet_config["io_channels"] = io_channels
        unet_config["input_concat_dim"] = state_dict['{}transformer.project_in.weight'.format(key_prefix)].shape[1] - io_channels

        local_add_cond = state_dict.get('{}transformer.layers.0.to_local_embed.0.weight'.format(key_prefix), None)
        if local_add_cond is not None:
            unet_config["local_add_cond_dim"] = local_add_cond.shape[1]

        global_cond_embed = state_dict.get('{}transformer.global_cond_embedder.0.weight'.format(key_prefix), None)
        if global_cond_embed is not None:
            unet_config["global_cond_shared_embed"] = True
            unet_config["global_cond_type"] = "adaLN"

        unet_config["depth"] = count_blocks(state_dict_keys, '{}transformer.layers.'.format(key_prefix) + '{}.')
        return unet_config

    if '{}double_layers.0.attn.w1q.weight'.format(key_prefix) in state_dict_keys: #aura flow dit
        unet_config = {}
        unet_config["max_seq"] = state_dict['{}positional_encoding'.format(key_prefix)].shape[1]
        unet_config["cond_seq_dim"] = state_dict['{}cond_seq_linear.weight'.format(key_prefix)].shape[1]
        double_layers = count_blocks(state_dict_keys, '{}double_layers.'.format(key_prefix) + '{}.')
        single_layers = count_blocks(state_dict_keys, '{}single_layers.'.format(key_prefix) + '{}.')
        unet_config["n_double_layers"] = double_layers
        unet_config["n_layers"] = double_layers + single_layers
        return unet_config

    if '{}mlp_t5.0.weight'.format(key_prefix) in state_dict_keys: #Hunyuan DiT
        unet_config = {}
        unet_config["image_model"] = "hydit"
        unet_config["depth"] = count_blocks(state_dict_keys, '{}blocks.'.format(key_prefix) + '{}.')
        unet_config["hidden_size"] = state_dict['{}x_embedder.proj.weight'.format(key_prefix)].shape[0]
        if unet_config["hidden_size"] == 1408 and unet_config["depth"] == 40: #DiT-g/2
            unet_config["mlp_ratio"] = 4.3637
        if state_dict['{}extra_embedder.0.weight'.format(key_prefix)].shape[1] == 3968:
            unet_config["size_cond"] = True
            unet_config["use_style_cond"] = True
            unet_config["image_model"] = "hydit1"
        return unet_config

    if '{}txt_in.individual_token_refiner.blocks.0.norm1.weight'.format(key_prefix) in state_dict_keys: #Hunyuan Video
        dit_config = {}
        in_w = state_dict['{}img_in.proj.weight'.format(key_prefix)]
        out_w = state_dict['{}final_layer.linear.weight'.format(key_prefix)]
        dit_config["image_model"] = "hunyuan_video"
        dit_config["in_channels"] = in_w.shape[1] #SkyReels img2video has 32 input channels
        dit_config["patch_size"] = list(in_w.shape[2:])
        dit_config["out_channels"] = out_w.shape[0] // math.prod(dit_config["patch_size"])
        if any(s.startswith('{}vector_in.'.format(key_prefix)) for s in state_dict_keys):
            dit_config["vec_in_dim"] = 768
        else:
            dit_config["vec_in_dim"] = None

        if len(dit_config["patch_size"]) == 2:
            dit_config["axes_dim"] = [64, 64]
        else:
            dit_config["axes_dim"] = [16, 56, 56]

        if any(s.startswith('{}time_r_in.'.format(key_prefix)) for s in state_dict_keys):
            dit_config["meanflow"] = True
        else:
            dit_config["meanflow"] = False

        dit_config["context_in_dim"] = state_dict['{}txt_in.input_embedder.weight'.format(key_prefix)].shape[1]
        dit_config["hidden_size"] = in_w.shape[0]
        dit_config["mlp_ratio"] = 4.0
        dit_config["num_heads"] = in_w.shape[0] // 128
        dit_config["depth"] = count_blocks(state_dict_keys, '{}double_blocks.'.format(key_prefix) + '{}.')
        dit_config["depth_single_blocks"] = count_blocks(state_dict_keys, '{}single_blocks.'.format(key_prefix) + '{}.')
        dit_config["theta"] = 256
        dit_config["qkv_bias"] = True
        if '{}byt5_in.fc1.weight'.format(key_prefix) in state_dict:
            dit_config["byt5"] = True
        else:
            dit_config["byt5"] = False

        guidance_keys = list(filter(lambda a: a.startswith("{}guidance_in.".format(key_prefix)), state_dict_keys))
        dit_config["guidance_embed"] = len(guidance_keys) > 0

        # HunyuanVideo 1.5
        if '{}cond_type_embedding.weight'.format(key_prefix) in state_dict_keys:
            dit_config["use_cond_type_embedding"] = True
        else:
            dit_config["use_cond_type_embedding"] = False
        if '{}vision_in.proj.0.weight'.format(key_prefix) in state_dict_keys:
            dit_config["vision_in_dim"] = state_dict['{}vision_in.proj.0.weight'.format(key_prefix)].shape[0]
            dit_config["meanflow_sum"] = True
        else:
            dit_config["vision_in_dim"] = None
            dit_config["meanflow_sum"] = False
        return dit_config

    if any_suffix_in(state_dict_keys, key_prefix, 'double_blocks.0.img_attn.norm.key_norm.', ["weight", "scale"]) and ('{}img_in.weight'.format(key_prefix) in state_dict_keys or any_suffix_in(state_dict_keys, key_prefix, 'distilled_guidance_layer.norms.0.', ["weight", "scale"])): #Flux, Chroma or Chroma Radiance (has no img_in.weight)
        dit_config = {}
        if '{}double_stream_modulation_img.lin.weight'.format(key_prefix) in state_dict_keys:
            dit_config["image_model"] = "flux2"
            dit_config["axes_dim"] = [32, 32, 32, 32]
            dit_config["num_heads"] = 48
            dit_config["mlp_ratio"] = 3.0
            dit_config["theta"] = 2000
            dit_config["out_channels"] = 128
            dit_config["global_modulation"] = True
            dit_config["mlp_silu_act"] = True
            dit_config["qkv_bias"] = False
            dit_config["ops_bias"] = False
            dit_config["default_ref_method"] = "index"
            dit_config["ref_index_scale"] = 10.0
            dit_config["txt_ids_dims"] = [3]
            patch_size = 1
        else:
            dit_config["image_model"] = "flux"
            dit_config["axes_dim"] = [16, 56, 56]
            dit_config["num_heads"] = 24
            dit_config["mlp_ratio"] = 4.0
            dit_config["theta"] = 10000
            dit_config["out_channels"] = 16
            dit_config["qkv_bias"] = True
            dit_config["txt_ids_dims"] = []
            patch_size = 2

        dit_config["in_channels"] = 16
        dit_config["hidden_size"] = 3072
        dit_config["context_in_dim"] = 4096

        dit_config["patch_size"] = patch_size
        in_key = "{}img_in.weight".format(key_prefix)
        if in_key in state_dict_keys:
            w = state_dict[in_key]
            dit_config["in_channels"] = w.shape[1] // (patch_size * patch_size)
            dit_config["hidden_size"] = w.shape[0]

        txt_in_key = "{}txt_in.weight".format(key_prefix)
        if txt_in_key in state_dict_keys:
            w = state_dict[txt_in_key]
            dit_config["context_in_dim"] = w.shape[1]
            dit_config["hidden_size"] = w.shape[0]

        vec_in_key = '{}vector_in.in_layer.weight'.format(key_prefix)
        if vec_in_key in state_dict_keys:
            dit_config["vec_in_dim"] = state_dict[vec_in_key].shape[1]
        else:
            dit_config["vec_in_dim"] = None

        dit_config["num_heads"] = dit_config["hidden_size"] // sum(dit_config["axes_dim"])

        dit_config["depth"] = count_blocks(state_dict_keys, '{}double_blocks.'.format(key_prefix) + '{}.')
        dit_config["depth_single_blocks"] = count_blocks(state_dict_keys, '{}single_blocks.'.format(key_prefix) + '{}.')

        if any_suffix_in(state_dict_keys, key_prefix, 'distilled_guidance_layer.0.norms.0.', ["weight", "scale"]) or any_suffix_in(state_dict_keys, key_prefix, 'distilled_guidance_layer.norms.0.', ["weight", "scale"]): #Chroma
            dit_config["image_model"] = "chroma"
            dit_config["in_channels"] = 64
            dit_config["out_channels"] = 64
            dit_config["in_dim"] = 64
            dit_config["out_dim"] = 3072
            dit_config["hidden_dim"] = 5120
            dit_config["n_layers"] = 5

            if any_suffix_in(state_dict_keys, key_prefix, 'nerf_blocks.0.norm.', ["weight", "scale"]): #Chroma Radiance
                dit_config["image_model"] = "chroma_radiance"
                dit_config["in_channels"] = 3
                dit_config["out_channels"] = 3
                dit_config["patch_size"] = state_dict.get('{}img_in_patch.weight'.format(key_prefix)).size(dim=-1)
                dit_config["nerf_hidden_size"] = 64
                dit_config["nerf_mlp_ratio"] = 4
                dit_config["nerf_depth"] = 4
                dit_config["nerf_max_freqs"] = 8
                dit_config["nerf_tile_size"] = 512
                dit_config["nerf_final_head_type"] = "conv" if any_suffix_in(state_dict_keys, key_prefix, 'nerf_final_layer_conv.norm.', ["weight", "scale"]) else "linear"
                dit_config["nerf_embedder_dtype"] = torch.float32
                if "{}__x0__".format(key_prefix) in state_dict_keys: # x0 pred
                    dit_config["use_x0"] = True
                else:
                    dit_config["use_x0"] = False
                if "{}__sequential__".format(key_prefix) in state_dict_keys: # sequential txt_ids
                    dit_config["use_sequential_txt_ids"] = True
                else:
                    dit_config["use_sequential_txt_ids"] = False
        else:
            dit_config["guidance_embed"] = "{}guidance_in.in_layer.weight".format(key_prefix) in state_dict_keys
            dit_config["yak_mlp"] = '{}double_blocks.0.img_mlp.gate_proj.weight'.format(key_prefix) in state_dict_keys
            dit_config["txt_norm"] = any_suffix_in(state_dict_keys, key_prefix, 'txt_norm.', ["weight", "scale"])
            if dit_config["yak_mlp"] and dit_config["txt_norm"]:  # Ovis model
                dit_config["txt_ids_dims"] = [1, 2]
            if dit_config.get("context_in_dim") == 3584 and dit_config["vec_in_dim"] is None:  # LongCat-Image
                dit_config["txt_ids_dims"] = [1, 2]

        return dit_config

    if '{}t5_yproj.weight'.format(key_prefix) in state_dict_keys: #Genmo mochi preview
        dit_config = {}
        dit_config["image_model"] = "mochi_preview"
        dit_config["depth"] = 48
        dit_config["patch_size"] = 2
        dit_config["num_heads"] = 24
        dit_config["hidden_size_x"] = 3072
        dit_config["hidden_size_y"] = 1536
        dit_config["mlp_ratio_x"] = 4.0
        dit_config["mlp_ratio_y"] = 4.0
        dit_config["learn_sigma"] = False
        dit_config["in_channels"] = 12
        dit_config["qk_norm"] = True
        dit_config["qkv_bias"] = False
        dit_config["out_bias"] = True
        dit_config["attn_drop"] = 0.0
        dit_config["patch_embed_bias"] = True
        dit_config["posenc_preserve_area"] = True
        dit_config["timestep_mlp_bias"] = True
        dit_config["attend_to_padding"] = False
        dit_config["timestep_scale"] = 1000.0
        dit_config["use_t5"] = True
        dit_config["t5_feat_dim"] = 4096
        dit_config["t5_token_length"] = 256
        dit_config["rope_theta"] = 10000.0
        return dit_config

    if '{}adaln_single.emb.timestep_embedder.linear_1.bias'.format(key_prefix) in state_dict_keys and '{}pos_embed.proj.bias'.format(key_prefix) in state_dict_keys:
        # PixArt diffusers
        return None

    if '{}adaln_single.emb.timestep_embedder.linear_1.bias'.format(key_prefix) in state_dict_keys: #Lightricks ltxv
        dit_config = {}
        dit_config["image_model"] = "ltxav" if f'{key_prefix}audio_adaln_single.linear.weight' in state_dict_keys else "ltxv"
        dit_config["num_layers"] = count_blocks(state_dict_keys, '{}transformer_blocks.'.format(key_prefix) + '{}.')
        shape = state_dict['{}transformer_blocks.0.attn2.to_k.weight'.format(key_prefix)].shape
        dit_config["attention_head_dim"] = shape[0] // 32
        dit_config["cross_attention_dim"] = shape[1]
        if metadata is not None and "config" in metadata:
            dit_config.update(json.loads(metadata["config"]).get("transformer", {}))
        return dit_config

    if '{}genre_embedder.weight'.format(key_prefix) in state_dict_keys: #ACE-Step model
        dit_config = {}
        dit_config["audio_model"] = "ace"
        dit_config["attention_head_dim"] = 128
        dit_config["in_channels"] = 8
        dit_config["inner_dim"] = 2560
        dit_config["max_height"] = 16
        dit_config["max_position"] = 32768
        dit_config["max_width"] = 32768
        dit_config["mlp_ratio"] = 2.5
        dit_config["num_attention_heads"] = 20
        dit_config["num_layers"] = 24
        dit_config["out_channels"] = 8
        dit_config["patch_size"] = [16, 1]
        dit_config["rope_theta"] = 1000000.0
        dit_config["speaker_embedding_dim"] = 512
        dit_config["text_embedding_dim"] = 768

        dit_config["ssl_encoder_depths"] = [8, 8]
        dit_config["ssl_latent_dims"] = [1024, 768]
        dit_config["ssl_names"] = ["mert", "m-hubert"]
        dit_config["lyric_encoder_vocab_size"] = 6693
        dit_config["lyric_hidden_size"] = 1024
        return dit_config

    if '{}t_block.1.weight'.format(key_prefix) in state_dict_keys: # PixArt
        patch_size = 2
        dit_config = {}
        dit_config["num_heads"] = 16
        dit_config["patch_size"] = patch_size
        dit_config["hidden_size"] = 1152
        dit_config["in_channels"] = 4
        dit_config["depth"] = count_blocks(state_dict_keys, '{}blocks.'.format(key_prefix) + '{}.')

        y_key = "{}y_embedder.y_embedding".format(key_prefix)
        if y_key in state_dict_keys:
            dit_config["model_max_length"] = state_dict[y_key].shape[0]

        pe_key = "{}pos_embed".format(key_prefix)
        if pe_key in state_dict_keys:
            dit_config["input_size"] = int(math.sqrt(state_dict[pe_key].shape[1])) * patch_size
            dit_config["pe_interpolation"] = dit_config["input_size"] // (512//8) # guess

        ar_key = "{}ar_embedder.mlp.0.weight".format(key_prefix)
        if ar_key in state_dict_keys:
            dit_config["image_model"] = "pixart_alpha"
            dit_config["micro_condition"] = True
        else:
            dit_config["image_model"] = "pixart_sigma"
            dit_config["micro_condition"] = False
        return dit_config

    if '{}blocks.block0.blocks.0.block.attn.to_q.0.weight'.format(key_prefix) in state_dict_keys:  # Cosmos
        dit_config = {}
        dit_config["image_model"] = "cosmos"
        dit_config["max_img_h"] = 240
        dit_config["max_img_w"] = 240
        dit_config["max_frames"] = 128
        concat_padding_mask = True
        dit_config["in_channels"] = (state_dict['{}x_embedder.proj.1.weight'.format(key_prefix)].shape[1] // 4) - int(concat_padding_mask)
        dit_config["out_channels"] = 16
        dit_config["patch_spatial"] = 2
        dit_config["patch_temporal"] = 1
        dit_config["model_channels"] = state_dict['{}blocks.block0.blocks.0.block.attn.to_q.0.weight'.format(key_prefix)].shape[0]
        dit_config["block_config"] = "FA-CA-MLP"
        dit_config["concat_padding_mask"] = concat_padding_mask
        dit_config["pos_emb_cls"] = "rope3d"
        dit_config["pos_emb_learnable"] = False
        dit_config["pos_emb_interpolation"] = "crop"
        dit_config["block_x_format"] = "THWBD"
        dit_config["affline_emb_norm"] = True
        dit_config["use_adaln_lora"] = True
        dit_config["adaln_lora_dim"] = 256

        if dit_config["model_channels"] == 4096:
            # 7B
            dit_config["num_blocks"] = 28
            dit_config["num_heads"] = 32
            dit_config["extra_per_block_abs_pos_emb"] = True
            dit_config["rope_h_extrapolation_ratio"] = 1.0
            dit_config["rope_w_extrapolation_ratio"] = 1.0
            dit_config["rope_t_extrapolation_ratio"] = 2.0
            dit_config["extra_per_block_abs_pos_emb_type"] = "learnable"
        else:  # 5120
            # 14B
            dit_config["num_blocks"] = 36
            dit_config["num_heads"] = 40
            dit_config["extra_per_block_abs_pos_emb"] = True
            dit_config["rope_h_extrapolation_ratio"] = 2.0
            dit_config["rope_w_extrapolation_ratio"] = 2.0
            dit_config["rope_t_extrapolation_ratio"] = 2.0
            dit_config["extra_h_extrapolation_ratio"] = 2.0
            dit_config["extra_w_extrapolation_ratio"] = 2.0
            dit_config["extra_t_extrapolation_ratio"] = 2.0
            dit_config["extra_per_block_abs_pos_emb_type"] = "learnable"
        return dit_config

    # PiD (Pixel Diffusion Decoder). Must check BEFORE plain PixelDiT_T2I.
    _lq_w_key = '{}lq_proj.latent_proj.0.weight'.format(key_prefix)
    if _lq_w_key in state_dict_keys:
        in_ch = int(state_dict[_lq_w_key].shape[1])
        _gate_prefix = '{}lq_proj.gate_modules.'.format(key_prefix)
        num_gates = len({k[len(_gate_prefix):].split('.')[0]
                         for k in state_dict_keys if k.startswith(_gate_prefix)})
        dit_config = {"image_model": "pid",
                      "lq_latent_channels": in_ch,
                      "latent_spatial_down_factor": 16 if in_ch >= 64 else 8}
        if num_gates > 0:
            dit_config["lq_interval"] = (14 + num_gates - 1) // num_gates
        return dit_config

    if '{}core.pixel_embedder.proj.weight'.format(key_prefix) in state_dict_keys:  # PixelDiT T2I
        return {"image_model": "pixeldit_t2i"}

    if '{}cap_embedder.1.weight'.format(key_prefix) in state_dict_keys and '{}noise_refiner.0.attention.k_norm.weight'.format(key_prefix) in state_dict_keys:  # Lumina 2
        dit_config = {}
        dit_config["image_model"] = "lumina2"
        dit_config["patch_size"] = 2
        dit_config["in_channels"] = 16
        w = state_dict['{}cap_embedder.1.weight'.format(key_prefix)]
        dit_config["dim"] = w.shape[0]
        dit_config["cap_feat_dim"] = w.shape[1]
        dit_config["n_layers"] = count_blocks(state_dict_keys, '{}layers.'.format(key_prefix) + '{}.')
        dit_config["qk_norm"] = True

        if dit_config["dim"] == 2304: # Original Lumina 2
            dit_config["n_heads"] = 24
            dit_config["n_kv_heads"] = 8
            dit_config["axes_dims"] = [32, 32, 32]
            dit_config["axes_lens"] = [300, 512, 512]
            dit_config["rope_theta"] = 10000.0
            dit_config["ffn_dim_multiplier"] = 4.0
            ctd_weight = state_dict.get('{}clip_text_pooled_proj.0.weight'.format(key_prefix), None)
            if ctd_weight is not None:  # NewBie
                dit_config["clip_text_dim"] = ctd_weight.shape[0]
                # NewBie also sets axes_lens = [1024, 512, 512] but it's not used in ComfyUI
        elif dit_config["dim"] == 3840:  # Z image
            dit_config["n_heads"] = 30
            dit_config["n_kv_heads"] = 30
            dit_config["axes_dims"] = [32, 48, 48]
            dit_config["axes_lens"] = [1536, 512, 512]
            dit_config["rope_theta"] = 256.0
            dit_config["ffn_dim_multiplier"] = (8.0 / 3.0)
            dit_config["z_image_modulation"] = True
            dit_config["time_scale"] = 1000.0
            try:
                dit_config["allow_fp16"] = torch.std(state_dict['{}layers.{}.ffn_norm1.weight'.format(key_prefix, dit_config["n_layers"] - 2)], unbiased=False).item() < 0.42
            except Exception:
                pass
            if '{}cap_pad_token'.format(key_prefix) in state_dict_keys:
                dit_config["pad_tokens_multiple"] = 32
            sig_weight = state_dict.get('{}siglip_embedder.0.weight'.format(key_prefix), None)
            if sig_weight is not None:
                dit_config["siglip_feat_dim"] = sig_weight.shape[0]

            dec_cond_key = '{}dec_net.cond_embed.weight'.format(key_prefix)
            if dec_cond_key in state_dict_keys:  # pixel-space variant
                dit_config["image_model"] = "zimage_pixel"
                # patch_size and in_channels are derived from x_embedder:
                #   x_embedder: Linear(patch_size * patch_size * in_channels, dim)
                # The decoder also receives the full flat patch, so decoder_in_channels = x_embedder input dim.
                x_emb_in = state_dict['{}x_embedder.weight'.format(key_prefix)].shape[1]
                dec_out = state_dict['{}dec_net.final_layer.linear.weight'.format(key_prefix)].shape[0]
                # patch_size: infer from decoder final layer output matching x_embedder input
                # in_channels: infer from dec_net input_embedder (in_features = dec_in_ch + max_freqs^2)
                embedder_w = state_dict['{}dec_net.input_embedder.embedder.0.weight'.format(key_prefix)]
                dec_in_ch = dec_out  # decoder in == decoder out (same pixel space)
                dit_config["patch_size"] = round((x_emb_in / 3) ** 0.5)  # assume RGB (in_channels=3)
                dit_config["in_channels"] = 3
                dit_config["decoder_in_channels"] = dec_in_ch
                dit_config["decoder_hidden_size"] = state_dict[dec_cond_key].shape[0]
                dit_config["decoder_num_res_blocks"] = count_blocks(
                    state_dict_keys, '{}dec_net.res_blocks.'.format(key_prefix) + '{}.'
                )
                dit_config["decoder_max_freqs"] = int((embedder_w.shape[1] - dec_in_ch) ** 0.5)
                if '{}__x0__'.format(key_prefix) in state_dict_keys:
                    dit_config["use_x0"] = True

        return dit_config

    if '{}blocks.0.norm1.linear.weight'.format(key_prefix) in state_dict_keys:  # CogVideoX
        dit_config = {}
        dit_config["image_model"] = "cogvideox"

        # Extract config from weight shapes
        norm1_weight = state_dict['{}blocks.0.norm1.linear.weight'.format(key_prefix)]
        time_embed_dim = norm1_weight.shape[1]
        dim = norm1_weight.shape[0] // 6

        dit_config["num_attention_heads"] = dim // 64
        dit_config["attention_head_dim"] = 64
        dit_config["time_embed_dim"] = time_embed_dim
        dit_config["num_layers"] = count_blocks(state_dict_keys, '{}blocks.'.format(key_prefix) + '{}.')

        # Detect in_channels from patch_embed
        patch_proj_key = '{}patch_embed.proj.weight'.format(key_prefix)
        if patch_proj_key in state_dict_keys:
            w = state_dict[patch_proj_key]
            if w.ndim == 4:
                # Conv2d: [out, in, kh, kw] — CogVideoX 1.0
                dit_config["in_channels"] = w.shape[1]
                dit_config["patch_size"] = w.shape[2]
            elif w.ndim == 2:
                # Linear: [out, in_channels * patch_size * patch_size * patch_size_t] — CogVideoX 1.5
                dit_config["patch_size"] = 2
                dit_config["patch_size_t"] = 2
                dit_config["in_channels"] = w.shape[1] // (2 * 2 * 2)  # 256 // 8 = 32

        text_proj_key = '{}patch_embed.text_proj.weight'.format(key_prefix)
        if text_proj_key in state_dict_keys:
            dit_config["text_embed_dim"] = state_dict[text_proj_key].shape[1]

        # Detect OFS embedding
        ofs_key = '{}ofs_embedding_linear_1.weight'.format(key_prefix)
        if ofs_key in state_dict_keys:
            dit_config["ofs_embed_dim"] = state_dict[ofs_key].shape[1]

        # Detect positional embedding type
        pos_key = '{}patch_embed.pos_embedding'.format(key_prefix)
        if pos_key in state_dict_keys:
            dit_config["use_learned_positional_embeddings"] = True
            dit_config["use_rotary_positional_embeddings"] = False
        else:
            dit_config["use_learned_positional_embeddings"] = False
            dit_config["use_rotary_positional_embeddings"] = True

        return dit_config

    if '{}head.modulation'.format(key_prefix) in state_dict_keys:  # Wan 2.1
        dit_config = {}
        dit_config["image_model"] = "wan2.1"
        dim = state_dict['{}head.modulation'.format(key_prefix)].shape[-1]
        out_dim = state_dict['{}head.head.weight'.format(key_prefix)].shape[0] // 4
        dit_config["dim"] = dim
        dit_config["out_dim"] = out_dim
        dit_config["num_heads"] = dim // 128
        dit_config["ffn_dim"] = state_dict['{}blocks.0.ffn.0.weight'.format(key_prefix)].shape[0]
        dit_config["num_layers"] = count_blocks(state_dict_keys, '{}blocks.'.format(key_prefix) + '{}.')
        dit_config["patch_size"] = (1, 2, 2)
        dit_config["freq_dim"] = 256
        dit_config["window_size"] = (-1, -1)
        dit_config["qk_norm"] = True
        dit_config["cross_attn_norm"] = True
        dit_config["eps"] = 1e-6
        dit_config["in_dim"] = state_dict['{}patch_embedding.weight'.format(key_prefix)].shape[1]
        if '{}vace_patch_embedding.weight'.format(key_prefix) in state_dict_keys:
            dit_config["model_type"] = "vace"
            dit_config["vace_in_dim"] = state_dict['{}vace_patch_embedding.weight'.format(key_prefix)].shape[1]
            dit_config["vace_layers"] = count_blocks(state_dict_keys, '{}vace_blocks.'.format(key_prefix) + '{}.')
        elif '{}control_adapter.conv.weight'.format(key_prefix) in state_dict_keys:
            if '{}img_emb.proj.0.bias'.format(key_prefix) in state_dict_keys:
                dit_config["model_type"] = "camera"
            else:
                dit_config["model_type"] = "camera_2.2"
        elif '{}casual_audio_encoder.encoder.final_linear.weight'.format(key_prefix) in state_dict_keys:
            dit_config["model_type"] = "s2v"
        elif '{}audio_proj.audio_proj_glob_1.layer.bias'.format(key_prefix) in state_dict_keys:
            dit_config["model_type"] = "humo"
        elif '{}face_adapter.fuser_blocks.0.k_norm.weight'.format(key_prefix) in state_dict_keys:
            dit_config["model_type"] = "animate"
        elif '{}patch_embedding_mask.weight'.format(key_prefix) in state_dict_keys:
            dit_config["model_type"] = "scail2"
        elif '{}patch_embedding_pose.weight'.format(key_prefix) in state_dict_keys:
            dit_config["model_type"] = "scail"
        elif '{}patch_embedding_global.weight'.format(key_prefix) in state_dict_keys:
            dit_config["model_type"] = "wandancer"
        else:
            if '{}img_emb.proj.0.bias'.format(key_prefix) in state_dict_keys:
                dit_config["model_type"] = "i2v"
            else:
                dit_config["model_type"] = "t2v"
        flf_weight = state_dict.get('{}img_emb.emb_pos'.format(key_prefix))
        if flf_weight is not None:
            dit_config["flf_pos_embed_token_number"] = flf_weight.shape[1]

        ref_conv_weight = state_dict.get('{}ref_conv.weight'.format(key_prefix))
        if ref_conv_weight is not None:
            dit_config["in_dim_ref_conv"] = ref_conv_weight.shape[1]

        if metadata is not None and "config" in metadata:
            dit_config.update(json.loads(metadata["config"]).get("transformer", {}))

        return dit_config

    if '{}latent_in.weight'.format(key_prefix) in state_dict_keys:  # Hunyuan 3D
        in_shape = state_dict['{}latent_in.weight'.format(key_prefix)].shape
        dit_config = {}
        dit_config["image_model"] = "hunyuan3d2"
        dit_config["in_channels"] = in_shape[1]
        dit_config["context_in_dim"] = state_dict['{}cond_in.weight'.format(key_prefix)].shape[1]
        dit_config["hidden_size"] = in_shape[0]
        dit_config["mlp_ratio"] = 4.0
        dit_config["num_heads"] = 16
        dit_config["depth"] = count_blocks(state_dict_keys, '{}double_blocks.'.format(key_prefix) + '{}.')
        dit_config["depth_single_blocks"] = count_blocks(state_dict_keys, '{}single_blocks.'.format(key_prefix) + '{}.')
        dit_config["qkv_bias"] = True
        dit_config["guidance_embed"] = "{}guidance_in.in_layer.weight".format(key_prefix) in state_dict_keys
        return dit_config

    if f"{key_prefix}t_embedder.mlp.2.weight" in state_dict_keys and f"{key_prefix}blocks.0.attn1.k_norm.weight" in state_dict_keys:  # Hunyuan 3D 2.1
        dit_config = {}
        dit_config["image_model"] = "hunyuan3d2_1"
        dit_config["in_channels"] = state_dict[f"{key_prefix}x_embedder.weight"].shape[1]
        dit_config["context_dim"] = 1024
        dit_config["hidden_size"] = state_dict[f"{key_prefix}x_embedder.weight"].shape[0]
        dit_config["mlp_ratio"] = 4.0
        dit_config["num_heads"] = 16
        dit_config["depth"] = count_blocks(state_dict_keys, f"{key_prefix}blocks.{{}}")
        dit_config["qkv_bias"] = False
        dit_config["guidance_cond_proj_dim"] = None#f"{key_prefix}t_embedder.cond_proj.weight" in state_dict_keys
        return dit_config

    if '{}cam_out_layer.weight'.format(key_prefix) in state_dict_keys and '{}repo_layers.0.final_map.weight'.format(key_prefix) in state_dict_keys:  # TripoSplat
        return {"image_model": "triposplat"}

    if '{}t_embedder1.mlp.0.weight'.format(key_prefix) in state_dict_keys and '{}x_embedder.proj1.weight'.format(key_prefix) in state_dict_keys:  # HiDream-O1
        return {"image_model": "hidream_o1"}

    if '{}caption_projection.0.linear.weight'.format(key_prefix) in state_dict_keys:  # HiDream
        dit_config = {}
        dit_config["image_model"] = "hidream"
        dit_config["attention_head_dim"] = 128
        dit_config["axes_dims_rope"] = [64, 32, 32]
        dit_config["caption_channels"] = [4096, 4096]
        dit_config["max_resolution"] = [128, 128]
        dit_config["in_channels"] = 16
        dit_config["llama_layers"] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31]
        dit_config["num_attention_heads"] = 20
        dit_config["num_routed_experts"] = 4
        dit_config["num_activated_experts"] = 2
        dit_config["num_layers"] = 16
        dit_config["num_single_layers"] = 32
        dit_config["out_channels"] = 16
        dit_config["patch_size"] = 2
        dit_config["text_emb_dim"] = 2048
        return dit_config

    if '{}blocks.0.mlp.layer1.weight'.format(key_prefix) in state_dict_keys:  # Cosmos predict2
        dit_config = {}
        dit_config["image_model"] = "cosmos_predict2"
        if "{}llm_adapter.blocks.0.cross_attn.q_proj.weight".format(key_prefix) in state_dict_keys:
            dit_config["image_model"] = "anima"
        dit_config["max_img_h"] = 240
        dit_config["max_img_w"] = 240
        dit_config["max_frames"] = 128
        concat_padding_mask = True
        dit_config["in_channels"] = (state_dict['{}x_embedder.proj.1.weight'.format(key_prefix)].shape[1] // 4) - int(concat_padding_mask)
        dit_config["out_channels"] = 16
        dit_config["patch_spatial"] = 2
        dit_config["patch_temporal"] = 1
        dit_config["model_channels"] = state_dict['{}x_embedder.proj.1.weight'.format(key_prefix)].shape[0]
        dit_config["concat_padding_mask"] = concat_padding_mask
        dit_config["crossattn_emb_channels"] = 1024
        dit_config["pos_emb_cls"] = "rope3d"
        dit_config["pos_emb_learnable"] = True
        dit_config["pos_emb_interpolation"] = "crop"
        dit_config["min_fps"] = 1
        dit_config["max_fps"] = 30

        dit_config["use_adaln_lora"] = True
        dit_config["adaln_lora_dim"] = 256
        if dit_config["model_channels"] == 2048:
            dit_config["num_blocks"] = 28
            dit_config["num_heads"] = 16
        elif dit_config["model_channels"] == 5120:
            dit_config["num_blocks"] = 36
            dit_config["num_heads"] = 40

        if dit_config["in_channels"] == 16:
            dit_config["extra_per_block_abs_pos_emb"] = False
            dit_config["rope_h_extrapolation_ratio"] = 4.0
            dit_config["rope_w_extrapolation_ratio"] = 4.0
            dit_config["rope_t_extrapolation_ratio"] = 1.0
        elif dit_config["in_channels"] == 17: # img to video
            if dit_config["model_channels"] == 2048:
                dit_config["extra_per_block_abs_pos_emb"] = False
                dit_config["rope_h_extrapolation_ratio"] = 3.0
                dit_config["rope_w_extrapolation_ratio"] = 3.0
                dit_config["rope_t_extrapolation_ratio"] = 1.0
            elif dit_config["model_channels"] == 5120:
                dit_config["rope_h_extrapolation_ratio"] = 2.0
                dit_config["rope_w_extrapolation_ratio"] = 2.0
                dit_config["rope_t_extrapolation_ratio"] = 0.8333333333333334

        dit_config["extra_h_extrapolation_ratio"] = 1.0
        dit_config["extra_w_extrapolation_ratio"] = 1.0
        dit_config["extra_t_extrapolation_ratio"] = 1.0
        dit_config["rope_enable_fps_modulation"] = False

        return dit_config

    if '{}double_stream_layers.0.img_instruct_attn.processor.img_to_q.weight'.format(key_prefix) in state_dict_keys:  # Boogu-Image (OmniGen2 derivative + dual-stream stage)
        dit_config = {}
        dit_config["image_model"] = "boogu"
        dit_config["hidden_size"] = state_dict['{}x_embedder.weight'.format(key_prefix)].shape[0]
        dit_config["num_layers"] = count_blocks(state_dict_keys, '{}single_stream_layers.'.format(key_prefix) + '{}.')
        dit_config["num_double_stream_layers"] = count_blocks(state_dict_keys, '{}double_stream_layers.'.format(key_prefix) + '{}.')
        dit_config["num_refiner_layers"] = count_blocks(state_dict_keys, '{}noise_refiner.'.format(key_prefix) + '{}.')
        dit_config["instruction_feat_dim"] = state_dict['{}time_caption_embed.caption_embedder.0.weight'.format(key_prefix)].shape[0]
        return dit_config

    if '{}time_caption_embed.timestep_embedder.linear_1.bias'.format(key_prefix) in state_dict_keys:  # Omnigen2
        dit_config = {}
        dit_config["image_model"] = "omnigen2"
        dit_config["axes_dim_rope"] = [40, 40, 40]
        dit_config["axes_lens"] = [1024, 1664, 1664]
        dit_config["ffn_dim_multiplier"] = None
        dit_config["hidden_size"] = 2520
        dit_config["in_channels"] = 16
        dit_config["multiple_of"] = 256
        dit_config["norm_eps"] = 1e-05
        dit_config["num_attention_heads"] = 21
        dit_config["num_kv_heads"] = 7
        dit_config["num_layers"] = 32
        dit_config["num_refiner_layers"] = 2
        dit_config["out_channels"] = None
        dit_config["patch_size"] = 2
        dit_config["text_feat_dim"] = 2048
        dit_config["timestep_scale"] = 1000.0
        return dit_config

    if '{}transformer_blocks.0.attn.norm_added_q.weight'.format(key_prefix) in state_dict_keys \
            and '{}transformer_blocks.0.img_mlp.w1.weight'.format(key_prefix) in state_dict_keys:  # Lens
        img_in_w = state_dict['{}img_in.weight'.format(key_prefix)]
        proj_out_w = state_dict['{}proj_out.weight'.format(key_prefix)]
        multi_layer = '{}txt_norm.0.weight'.format(key_prefix) in state_dict_keys
        if multi_layer:
            enc_hidden_dim = state_dict['{}txt_norm.0.weight'.format(key_prefix)].shape[0]
            # Indices are TE-side; the DiT just consumes L layers in order.
            selected_layer_index = tuple(range(count_blocks(state_dict_keys, '{}txt_norm.'.format(key_prefix) + '{}.')))
        else:
            enc_hidden_dim = state_dict['{}txt_norm.weight'.format(key_prefix)].shape[0]
            selected_layer_index = (0,)

        return {
            "image_model": "lens",
            "in_channels": img_in_w.shape[1],
            "out_channels": proj_out_w.shape[0] // 4,  # patch_size ** 2 (=2² default)
            "num_layers": count_blocks(state_dict_keys, '{}transformer_blocks.'.format(key_prefix) + '{}.'),
            "num_attention_heads": img_in_w.shape[0] // 64,  # // attention_head_dim default
            "enc_hidden_dim": enc_hidden_dim,
            "multi_layer_encoder_feature": multi_layer,
            "selected_layer_index": selected_layer_index,
        }

    if '{}txt_norm.weight'.format(key_prefix) in state_dict_keys:  # Qwen Image
        dit_config = {}
        dit_config["image_model"] = "qwen_image"
        dit_config["in_channels"] = state_dict['{}img_in.weight'.format(key_prefix)].shape[1]
        dit_config["num_layers"] = count_blocks(state_dict_keys, '{}transformer_blocks.'.format(key_prefix) + '{}.')
        if "{}__index_timestep_zero__".format(key_prefix) in state_dict_keys:  # 2511
            dit_config["default_ref_method"] = "index_timestep_zero"
        if "{}time_text_embed.addition_t_embedding.weight".format(key_prefix) in state_dict_keys:  # Layered
            dit_config["use_additional_t_cond"] = True
            dit_config["default_ref_method"] = "negative_index"
        return dit_config

    if '{}embed_image_indicator.weight'.format(key_prefix) in state_dict_keys:  # Ideogram 4
        dit_config = {}
        dit_config["image_model"] = "ideogram4"
        dit_config["in_channels"] = state_dict['{}input_proj.weight'.format(key_prefix)].shape[1]
        dit_config["num_layers"] = count_blocks(state_dict_keys, '{}layers.'.format(key_prefix) + '{}.')
        return dit_config

    if '{}txtfusion.projector.weight'.format(key_prefix) in state_dict_keys:  # Krea 2 (K2)
        dit_config = {}
        dit_config["image_model"] = "krea2"
        head_dim = 128
        first_w = state_dict['{}first.weight'.format(key_prefix)]  # (features, channels*patch^2)
        dit_config["features"] = first_w.shape[0]
        dit_config["channels"] = first_w.shape[1] // (2 * 2)  # patch=2
        dit_config["patch"] = 2
        dit_config["layers"] = count_blocks(state_dict_keys, '{}blocks.'.format(key_prefix) + '{}.')
        dit_config["heads"] = state_dict['{}blocks.0.attn.wq.weight'.format(key_prefix)].shape[0] // head_dim
        dit_config["kvheads"] = state_dict['{}blocks.0.attn.wk.weight'.format(key_prefix)].shape[0] // head_dim
        dit_config["txtlayers"] = state_dict['{}txtfusion.projector.weight'.format(key_prefix)].shape[1]
        dit_config["txtdim"] = state_dict['{}txtfusion.layerwise_blocks.0.prenorm.scale'.format(key_prefix)].shape[0]
        return dit_config

    if '{}visual_transformer_blocks.0.cross_attention.key_norm.weight'.format(key_prefix) in state_dict_keys: # Kandinsky 5
        dit_config = {}
        model_dim = state_dict['{}visual_embeddings.in_layer.bias'.format(key_prefix)].shape[0]
        dit_config["model_dim"] = model_dim
        if model_dim in [4096, 2560]: # pro video and lite image
            dit_config["axes_dims"] = (32, 48, 48)
            if model_dim == 2560: # lite image
                dit_config["rope_scale_factor"] = (1.0, 1.0, 1.0)
        elif model_dim == 1792: # lite video
            dit_config["axes_dims"] = (16, 24, 24)
        dit_config["time_dim"] = state_dict['{}time_embeddings.in_layer.bias'.format(key_prefix)].shape[0]
        dit_config["image_model"] = "kandinsky5"
        dit_config["ff_dim"] = state_dict['{}visual_transformer_blocks.0.feed_forward.in_layer.weight'.format(key_prefix)].shape[0]
        dit_config["visual_embed_dim"] = state_dict['{}visual_embeddings.in_layer.weight'.format(key_prefix)].shape[1]
        dit_config["num_text_blocks"] = count_blocks(state_dict_keys, '{}text_transformer_blocks.'.format(key_prefix) + '{}.')
        dit_config["num_visual_blocks"] = count_blocks(state_dict_keys, '{}visual_transformer_blocks.'.format(key_prefix) + '{}.')
        return dit_config

    if '{}encoder.lyric_encoder.layers.0.input_layernorm.weight'.format(key_prefix) in state_dict_keys:
        dit_config = {}
        dit_config["audio_model"] = "ace1.5"
        head_dim = 128
        dit_config["hidden_size"] = state_dict['{}decoder.layers.0.self_attn_norm.weight'.format(key_prefix)].shape[0]
        dit_config["intermediate_size"] = state_dict['{}decoder.layers.0.mlp.gate_proj.weight'.format(key_prefix)].shape[0]
        dit_config["num_heads"] = state_dict['{}decoder.layers.0.self_attn.q_proj.weight'.format(key_prefix)].shape[0] // head_dim

        dit_config["encoder_hidden_size"] = state_dict['{}encoder.lyric_encoder.layers.0.input_layernorm.weight'.format(key_prefix)].shape[0]
        dit_config["encoder_num_heads"] = state_dict['{}encoder.lyric_encoder.layers.0.self_attn.q_proj.weight'.format(key_prefix)].shape[0] // head_dim
        dit_config["encoder_intermediate_size"] = state_dict['{}encoder.lyric_encoder.layers.0.mlp.gate_proj.weight'.format(key_prefix)].shape[0]
        dit_config["num_dit_layers"] = count_blocks(state_dict_keys, '{}decoder.layers.'.format(key_prefix) + '{}.')
        return dit_config

    if '{}encoder.pan_blocks.1.cv4.conv.weight'.format(key_prefix) in state_dict_keys: # RT-DETR_v4
        dit_config = {}
        dit_config["image_model"] = "RT_DETR_v4"
        dit_config["enc_h"] = state_dict['{}encoder.pan_blocks.1.cv4.conv.weight'.format(key_prefix)].shape[0]
        return dit_config

    # Depth Anything 3 (repackaged to ComfyUI's native Dinov2Model layout via scripts/convert_da3.py)
    if '{}backbone.embeddings.patch_embeddings.projection.weight'.format(key_prefix) in state_dict_keys:
        dit_config = {}
        dit_config["image_model"] = "DepthAnything3"

        patch_w = state_dict['{}backbone.embeddings.patch_embeddings.projection.weight'.format(key_prefix)]
        embed_dim = patch_w.shape[0]
        depth = count_blocks(state_dict_keys, '{}backbone.encoder.layer.'.format(key_prefix) + '{}.')

        # Backbone preset is determined by embed_dim (matches vits/vitb/vitl/vitg).
        backbone_name = {384: "vits", 768: "vitb", 1024: "vitl", 1536: "vitg"}.get(embed_dim)
        if backbone_name is None:
            return None
        dit_config["backbone_name"] = backbone_name

        # Detect DA3 extensions on top of vanilla DINOv2.
        has_camera_token = '{}backbone.embeddings.camera_token'.format(key_prefix) in state_dict_keys
        # qk-norm shows up as `attention.q_norm.weight` on enabled blocks.
        qknorm_indices = [
            i for i in range(depth)
            if '{}backbone.encoder.layer.{}.attention.q_norm.weight'.format(key_prefix, i) in state_dict_keys
        ]
        qknorm_start = qknorm_indices[0] if qknorm_indices else -1

        # The DA3 main-series configs always set alt_start == qknorm_start == rope_start.
        # cat_token=True is implied by the presence of camera_token.
        if has_camera_token:
            dit_config["alt_start"] = qknorm_start
            dit_config["rope_start"] = qknorm_start
            dit_config["qknorm_start"] = qknorm_start
            dit_config["cat_token"] = True
        else:
            dit_config["alt_start"] = -1
            dit_config["rope_start"] = -1
            dit_config["qknorm_start"] = -1
            dit_config["cat_token"] = False

        # Detect head type and config.
        has_aux = '{}head.scratch.refinenet1_aux.out_conv.weight'.format(key_prefix) in state_dict_keys
        dit_config["head_dim_in"] = state_dict['{}head.projects.0.weight'.format(key_prefix)].shape[1]
        dit_config["head_features"] = state_dict['{}head.scratch.refinenet1.out_conv.weight'.format(key_prefix)].shape[0]
        dit_config["head_out_channels"] = [
            state_dict['{}head.projects.{}.weight'.format(key_prefix, i)].shape[0]
            for i in range(4)
        ]
        if has_aux:
            # DualDPT: dim_in = 2 * embed_dim (because cat_token doubles token width).
            dit_config["head_type"] = "dualdpt"
            dit_config["head_output_dim"] = 2
            dit_config["head_use_sky_head"] = False
        else:
            dit_config["head_type"] = "dpt"
            dit_config["head_output_dim"] = state_dict[
                '{}head.scratch.output_conv2.2.weight'.format(key_prefix)
            ].shape[0]
            dit_config["head_use_sky_head"] = (
                '{}head.scratch.sky_output_conv2.0.weight'.format(key_prefix) in state_dict_keys
            )

        # out_layers: hard-coded per upstream YAML config (depth-aware default).
        if depth >= 24:
            # vitl: depths used vary between DA3-Large (DualDPT) and Mono/Metric (DPT).
            if has_aux:
                dit_config["out_layers"] = [11, 15, 19, 23]
            else:
                dit_config["out_layers"] = [4, 11, 17, 23]
        else:
            # vits/vitb: 12 blocks
            dit_config["out_layers"] = [5, 7, 9, 11]

        # Camera encoder/decoder presence (multi-view + pose path).
        has_cam_enc = '{}cam_enc.token_norm.weight'.format(key_prefix) in state_dict_keys
        has_cam_dec = '{}cam_dec.fc_t.weight'.format(key_prefix) in state_dict_keys
        dit_config["has_cam_enc"] = has_cam_enc
        dit_config["has_cam_dec"] = has_cam_dec
        if has_cam_enc:
            cam_enc_w = state_dict.get(
                '{}cam_enc.pose_branch.fc2.weight'.format(key_prefix)
            )
            if cam_enc_w is not None:
                dit_config["cam_dim_out"] = cam_enc_w.shape[0]
        if has_cam_dec:
            cam_dec_w = state_dict.get(
                '{}cam_dec.fc_t.weight'.format(key_prefix)
            )
            if cam_dec_w is not None:
                dit_config["cam_dec_dim_in"] = cam_dec_w.shape[1]
        return dit_config

    if '{}layers.0.mlp.linear_fc2.weight'.format(key_prefix) in state_dict_keys: # Ernie Image
        dit_config = {}
        dit_config["image_model"] = "ernie"
        return dit_config

    if 'detector.backbone.vision_backbone.trunk.blocks.0.attn.qkv.weight' in state_dict_keys: # SAM3 / SAM3.1
        if 'detector.transformer.decoder.query_embed.weight' in state_dict_keys:
            dit_config = {}
            dit_config["image_model"] = "SAM3"
            if 'detector.backbone.vision_backbone.propagation_convs.0.conv_1x1.weight' in state_dict_keys:
                dit_config["image_model"] = "SAM31"
            return dit_config

    if '{}input_blocks.0.0.weight'.format(key_prefix) not in state_dict_keys:
        return None

    unet_config = {
        "use_checkpoint": False,
        "image_size": 32,
        "use_spatial_transformer": True,
        "legacy": False
    }

    y_input = '{}label_emb.0.0.weight'.format(key_prefix)
    if y_input in state_dict_keys:
        unet_config["num_classes"] = "sequential"
        unet_config["adm_in_channels"] = state_dict[y_input].shape[1]
    else:
        unet_config["adm_in_channels"] = None

    model_channels = state_dict['{}input_blocks.0.0.weight'.format(key_prefix)].shape[0]
    in_channels = state_dict['{}input_blocks.0.0.weight'.format(key_prefix)].shape[1]

    out_key = '{}out.2.weight'.format(key_prefix)
    if out_key in state_dict:
        out_channels = state_dict[out_key].shape[0]
    else:
        out_channels = 4

    num_res_blocks = []
    channel_mult = []
    transformer_depth = []
    transformer_depth_output = []
    context_dim = None
    use_linear_in_transformer = False

    video_model = False
    video_model_cross = False

    current_res = 1
    count = 0

    last_res_blocks = 0
    last_channel_mult = 0

    input_block_count = count_blocks(state_dict_keys, '{}input_blocks'.format(key_prefix) + '.{}.')
    for count in range(input_block_count):
        prefix = '{}input_blocks.{}.'.format(key_prefix, count)
        prefix_output = '{}output_blocks.{}.'.format(key_prefix, input_block_count - count - 1)

        block_keys = sorted(list(filter(lambda a: a.startswith(prefix), state_dict_keys)))
        if len(block_keys) == 0:
            break

        block_keys_output = sorted(list(filter(lambda a: a.startswith(prefix_output), state_dict_keys)))

        if "{}0.op.weight".format(prefix) in block_keys: #new layer
            num_res_blocks.append(last_res_blocks)
            channel_mult.append(last_channel_mult)

            current_res *= 2
            last_res_blocks = 0
            last_channel_mult = 0
            out = calculate_transformer_depth(prefix_output, state_dict_keys, state_dict)
            if out is not None:
                transformer_depth_output.append(out[0])
            else:
                transformer_depth_output.append(0)
        else:
            res_block_prefix = "{}0.in_layers.0.weight".format(prefix)
            if res_block_prefix in block_keys:
                last_res_blocks += 1
                last_channel_mult = state_dict["{}0.out_layers.3.weight".format(prefix)].shape[0] // model_channels

                out = calculate_transformer_depth(prefix, state_dict_keys, state_dict)
                if out is not None:
                    transformer_depth.append(out[0])
                    if context_dim is None:
                        context_dim = out[1]
                        use_linear_in_transformer = out[2]
                        video_model = out[3]
                        video_model_cross = out[4]
                else:
                    transformer_depth.append(0)

            res_block_prefix = "{}0.in_layers.0.weight".format(prefix_output)
            if res_block_prefix in block_keys_output:
                out = calculate_transformer_depth(prefix_output, state_dict_keys, state_dict)
                if out is not None:
                    transformer_depth_output.append(out[0])
                else:
                    transformer_depth_output.append(0)


    num_res_blocks.append(last_res_blocks)
    channel_mult.append(last_channel_mult)
    if "{}middle_block.1.proj_in.weight".format(key_prefix) in state_dict_keys:
        transformer_depth_middle = count_blocks(state_dict_keys, '{}middle_block.1.transformer_blocks.'.format(key_prefix) + '{}')
    elif "{}middle_block.0.in_layers.0.weight".format(key_prefix) in state_dict_keys:
        transformer_depth_middle = -1
    else:
        transformer_depth_middle = -2

    unet_config["in_channels"] = in_channels
    unet_config["out_channels"] = out_channels
    unet_config["model_channels"] = model_channels
    unet_config["num_res_blocks"] = num_res_blocks
    unet_config["transformer_depth"] = transformer_depth
    unet_config["transformer_depth_output"] = transformer_depth_output
    unet_config["channel_mult"] = channel_mult
    unet_config["transformer_depth_middle"] = transformer_depth_middle
    unet_config['use_linear_in_transformer'] = use_linear_in_transformer
    unet_config["context_dim"] = context_dim

    if video_model:
        unet_config["extra_ff_mix_layer"] = True
        unet_config["use_spatial_context"] = True
        unet_config["merge_strategy"] = "learned_with_images"
        unet_config["merge_factor"] = 0.0
        unet_config["video_kernel_size"] = [3, 1, 1]
        unet_config["use_temporal_resblock"] = True
        unet_config["use_temporal_attention"] = True
        unet_config["disable_temporal_crossattention"] = not video_model_cross
    else:
        unet_config["use_temporal_resblock"] = False
        unet_config["use_temporal_attention"] = False

    heatmap_key = '{}heatmap_head.conv_layers.0.weight'.format(key_prefix)
    if heatmap_key in state_dict_keys:
        unet_config["heatmap_head"] = True

    return unet_config

def model_config_from_unet_config(unet_config, state_dict=None):
    for model_config in comfy.supported_models.models:
        if model_config.matches(unet_config, state_dict):
            return model_config(unet_config)

    logging.error("no match {}".format(unet_config))
    return None

def model_config_from_unet(state_dict, unet_key_prefix, use_base_if_no_match=False, metadata=None):
    unet_config = detect_unet_config(state_dict, unet_key_prefix, metadata=metadata)
    if unet_config is None:
        return None
    model_config = model_config_from_unet_config(unet_config, state_dict)
    if model_config is None and use_base_if_no_match:
        model_config = comfy.supported_models_base.BASE(unet_config)

    # Detect per-layer quantization (mixed precision)
    quant_config = comfy.utils.detect_layer_quantization(state_dict, unet_key_prefix)
    if quant_config:
        model_config.quant_config = quant_config
        logging.info("Detected mixed precision quantization")

    return model_config

def unet_prefix_from_state_dict(state_dict):
    # SAM3: detector.* and tracker.* at top level, no common prefix
    if any(k.startswith("detector.") for k in state_dict) and any(k.startswith("tracker.") for k in state_dict):
        return ""

    candidates = ["model.diffusion_model.", #ldm/sgm models
                  "model.model.", #audio models
                  "net.", #cosmos
                  ]
    counts = {k: 0 for k in candidates}
    for k in state_dict:
        for c in candidates:
            if k.startswith(c):
                counts[c] += 1
                break

    top = max(counts, key=counts.get)
    if counts[top] > 5:
        return top
    else:
        return "model." #aura flow and others


def convert_config(unet_config):
    new_config = unet_config.copy()
    num_res_blocks = new_config.get("num_res_blocks", None)
    channel_mult = new_config.get("channel_mult", None)

    if isinstance(num_res_blocks, int):
        num_res_blocks = len(channel_mult) * [num_res_blocks]

    if "attention_resolutions" in new_config:
        attention_resolutions = new_config.pop("attention_resolutions")
        transformer_depth = new_config.get("transformer_depth", None)
        transformer_depth_middle = new_config.get("transformer_depth_middle", None)

        if isinstance(transformer_depth, int):
            transformer_depth = len(channel_mult) * [transformer_depth]
        if transformer_depth_middle is None:
            transformer_depth_middle =  transformer_depth[-1]
        t_in = []
        t_out = []
        s = 1
        for i in range(len(num_res_blocks)):
            res = num_res_blocks[i]
            d = 0
            if s in attention_resolutions:
                d = transformer_depth[i]

            t_in += [d] * res
            t_out += [d] * (res + 1)
            s *= 2
        transformer_depth = t_in
        new_config["transformer_depth"] = t_in
        new_config["transformer_depth_output"] = t_out
        new_config["transformer_depth_middle"] = transformer_depth_middle

    new_config["num_res_blocks"] = num_res_blocks
    return new_config


def unet_config_from_diffusers_unet(state_dict, dtype=None):
    if "conv_in.weight" not in state_dict:
        return None

    match = {}
    transformer_depth = []

    attn_res = 1
    down_blocks = count_blocks(state_dict, "down_blocks.{}")
    for i in range(down_blocks):
        attn_blocks = count_blocks(state_dict, "down_blocks.{}.attentions.".format(i) + '{}')
        res_blocks = count_blocks(state_dict, "down_blocks.{}.resnets.".format(i) + '{}')
        for ab in range(attn_blocks):
            transformer_count = count_blocks(state_dict, "down_blocks.{}.attentions.{}.transformer_blocks.".format(i, ab) + '{}')
            transformer_depth.append(transformer_count)
            if transformer_count > 0:
                match["context_dim"] = state_dict["down_blocks.{}.attentions.{}.transformer_blocks.0.attn2.to_k.weight".format(i, ab)].shape[1]

        attn_res *= 2
        if attn_blocks == 0:
            for i in range(res_blocks):
                transformer_depth.append(0)

    match["transformer_depth"] = transformer_depth

    match["model_channels"] = state_dict["conv_in.weight"].shape[0]
    match["in_channels"] = state_dict["conv_in.weight"].shape[1]
    match["adm_in_channels"] = None
    if "class_embedding.linear_1.weight" in state_dict:
        match["adm_in_channels"] = state_dict["class_embedding.linear_1.weight"].shape[1]
    elif "add_embedding.linear_1.weight" in state_dict:
        match["adm_in_channels"] = state_dict["add_embedding.linear_1.weight"].shape[1]

    SDXL = {'use_checkpoint': False, 'image_size': 32, 'out_channels': 4, 'use_spatial_transformer': True, 'legacy': False,
            'num_classes': 'sequential', 'adm_in_channels': 2816, 'dtype': dtype, 'in_channels': 4, 'model_channels': 320,
            'num_res_blocks': [2, 2, 2], 'transformer_depth': [0, 0, 2, 2, 10, 10], 'channel_mult': [1, 2, 4], 'transformer_depth_middle': 10,
            'use_linear_in_transformer': True, 'context_dim': 2048, 'num_head_channels': 64, 'transformer_depth_output': [0, 0, 0, 2, 2, 2, 10, 10, 10],
            'use_temporal_attention': False, 'use_temporal_resblock': False}

    SDXL_refiner = {'use_checkpoint': False, 'image_size': 32, 'out_channels': 4, 'use_spatial_transformer': True, 'legacy': False,
                    'num_classes': 'sequential', 'adm_in_channels': 2560, 'dtype': dtype, 'in_channels': 4, 'model_channels': 384,
                    'num_res_blocks': [2, 2, 2, 2], 'transformer_depth': [0, 0, 4, 4, 4, 4, 0, 0], 'channel_mult': [1, 2, 4, 4], 'transformer_depth_middle': 4,
                    'use_linear_in_transformer': True, 'context_dim': 1280, 'num_head_channels': 64, 'transformer_depth_output': [0, 0, 0, 4, 4, 4, 4, 4, 4, 0, 0, 0],
                    'use_temporal_attention': False, 'use_temporal_resblock': False}

    SD21 = {'use_checkpoint': False, 'image_size': 32, 'out_channels': 4, 'use_spatial_transformer': True, 'legacy': False,
            'adm_in_channels': None, 'dtype': dtype, 'in_channels': 4, 'model_channels': 320, 'num_res_blocks': [2, 2, 2, 2],
            'transformer_depth': [1, 1, 1, 1, 1, 1, 0, 0], 'channel_mult': [1, 2, 4, 4], 'transformer_depth_middle': 1, 'use_linear_in_transformer': True,
            'context_dim': 1024, 'num_head_channels': 64, 'transformer_depth_output': [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
            'use_temporal_attention': False, 'use_temporal_resblock': False}

    SD21_uncliph = {'use_checkpoint': False, 'image_size': 32, 'out_channels': 4, 'use_spatial_transformer': True, 'legacy': False,
                    'num_classes': 'sequential', 'adm_in_channels': 2048, 'dtype': dtype, 'in_channels': 4, 'model_channels': 320,
                    'num_res_blocks': [2, 2, 2, 2], 'transformer_depth': [1, 1, 1, 1, 1, 1, 0, 0], 'channel_mult': [1, 2, 4, 4], 'transformer_depth_middle': 1,
                    'use_linear_in_transformer': True, 'context_dim': 1024, 'num_head_channels': 64, 'transformer_depth_output': [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                    'use_temporal_attention': False, 'use_temporal_resblock': False}

    SD21_unclipl = {'use_checkpoint': False, 'image_size': 32, 'out_channels': 4, 'use_spatial_transformer': True, 'legacy': False,
                    'num_classes': 'sequential', 'adm_in_channels': 1536, 'dtype': dtype, 'in_channels': 4, 'model_channels': 320,
                    'num_res_blocks': [2, 2, 2, 2], 'transformer_depth': [1, 1, 1, 1, 1, 1, 0, 0], 'channel_mult': [1, 2, 4, 4], 'transformer_depth_middle': 1,
                    'use_linear_in_transformer': True, 'context_dim': 1024, 'num_head_channels': 64, 'transformer_depth_output': [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                    'use_temporal_attention': False, 'use_temporal_resblock': False}

    SD15 = {'use_checkpoint': False, 'image_size': 32, 'out_channels': 4, 'use_spatial_transformer': True, 'legacy': False, 'adm_in_channels': None,
            'dtype': dtype, 'in_channels': 4, 'model_channels': 320, 'num_res_blocks': [2, 2, 2, 2], 'transformer_depth': [1, 1, 1, 1, 1, 1, 0, 0],
            'channel_mult': [1, 2, 4, 4], 'transformer_depth_middle': 1, 'use_linear_in_transformer': False, 'context_dim': 768, 'num_heads': 8,
            'transformer_depth_output': [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
            'use_temporal_attention': False, 'use_temporal_resblock': False}

    SDXL_mid_cnet = {'use_checkpoint': False, 'image_size': 32, 'out_channels': 4, 'use_spatial_transformer': True, 'legacy': False,
                     'num_classes': 'sequential', 'adm_in_channels': 2816, 'dtype': dtype, 'in_channels': 4, 'model_channels': 320,
                     'num_res_blocks': [2, 2, 2], 'transformer_depth': [0, 0, 0, 0, 1, 1], 'channel_mult': [1, 2, 4], 'transformer_depth_middle': 1,
                     'use_linear_in_transformer': True, 'context_dim': 2048, 'num_head_channels': 64, 'transformer_depth_output': [0, 0, 0, 0, 0, 0, 1, 1, 1],
                     'use_temporal_attention': False, 'use_temporal_resblock': False}

    SDXL_small_cnet = {'use_checkpoint': False, 'image_size': 32, 'out_channels': 4, 'use_spatial_transformer': True, 'legacy': False,
                       'num_classes': 'sequential', 'adm_in_channels': 2816, 'dtype': dtype, 'in_channels': 4, 'model_channels': 320,
                       'num_res_blocks': [2, 2, 2], 'transformer_depth': [0, 0, 0, 0, 0, 0], 'channel_mult': [1, 2, 4], 'transformer_depth_middle': 0,
                       'use_linear_in_transformer': True, 'num_head_channels': 64, 'context_dim': 1, 'transformer_depth_output': [0, 0, 0, 0, 0, 0, 0, 0, 0],
                       'use_temporal_attention': False, 'use_temporal_resblock': False}

    SDXL_diffusers_inpaint = {'use_checkpoint': False, 'image_size': 32, 'out_channels': 4, 'use_spatial_transformer': True, 'legacy': False,
                              'num_classes': 'sequential', 'adm_in_channels': 2816, 'dtype': dtype, 'in_channels': 9, 'model_channels': 320,
                              'num_res_blocks': [2, 2, 2], 'transformer_depth': [0, 0, 2, 2, 10, 10], 'channel_mult': [1, 2, 4], 'transformer_depth_middle': 10,
                              'use_linear_in_transformer': True, 'context_dim': 2048, 'num_head_channels': 64, 'transformer_depth_output': [0, 0, 0, 2, 2, 2, 10, 10, 10],
                              'use_temporal_attention': False, 'use_temporal_resblock': False}

    SDXL_diffusers_ip2p = {'use_checkpoint': False, 'image_size': 32, 'out_channels': 4, 'use_spatial_transformer': True, 'legacy': False,
                              'num_classes': 'sequential', 'adm_in_channels': 2816, 'dtype': dtype, 'in_channels': 8, 'model_channels': 320,
                              'num_res_blocks': [2, 2, 2], 'transformer_depth': [0, 0, 2, 2, 10, 10], 'channel_mult': [1, 2, 4], 'transformer_depth_middle': 10,
                              'use_linear_in_transformer': True, 'context_dim': 2048, 'num_head_channels': 64, 'transformer_depth_output': [0, 0, 0, 2, 2, 2, 10, 10, 10],
                              'use_temporal_attention': False, 'use_temporal_resblock': False}

    SSD_1B = {'use_checkpoint': False, 'image_size': 32, 'out_channels': 4, 'use_spatial_transformer': True, 'legacy': False,
              'num_classes': 'sequential', 'adm_in_channels': 2816, 'dtype': dtype, 'in_channels': 4, 'model_channels': 320,
              'num_res_blocks': [2, 2, 2], 'transformer_depth': [0, 0, 2, 2, 4, 4], 'transformer_depth_output': [0, 0, 0, 1, 1, 2, 10, 4, 4],
              'channel_mult': [1, 2, 4], 'transformer_depth_middle': -1, 'use_linear_in_transformer': True, 'context_dim': 2048, 'num_head_channels': 64,
              'use_temporal_attention': False, 'use_temporal_resblock': False}

    Segmind_Vega = {'use_checkpoint': False, 'image_size': 32, 'out_channels': 4, 'use_spatial_transformer': True, 'legacy': False,
              'num_classes': 'sequential', 'adm_in_channels': 2816, 'dtype': dtype, 'in_channels': 4, 'model_channels': 320,
              'num_res_blocks': [2, 2, 2], 'transformer_depth': [0, 0, 1, 1, 2, 2], 'transformer_depth_output': [0, 0, 0, 1, 1, 1, 2, 2, 2],
              'channel_mult': [1, 2, 4], 'transformer_depth_middle': -1, 'use_linear_in_transformer': True, 'context_dim': 2048, 'num_head_channels': 64,
              'use_temporal_attention': False, 'use_temporal_resblock': False}

    KOALA_700M = {'use_checkpoint': False, 'image_size': 32, 'out_channels': 4, 'use_spatial_transformer': True, 'legacy': False,
              'num_classes': 'sequential', 'adm_in_channels': 2816, 'dtype': dtype, 'in_channels': 4, 'model_channels': 320,
              'num_res_blocks': [1, 1, 1], 'transformer_depth': [0, 2, 5], 'transformer_depth_output': [0, 0, 2, 2, 5, 5],
              'channel_mult': [1, 2, 4], 'transformer_depth_middle': -2, 'use_linear_in_transformer': True, 'context_dim': 2048, 'num_head_channels': 64,
              'use_temporal_attention': False, 'use_temporal_resblock': False}

    KOALA_1B = {'use_checkpoint': False, 'image_size': 32, 'out_channels': 4, 'use_spatial_transformer': True, 'legacy': False,
              'num_classes': 'sequential', 'adm_in_channels': 2816, 'dtype': dtype, 'in_channels': 4, 'model_channels': 320,
              'num_res_blocks': [1, 1, 1], 'transformer_depth': [0, 2, 6], 'transformer_depth_output': [0, 0, 2, 2, 6, 6],
              'channel_mult': [1, 2, 4], 'transformer_depth_middle': 6, 'use_linear_in_transformer': True, 'context_dim': 2048, 'num_head_channels': 64,
              'use_temporal_attention': False, 'use_temporal_resblock': False}

    SD09_XS = {'use_checkpoint': False, 'image_size': 32, 'out_channels': 4, 'use_spatial_transformer': True, 'legacy': False,
            'adm_in_channels': None, 'dtype': dtype, 'in_channels': 4, 'model_channels': 320, 'num_res_blocks': [1, 1, 1],
            'transformer_depth': [1, 1, 1], 'channel_mult': [1, 2, 4], 'transformer_depth_middle': -2, 'use_linear_in_transformer': True,
            'context_dim': 1024, 'num_head_channels': 64, 'transformer_depth_output': [1, 1, 1, 1, 1, 1],
            'use_temporal_attention': False, 'use_temporal_resblock': False, 'disable_self_attentions': [True, False, False]}

    SD_XS = {'use_checkpoint': False, 'image_size': 32, 'out_channels': 4, 'use_spatial_transformer': True, 'legacy': False,
            'adm_in_channels': None, 'dtype': dtype, 'in_channels': 4, 'model_channels': 320, 'num_res_blocks': [1, 1, 1],
            'transformer_depth': [0, 1, 1], 'channel_mult': [1, 2, 4], 'transformer_depth_middle': -2, 'use_linear_in_transformer': False,
            'context_dim': 768, 'num_head_channels': 64, 'transformer_depth_output': [0, 0, 1, 1, 1, 1],
            'use_temporal_attention': False, 'use_temporal_resblock': False}

    SD15_diffusers_inpaint = {'use_checkpoint': False, 'image_size': 32, 'out_channels': 4, 'use_spatial_transformer': True, 'legacy': False, 'adm_in_channels': None,
            'dtype': dtype, 'in_channels': 9, 'model_channels': 320, 'num_res_blocks': [2, 2, 2, 2], 'transformer_depth': [1, 1, 1, 1, 1, 1, 0, 0],
            'channel_mult': [1, 2, 4, 4], 'transformer_depth_middle': 1, 'use_linear_in_transformer': False, 'context_dim': 768, 'num_heads': 8,
            'transformer_depth_output': [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
            'use_temporal_attention': False, 'use_temporal_resblock': False}

    LotusD = {'use_checkpoint': False, 'image_size': 32, 'out_channels': 4, 'use_spatial_transformer': True, 'legacy': False, 'adm_in_channels': 4,
            'dtype': dtype, 'in_channels': 4, 'model_channels': 320, 'num_res_blocks': [2, 2, 2, 2], 'transformer_depth': [1, 1, 1, 1, 1, 1, 0, 0],
            'channel_mult': [1, 2, 4, 4], 'transformer_depth_middle': 1, 'use_linear_in_transformer': True, 'context_dim': 1024, 'num_head_channels': 64,
            'transformer_depth_output': [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
            'use_temporal_attention': False, 'use_temporal_resblock': False}

    supported_models = [LotusD, SDXL, SDXL_refiner, SD21, SD15, SD21_uncliph, SD21_unclipl, SDXL_mid_cnet, SDXL_small_cnet, SDXL_diffusers_inpaint, SSD_1B, Segmind_Vega, KOALA_700M, KOALA_1B, SD09_XS, SD_XS, SDXL_diffusers_ip2p, SD15_diffusers_inpaint]

    for unet_config in supported_models:
        matches = True
        for k in match:
            if match[k] != unet_config[k]:
                matches = False
                break
        if matches:
            return convert_config(unet_config)
    return None

def model_config_from_diffusers_unet(state_dict):
    unet_config = unet_config_from_diffusers_unet(state_dict)
    if unet_config is not None:
        return model_config_from_unet_config(unet_config)
    return None

def convert_diffusers_mmdit(state_dict, output_prefix=""):
    out_sd = {}

    if 'joint_transformer_blocks.0.attn.add_k_proj.weight' in state_dict: #AuraFlow
        num_joint = count_blocks(state_dict, 'joint_transformer_blocks.{}.')
        num_single = count_blocks(state_dict, 'single_transformer_blocks.{}.')
        sd_map = comfy.utils.auraflow_to_diffusers({"n_double_layers": num_joint, "n_layers": num_joint + num_single}, output_prefix=output_prefix)
    elif 'adaln_single.emb.timestep_embedder.linear_1.bias' in state_dict and 'pos_embed.proj.bias' in state_dict: # PixArt
        num_blocks = count_blocks(state_dict, 'transformer_blocks.{}.')
        sd_map = comfy.utils.pixart_to_diffusers({"depth": num_blocks}, output_prefix=output_prefix)
    elif 'noise_refiner.0.attention.norm_k.weight' in state_dict:
        n_layers = count_blocks(state_dict, 'layers.{}.')
        dim = state_dict['noise_refiner.0.attention.to_k.weight'].shape[0]
        sd_map = comfy.utils.z_image_to_diffusers({"n_layers": n_layers, "dim": dim}, output_prefix=output_prefix)
        for k in state_dict: # For zeta chroma
            if k not in sd_map:
                sd_map[k] = k
    elif 'x_embedder.weight' in state_dict: #Flux
        depth = count_blocks(state_dict, 'transformer_blocks.{}.')
        depth_single_blocks = count_blocks(state_dict, 'single_transformer_blocks.{}.')
        hidden_size = state_dict["x_embedder.bias"].shape[0]
        sd_map = comfy.utils.flux_to_diffusers({"depth": depth, "depth_single_blocks": depth_single_blocks, "hidden_size": hidden_size}, output_prefix=output_prefix)
    elif 'transformer_blocks.0.attn.add_q_proj.weight' in state_dict and 'pos_embed.proj.weight' in state_dict: #SD3
        num_blocks = count_blocks(state_dict, 'transformer_blocks.{}.')
        depth = state_dict["pos_embed.proj.weight"].shape[0] // 64
        sd_map = comfy.utils.mmdit_to_diffusers({"depth": depth, "num_blocks": num_blocks}, output_prefix=output_prefix)
    else:
        return None

    for k in sd_map:
        weight = state_dict.get(k, None)
        if weight is not None:
            t = sd_map[k]

            if not isinstance(t, str):
                if len(t) > 2:
                    fun = t[2]
                else:
                    fun = lambda a: a
                offset = t[1]
                if offset is not None:
                    old_weight = out_sd.get(t[0], None)
                    if old_weight is None:
                        old_weight = torch.empty_like(weight)
                    if old_weight.shape[offset[0]] < offset[1] + offset[2]:
                        exp = list(weight.shape)
                        exp[offset[0]] = offset[1] + offset[2]
                        new = torch.empty(exp, device=weight.device, dtype=weight.dtype)
                        new[:old_weight.shape[0]] = old_weight
                        old_weight = new

                    if old_weight is out_sd.get(t[0], None) and comfy.memory_management.aimdo_enabled:
                        old_weight = old_weight.clone()

                    w = old_weight.narrow(offset[0], offset[1], offset[2])
                else:
                    if comfy.memory_management.aimdo_enabled:
                        weight = weight.clone()
                    old_weight = weight
                    w = weight
                w[:] = fun(weight)
                t = t[0]
                out_sd[t] = old_weight
            else:
                out_sd[t] = weight
            state_dict.pop(k)

    return out_sd
