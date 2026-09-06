# copyright 2025 Maifee Ul Asad @ github.com/maifeeulasad
# copyright under GNU GENERAL PUBLIC LICENSE, Version 3, 29 June 2007

"""
Enhanced model loading nodes with GPUDirect Storage support
"""

import logging
import time
import asyncio
from typing import Optional, Dict, Any

import torch
import folder_paths
import comfy.sd
import comfy.utils
from comfy.comfy_types import IO, ComfyNodeABC, InputTypeDict


class CheckpointLoaderGDS(ComfyNodeABC):
    """
    Enhanced checkpoint loader with GPUDirect Storage support
    Provides direct SSD-to-GPU loading and prefetching capabilities
    """
    
    @classmethod
    def INPUT_TYPES(s) -> InputTypeDict:
        return {
            "required": {
                "ckpt_name": (folder_paths.get_filename_list("checkpoints"), {
                    "tooltip": "The name of the checkpoint (model) to load with GDS optimization."
                }),
            },
            "optional": {
                "prefetch": ("BOOLEAN", {
                    "default": False, 
                    "tooltip": "Prefetch model to GPU cache for faster loading."
                }),
                "use_gds": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Use GPUDirect Storage if available."
                }),
                "target_device": (["auto", "cuda:0", "cuda:1", "cuda:2", "cuda:3", "cpu"], {
                    "default": "auto",
                    "tooltip": "Target device for model loading."
                })
            }
        }
    
    RETURN_TYPES = ("MODEL", "CLIP", "VAE", "STRING")
    RETURN_NAMES = ("model", "clip", "vae", "load_info")
    OUTPUT_TOOLTIPS = (
        "The model used for denoising latents.",
        "The CLIP model used for encoding text prompts.",
        "The VAE model used for encoding and decoding images to and from latent space.",
        "Loading information and statistics."
    )
    FUNCTION = "load_checkpoint_gds"
    CATEGORY = "loaders/advanced"
    DESCRIPTION = "Enhanced checkpoint loader with GPUDirect Storage support for direct SSD-to-GPU loading."
    EXPERIMENTAL = True

    def load_checkpoint_gds(self, ckpt_name: str, prefetch: bool = False, use_gds: bool = True, target_device: str = "auto"):
        start_time = time.time()
        
        ckpt_path = folder_paths.get_full_path_or_raise("checkpoints", ckpt_name)
        
        # Determine target device
        if target_device == "auto":
            device = None  # Let the system decide
        elif target_device == "cpu":
            device = torch.device("cpu")
        else:
            device = torch.device(target_device)
        
        load_info = {
            "file": ckpt_name,
            "path": ckpt_path,
            "target_device": str(device) if device else "auto",
            "gds_enabled": use_gds,
            "prefetch_used": prefetch
        }
        
        try:
            # Prefetch if requested
            if prefetch and use_gds:
                try:
                    from comfy.gds_loader import prefetch_model_gds
                    prefetch_success = prefetch_model_gds(ckpt_path)
                    load_info["prefetch_success"] = prefetch_success
                    if prefetch_success:
                        logging.info(f"Prefetched {ckpt_name} to GPU cache")
                except Exception as e:
                    logging.warning(f"Prefetch failed for {ckpt_name}: {e}")
                    load_info["prefetch_error"] = str(e)
            
            # Load checkpoint with potential GDS optimization
            if use_gds and device and device.type == 'cuda':
                try:
                    from comfy.gds_loader import get_gds_instance
                    gds = get_gds_instance()
                    
                    # Check if GDS should be used for this file
                    if gds._should_use_gds(ckpt_path):
                        load_info["loader_used"] = "GDS"
                        logging.info(f"Loading {ckpt_name} with GDS")
                    else:
                        load_info["loader_used"] = "Standard"
                        logging.info(f"Loading {ckpt_name} with standard method (file too small for GDS)")
                        
                except Exception as e:
                    logging.warning(f"GDS check failed, using standard loading: {e}")
                    load_info["loader_used"] = "Standard (GDS failed)"
            else:
                load_info["loader_used"] = "Standard"
            
            # Load the actual checkpoint
            out = comfy.sd.load_checkpoint_guess_config(
                ckpt_path, 
                output_vae=True, 
                output_clip=True, 
                embedding_directory=folder_paths.get_folder_paths("embeddings")
            )
            
            load_time = time.time() - start_time
            load_info["load_time_seconds"] = round(load_time, 3)
            load_info["load_success"] = True
            
            # Format load info as string
            info_str = f"Loaded: {ckpt_name}\n"
            info_str += f"Method: {load_info['loader_used']}\n"
            info_str += f"Time: {load_info['load_time_seconds']}s\n"
            info_str += f"Device: {load_info['target_device']}"
            
            if "prefetch_success" in load_info:
                info_str += f"\nPrefetch: {'✓' if load_info['prefetch_success'] else '✗'}"
            
            logging.info(f"Checkpoint loaded: {ckpt_name} in {load_time:.3f}s using {load_info['loader_used']}")
            
            return (*out[:3], info_str)
            
        except Exception as e:
            load_info["load_success"] = False
            load_info["error"] = str(e)
            error_str = f"Failed to load: {ckpt_name}\nError: {str(e)}"
            logging.error(f"Checkpoint loading failed: {e}")
            raise RuntimeError(error_str)


class ModelPrefetcher(ComfyNodeABC):
    """
    Node for prefetching models to GPU cache
    """
    
    @classmethod
    def INPUT_TYPES(s) -> InputTypeDict:
        return {
            "required": {
                "checkpoint_names": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "List of checkpoint names to prefetch (one per line)."
                }),
                "prefetch_enabled": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Enable/disable prefetching."
                })
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prefetch_report",)
    OUTPUT_TOOLTIPS = ("Report of prefetch operations.",)
    FUNCTION = "prefetch_models"
    CATEGORY = "loaders/advanced"
    DESCRIPTION = "Prefetch multiple models to GPU cache for faster loading."
    OUTPUT_NODE = True

    def prefetch_models(self, checkpoint_names: str, prefetch_enabled: bool = True):
        if not prefetch_enabled:
            return ("Prefetching disabled",)
        
        # Parse checkpoint names
        names = [name.strip() for name in checkpoint_names.split('\n') if name.strip()]
        
        if not names:
            return ("No checkpoints specified for prefetching",)
        
        try:
            from comfy.gds_loader import prefetch_model_gds
        except ImportError:
            return ("GDS not available for prefetching",)
        
        results = []
        successful_prefetches = 0
        
        for name in names:
            try:
                ckpt_path = folder_paths.get_full_path_or_raise("checkpoints", name)
                success = prefetch_model_gds(ckpt_path)
                
                if success:
                    results.append(f"✓ {name}")
                    successful_prefetches += 1
                else:
                    results.append(f"✗ {name} (prefetch failed)")
                    
            except Exception as e:
                results.append(f"✗ {name} (error: {str(e)[:50]})")
        
        report = f"Prefetch Report ({successful_prefetches}/{len(names)} successful):\n"
        report += "\n".join(results)
        
        return (report,)


class GDSStats(ComfyNodeABC):
    """
    Node for displaying GDS statistics
    """
    
    @classmethod
    def INPUT_TYPES(s) -> InputTypeDict:
        return {
            "required": {
                "refresh": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Refresh statistics."
                })
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("stats_report",)
    OUTPUT_TOOLTIPS = ("GDS statistics and performance report.",)
    FUNCTION = "get_stats"
    CATEGORY = "utils/advanced"
    DESCRIPTION = "Display GPUDirect Storage statistics and performance metrics."
    OUTPUT_NODE = True

    def get_stats(self, refresh: bool = False):
        try:
            from comfy.gds_loader import get_gds_stats
            stats = get_gds_stats()
            
            report = "=== GPUDirect Storage Statistics ===\n\n"
            
            # Availability
            report += f"GDS Available: {'✓' if stats['gds_available'] else '✗'}\n"
            
            # Usage statistics
            report += f"Total Loads: {stats['total_loads']}\n"
            report += f"GDS Loads: {stats['gds_loads']} ({stats['gds_usage_percent']:.1f}%)\n"
            report += f"Fallback Loads: {stats['fallback_loads']}\n\n"
            
            # Performance metrics
            if stats['total_bytes_gds'] > 0:
                gb_transferred = stats['total_bytes_gds'] / (1024**3)
                report += f"Data Transferred: {gb_transferred:.2f} GB\n"
                report += f"Average Bandwidth: {stats['avg_bandwidth_gbps']:.2f} GB/s\n"
                report += f"Total GDS Time: {stats['total_time_gds']:.2f}s\n\n"
            
            # Configuration
            config = stats.get('config', {})
            if config:
                report += "Configuration:\n"
                report += f"- Enabled: {config.get('enabled', 'Unknown')}\n"
                report += f"- Min File Size: {config.get('min_file_size_mb', 'Unknown')} MB\n"
                report += f"- Chunk Size: {config.get('chunk_size_mb', 'Unknown')} MB\n"
                report += f"- Max Streams: {config.get('max_concurrent_streams', 'Unknown')}\n"
                report += f"- Prefetch: {config.get('prefetch_enabled', 'Unknown')}\n"
                report += f"- Fallback: {config.get('fallback_to_cpu', 'Unknown')}\n"
            
            return (report,)
            
        except ImportError:
            return ("GDS module not available",)
        except Exception as e:
            return (f"Error retrieving GDS stats: {str(e)}",)


# Node mappings
NODE_CLASS_MAPPINGS = {
    "CheckpointLoaderGDS": CheckpointLoaderGDS,
    "ModelPrefetcher": ModelPrefetcher,
    "GDSStats": GDSStats,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CheckpointLoaderGDS": "Load Checkpoint (GDS)",
    "ModelPrefetcher": "Model Prefetcher",
    "GDSStats": "GDS Statistics",
}