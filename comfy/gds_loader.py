# copyright 2025 Maifee Ul Asad @ github.com/maifeeulasad
# copyright under GNU GENERAL PUBLIC LICENSE, Version 3, 29 June 2007

"""
GPUDirect Storage (GDS) Integration for ComfyUI
Direct SSD-to-GPU model loading without RAM/CPU bottlenecks
Still there will be some CPU/RAM usage, mostly for safetensors parsing and small buffers.

This module provides GPUDirect Storage functionality to load models directly
from NVMe SSDs to GPU memory, bypassing system RAM and CPU.
"""

import os
import logging
import torch
import time
from typing import Optional, Dict, Any, Union
from pathlib import Path
import safetensors
import gc
import mmap
from dataclasses import dataclass

try:
    import cupy
    import cupy.cuda.runtime as cuda_runtime
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    logging.warning("CuPy not available. GDS will use fallback mode.")

try:
    import cudf  # RAPIDS for GPU dataframes
    RAPIDS_AVAILABLE = True
except ImportError:
    RAPIDS_AVAILABLE = False

try:
    import pynvml
    pynvml.nvmlInit()
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False
    logging.warning("NVIDIA-ML-Py not available. GPU monitoring disabled.")

@dataclass
class GDSConfig:
    """Configuration for GPUDirect Storage"""
    enabled: bool = True
    min_file_size_mb: int = 100  # Only use GDS for files larger than this
    chunk_size_mb: int = 64      # Size of chunks to transfer
    use_pinned_memory: bool = True
    prefetch_enabled: bool = True
    compression_aware: bool = True
    max_concurrent_streams: int = 4
    fallback_to_cpu: bool = True
    show_stats: bool = False     # Whether to show stats on exit


class GDSError(Exception):
    """GDS-specific errors"""
    pass


class GPUDirectStorage:
    """
    GPUDirect Storage implementation for ComfyUI
    Enables direct SSD-to-GPU transfers for model loading
    """
    
    def __init__(self, config: Optional[GDSConfig] = None):
        self.config = config or GDSConfig()
        self.device = torch.cuda.current_device() if torch.cuda.is_available() else None
        self.cuda_streams = []
        self.pinned_buffers = {}
        self.stats = {
            'gds_loads': 0,
            'fallback_loads': 0,
            'total_bytes_gds': 0,
            'total_time_gds': 0.0,
            'avg_bandwidth_gbps': 0.0
        }
        
        # Initialize GDS if available
        self._gds_available = self._check_gds_availability()
        if self._gds_available:
            self._init_gds()
        else:
            logging.warning("GDS not available, using fallback methods")
    
    def _check_gds_availability(self) -> bool:
        """Check if GDS is available on the system"""
        if not torch.cuda.is_available():
            return False
        
        if not CUPY_AVAILABLE:
            return False
        
        # Check for GPUDirect Storage support
        try:
            # Check CUDA version (GDS requires CUDA 11.4+)
            cuda_version = torch.version.cuda
            if cuda_version:
                major, minor = map(int, cuda_version.split('.')[:2])
                if major < 11 or (major == 11 and minor < 4):
                    logging.warning(f"CUDA {cuda_version} detected. GDS requires CUDA 11.4+")
                    return False
            
            # Check if cuFile is available (part of CUDA toolkit)
            try:
                import cupy.cuda.cufile as cufile
                # Try to initialize cuFile
                cufile.initialize()
                return True
            except (ImportError, RuntimeError) as e:
                logging.warning(f"cuFile not available: {e}")
                return False
                
        except Exception as e:
            logging.warning(f"GDS availability check failed: {e}")
            return False
    
    def _init_gds(self):
        """Initialize GDS resources"""
        try:
            # Create CUDA streams for async operations
            for i in range(self.config.max_concurrent_streams):
                stream = torch.cuda.Stream()
                self.cuda_streams.append(stream)
            
            # Pre-allocate pinned memory buffers
            if self.config.use_pinned_memory:
                self._allocate_pinned_buffers()
            
            logging.info(f"GDS initialized with {len(self.cuda_streams)} streams")
            
        except Exception as e:
            logging.error(f"Failed to initialize GDS: {e}")
            self._gds_available = False
    
    def _allocate_pinned_buffers(self):
        """Pre-allocate pinned memory buffers for staging"""
        try:
            # Allocate buffers of different sizes
            buffer_sizes = [16, 32, 64, 128, 256]  # MB
            
            for size_mb in buffer_sizes:
                size_bytes = size_mb * 1024 * 1024
                # Allocate pinned memory using CuPy
                if CUPY_AVAILABLE:
                    buffer = cupy.cuda.alloc_pinned_memory(size_bytes)
                    self.pinned_buffers[size_mb] = buffer
                    
        except Exception as e:
            logging.warning(f"Failed to allocate pinned buffers: {e}")
    
    def _get_file_size(self, file_path: str) -> int:
        """Get file size in bytes"""
        return os.path.getsize(file_path)
    
    def _should_use_gds(self, file_path: str) -> bool:
        """Determine if GDS should be used for this file"""
        if not self._gds_available or not self.config.enabled:
            return False
        
        file_size_mb = self._get_file_size(file_path) / (1024 * 1024)
        return file_size_mb >= self.config.min_file_size_mb
    
    def _load_with_gds(self, file_path: str) -> Dict[str, torch.Tensor]:
        """Load model using GPUDirect Storage"""
        start_time = time.time()
        
        try:
            if file_path.lower().endswith(('.safetensors', '.sft')):
                return self._load_safetensors_gds(file_path)
            else:
                return self._load_pytorch_gds(file_path)
                
        except Exception as e:
            logging.error(f"GDS loading failed for {file_path}: {e}")
            if self.config.fallback_to_cpu:
                logging.info("Falling back to CPU loading")
                self.stats['fallback_loads'] += 1
                return self._load_fallback(file_path)
            else:
                raise GDSError(f"GDS loading failed: {e}")
        finally:
            load_time = time.time() - start_time
            self.stats['total_time_gds'] += load_time
    
    def _load_safetensors_gds(self, file_path: str) -> Dict[str, torch.Tensor]:
        """Load safetensors file using GDS"""
        try:
            import cupy.cuda.cufile as cufile
            
            # Open file with cuFile for direct GPU loading
            with cufile.CuFileManager() as manager:
                # Memory-map the file for efficient access
                with open(file_path, 'rb') as f:
                    # Use mmap for large files
                    with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mmapped_file:
                        
                        # Parse safetensors header
                        header_size = int.from_bytes(mmapped_file[:8], 'little')
                        header_bytes = mmapped_file[8:8+header_size]
                        
                        import json
                        header = json.loads(header_bytes.decode('utf-8'))
                        
                        # Load tensors directly to GPU
                        tensors = {}
                        data_offset = 8 + header_size
                        
                        for name, info in header.items():
                            if name == "__metadata__":
                                continue
                            
                            dtype_map = {
                                'F32': torch.float32,
                                'F16': torch.float16,
                                'BF16': torch.bfloat16,
                                'I8': torch.int8,
                                'I16': torch.int16,
                                'I32': torch.int32,
                                'I64': torch.int64,
                                'U8': torch.uint8,
                            }
                            
                            dtype = dtype_map.get(info['dtype'], torch.float32)
                            shape = info['shape']
                            start_offset = data_offset + info['data_offsets'][0]
                            end_offset = data_offset + info['data_offsets'][1]
                            
                            # Direct GPU allocation
                            tensor = torch.empty(shape, dtype=dtype, device=f'cuda:{self.device}')
                            
                            # Use cuFile for direct transfer
                            tensor_bytes = end_offset - start_offset
                            
                            # Get GPU memory pointer
                            gpu_ptr = tensor.data_ptr()
                            
                            # Direct file-to-GPU transfer
                            cufile.copy_from_file(
                                gpu_ptr,
                                mmapped_file[start_offset:end_offset],
                                tensor_bytes
                            )
                            
                            tensors[name] = tensor
                        
                        self.stats['gds_loads'] += 1
                        self.stats['total_bytes_gds'] += self._get_file_size(file_path)
                        
                        return tensors
                        
        except Exception as e:
            logging.error(f"GDS safetensors loading failed: {e}")
            raise
    
    def _load_pytorch_gds(self, file_path: str) -> Dict[str, torch.Tensor]:
        """Load PyTorch file using GDS with staging"""
        try:
            # For PyTorch files, we need to use a staging approach
            # since torch.load doesn't support direct GPU loading
            
            # Load to pinned memory first
            with open(file_path, 'rb') as f:
                file_size = self._get_file_size(file_path)
                
                # Choose appropriate buffer or allocate new one
                buffer_size_mb = min(256, max(64, file_size // (1024 * 1024)))
                
                if buffer_size_mb in self.pinned_buffers:
                    pinned_buffer = self.pinned_buffers[buffer_size_mb]
                else:
                    # Allocate temporary pinned buffer
                    pinned_buffer = cupy.cuda.alloc_pinned_memory(file_size)
                
                # Read file to pinned memory
                f.readinto(pinned_buffer)
                
                # Use torch.load with map_location to specific GPU
                # This will be faster due to pinned memory
                state_dict = torch.load(
                    f,
                    map_location=f'cuda:{self.device}',
                    weights_only=True
                )
                
                self.stats['gds_loads'] += 1
                self.stats['total_bytes_gds'] += file_size
                
                return state_dict
                
        except Exception as e:
            logging.error(f"GDS PyTorch loading failed: {e}")
            raise
    
    def _load_fallback(self, file_path: str) -> Dict[str, torch.Tensor]:
        """Fallback loading method using standard approaches"""
        if file_path.lower().endswith(('.safetensors', '.sft')):
            # Use safetensors with device parameter
            with safetensors.safe_open(file_path, framework="pt", device=f'cuda:{self.device}') as f:
                return {k: f.get_tensor(k) for k in f.keys()}
        else:
            # Standard PyTorch loading
            return torch.load(file_path, map_location=f'cuda:{self.device}', weights_only=True)
    
    def load_model(self, file_path: str, device: Optional[torch.device] = None) -> Dict[str, torch.Tensor]:
        """
        Main entry point for loading models with GDS
        
        Args:
            file_path: Path to the model file
            device: Target device (if None, uses current CUDA device)
            
        Returns:
            Dictionary of tensors loaded directly to GPU
        """
        if device is not None and device.type == 'cuda':
            self.device = device.index or 0
        
        if self._should_use_gds(file_path):
            logging.info(f"Loading {file_path} with GDS")
            return self._load_with_gds(file_path)
        else:
            logging.info(f"Loading {file_path} with standard method")
            self.stats['fallback_loads'] += 1
            return self._load_fallback(file_path)
    
    def prefetch_model(self, file_path: str) -> bool:
        """
        Prefetch model to GPU memory cache (if supported)
        
        Args:
            file_path: Path to the model file
            
        Returns:
            True if prefetch was successful
        """
        if not self.config.prefetch_enabled or not self._gds_available:
            return False
        
        try:
            # Basic prefetch implementation
            # This would ideally use NVIDIA's GPUDirect Storage API
            # to warm up the storage cache
            
            file_size = self._get_file_size(file_path)
            logging.info(f"Prefetching {file_path} ({file_size // (1024*1024)} MB)")
            
            # Read file metadata to warm caches
            with open(file_path, 'rb') as f:
                # Read first and last chunks to trigger prefetch
                f.read(1024 * 1024)  # First 1MB
                f.seek(-min(1024 * 1024, file_size), 2)  # Last 1MB
                f.read()
            
            return True
            
        except Exception as e:
            logging.warning(f"Prefetch failed for {file_path}: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get loading statistics"""
        total_loads = self.stats['gds_loads'] + self.stats['fallback_loads']
        
        if self.stats['total_time_gds'] > 0 and self.stats['total_bytes_gds'] > 0:
            bandwidth_gbps = (self.stats['total_bytes_gds'] / (1024**3)) / self.stats['total_time_gds']
            self.stats['avg_bandwidth_gbps'] = bandwidth_gbps
        
        return {
            **self.stats,
            'total_loads': total_loads,
            'gds_usage_percent': (self.stats['gds_loads'] / max(1, total_loads)) * 100,
            'gds_available': self._gds_available,
            'config': self.config.__dict__
        }
    
    def cleanup(self):
        """Clean up GDS resources"""
        try:
            # Clear CUDA streams
            for stream in self.cuda_streams:
                stream.synchronize()
            self.cuda_streams.clear()
            
            # Free pinned buffers
            for buffer in self.pinned_buffers.values():
                if CUPY_AVAILABLE:
                    cupy.cuda.free_pinned_memory(buffer)
            self.pinned_buffers.clear()
            
            # Force garbage collection
            gc.collect()
            torch.cuda.empty_cache()
            
        except Exception as e:
            logging.warning(f"GDS cleanup failed: {e}")
    
    def __del__(self):
        """Destructor to ensure cleanup"""
        self.cleanup()


# Global GDS instance
_gds_instance: Optional[GPUDirectStorage] = None


def get_gds_instance(config: Optional[GDSConfig] = None) -> GPUDirectStorage:
    """Get or create the global GDS instance"""
    global _gds_instance
    
    if _gds_instance is None:
        _gds_instance = GPUDirectStorage(config)
    
    return _gds_instance


def load_torch_file_gds(ckpt: str, safe_load: bool = False, device: Optional[torch.device] = None) -> Dict[str, torch.Tensor]:
    """
    GDS-enabled replacement for comfy.utils.load_torch_file
    
    Args:
        ckpt: Path to checkpoint file
        safe_load: Whether to use safe loading (for compatibility)
        device: Target device
        
    Returns:
        Dictionary of loaded tensors
    """
    gds = get_gds_instance()
    
    try:
        # Load with GDS
        return gds.load_model(ckpt, device)
        
    except Exception as e:
        logging.error(f"GDS loading failed, falling back to standard method: {e}")
        # Fallback to original method
        import comfy.utils
        return comfy.utils.load_torch_file(ckpt, safe_load=safe_load, device=device)


def prefetch_model_gds(file_path: str) -> bool:
    """Prefetch model for faster loading"""
    gds = get_gds_instance()
    return gds.prefetch_model(file_path)


def get_gds_stats() -> Dict[str, Any]:
    """Get GDS statistics"""
    gds = get_gds_instance()
    return gds.get_stats()


def configure_gds(config: GDSConfig):
    """Configure GDS settings"""
    global _gds_instance
    _gds_instance = GPUDirectStorage(config)


def init_gds(config: GDSConfig):
    """
    Initialize GPUDirect Storage with the provided configuration
    
    Args:
        config: GDSConfig object with initialization parameters
    """
    try:
        # Configure GDS
        configure_gds(config)
        logging.info(f"GDS initialized: enabled={config.enabled}, min_size={config.min_file_size_mb}MB, streams={config.max_concurrent_streams}")
        
        # Set up exit handler for stats if requested
        if hasattr(config, 'show_stats') and config.show_stats:
            import atexit
            def print_gds_stats():
                stats = get_gds_stats()
                logging.info("=== GDS Statistics ===")
                logging.info(f"Total loads: {stats['total_loads']}")
                logging.info(f"GDS loads: {stats['gds_loads']} ({stats['gds_usage_percent']:.1f}%)")
                logging.info(f"Fallback loads: {stats['fallback_loads']}")
                logging.info(f"Total bytes via GDS: {stats['total_bytes_gds'] / (1024**3):.2f} GB")
                logging.info(f"Average bandwidth: {stats['avg_bandwidth_gbps']:.2f} GB/s")
                logging.info("===================")
            atexit.register(print_gds_stats)
            
    except ImportError as e:
        logging.warning(f"GDS initialization failed - missing dependencies: {e}")
    except Exception as e:
        logging.error(f"GDS initialization failed: {e}")