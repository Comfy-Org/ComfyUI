#!/usr/bin/env python3
"""
Static Extensions Builder for ComfyUI

This script replicates ComfyUI's custom node web directory scanning logic
and uploads all web files to OSS with the same directory structure.

Usage:
    python build_static_extensions.py [--oss] [--output-dir ./static_extensions] [--clean]
"""

import os
import sys
import glob
import shutil
import argparse
import logging
import importlib.util
from pathlib import Path

# Add ComfyUI modules to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "comfy"))

import folder_paths

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


# OSS Configuration - Try bucket-specific endpoint
OSS_CONFIG = {
    'endpoint': 'https://oss-cn-shanghai.aliyuncs.com',  # Bucket-specific endpoint
    'access_key_id': '',
    'access_key_secret': '',
    'bucket_name': 'tusi-assets',
    'base_path': 'comfyui/howell5/static_extensions/',
}


def get_oss_client():
    """Initialize and return OSS client. Returns None if not configured."""
    try:
        # Check if OSS is configured
        if not all([OSS_CONFIG['endpoint'], OSS_CONFIG['access_key_id'], 
                   OSS_CONFIG['access_key_secret'], OSS_CONFIG['bucket_name']]):
            logger.warning("OSS not configured. Falling back to local mode.")
            return None
            
        # Try to import oss2 - if not available, fall back to local mode
        try:
            import oss2
            from oss2.credentials import StaticCredentialsProvider
        except ImportError:
            logger.warning("oss2 package not found. Install with: pip install oss2")
            return None
            
        # Initialize OSS client with V4 auth and region
        # Use StaticCredentialsProvider for direct key/secret
        credentials_provider = StaticCredentialsProvider(
            OSS_CONFIG['access_key_id'], 
            OSS_CONFIG['access_key_secret']
        )
        auth = oss2.ProviderAuthV4(credentials_provider)
        
        # Extract region from endpoint or use default
        region = "cn-shanghai"  # extracted from oss-cn-shanghai.aliyuncs.com
        
        bucket = oss2.Bucket(
            auth, 
            OSS_CONFIG['endpoint'], 
            OSS_CONFIG['bucket_name'], 
            region=region
        )
        
        # Test connection with a simple list operation
        logger.info(f"Testing OSS connection to {OSS_CONFIG['endpoint']}/{OSS_CONFIG['bucket_name']}")
        
        try:
            result = bucket.list_objects(max_keys=1)
            logger.info(f"OSS connection established to bucket: {OSS_CONFIG['bucket_name']}")
            return bucket
        except Exception as list_error:
            logger.error(f"OSS list test failed: {list_error}")
            logger.info("Trying without connection test...")
            # Return bucket anyway, let the upload operations handle errors
            return bucket
        
    except Exception as e:
        logger.warning(f"Failed to initialize OSS client: {e}. Falling back to local mode.")
        return None


def upload_to_oss(bucket, local_file_path, oss_key):
    """Upload file to OSS."""
    try:
        bucket.put_object_from_file(oss_key, local_file_path)
        return True
    except Exception as e:
        logger.error(f"Failed to upload {local_file_path} to OSS: {e}")
        return False


def get_module_name(module_path: str) -> str:
    """
    Returns the module name based on the given module path.
    Copied from nodes.py
    """
    base_path = os.path.basename(module_path)
    if os.path.isfile(module_path):
        base_path = os.path.splitext(base_path)[0]
    return base_path


def scan_custom_node_web_dirs():
    """
    Scan custom nodes and identify their web directories.
    Replicates the logic from nodes.py load_custom_node()
    """
    extension_web_dirs = {}
    
    # Get all custom node paths
    node_paths = folder_paths.get_folder_paths("custom_nodes")
    
    for custom_node_path in node_paths:
        if not os.path.exists(custom_node_path):
            continue
            
        possible_modules = os.listdir(os.path.realpath(custom_node_path))
        
        # Filter out __pycache__ and .disabled files
        if "__pycache__" in possible_modules:
            possible_modules.remove("__pycache__")
            
        for possible_module in possible_modules:
            module_path = os.path.join(custom_node_path, possible_module)
            
            # Skip non-python files that aren't directories
            if os.path.isfile(module_path) and os.path.splitext(module_path)[1] != ".py":
                continue
                
            # Skip disabled modules
            if module_path.endswith(".disabled"):
                continue
                
            module_name = get_module_name(module_path)
            logger.info(f"Scanning custom node: {module_name}")
            
            # Try to detect web directory
            web_dir = None
            
            # Method 1: Try to load module and check WEB_DIRECTORY attribute
            try:
                if os.path.isfile(module_path):
                    module_spec = importlib.util.spec_from_file_location(f"temp_{module_name}", module_path)
                    module_dir = os.path.split(module_path)[0]
                else:
                    init_file = os.path.join(module_path, "__init__.py")
                    if os.path.exists(init_file):
                        module_spec = importlib.util.spec_from_file_location(f"temp_{module_name}", init_file)
                        module_dir = module_path
                    else:
                        logger.warning(f"No __init__.py found in {module_path}, skipping")
                        continue
                
                if module_spec and module_spec.loader:
                    module = importlib.util.module_from_spec(module_spec)
                    module_spec.loader.exec_module(module)
                    
                    if hasattr(module, "WEB_DIRECTORY") and getattr(module, "WEB_DIRECTORY") is not None:
                        web_dir_name = getattr(module, "WEB_DIRECTORY")
                        web_dir = os.path.abspath(os.path.join(module_dir, web_dir_name))
                        logger.info(f"Found WEB_DIRECTORY: {web_dir}")
                        
            except Exception as e:
                logger.warning(f"Could not load module {module_path}: {e}")
                
            # Method 2: Try common web directory names if no WEB_DIRECTORY found
            if not web_dir or not os.path.isdir(web_dir):
                common_web_dirs = ["web", "js", "frontend", "static", "assets"]
                for web_dir_name in common_web_dirs:
                    potential_web_dir = os.path.join(module_path if os.path.isdir(module_path) else os.path.dirname(module_path), web_dir_name)
                    if os.path.isdir(potential_web_dir):
                        web_dir = potential_web_dir
                        logger.info(f"Found web directory by convention: {web_dir}")
                        break
                        
            # Register web directory if found
            if web_dir and os.path.isdir(web_dir):
                extension_web_dirs[module_name] = web_dir
                logger.info(f"Registered web directory for {module_name}: {web_dir}")
            else:
                logger.debug(f"No web directory found for {module_name}")
                
    return extension_web_dirs


def copy_web_files(extension_web_dirs, output_dir, file_patterns=None, use_oss=False):
    """
    Copy all web files from custom nodes to static extensions directory or upload to OSS.
    """
    if file_patterns is None:
        # Include common web file types
        file_patterns = ['**/*.js', '**/*.css', '**/*.html', '**/*.json', 
                        '**/*.png', '**/*.jpg', '**/*.jpeg', '**/*.gif', 
                        '**/*.svg', '**/*.ico', '**/*.woff', '**/*.woff2',
                        '**/*.ttf', '**/*.eot']
    
    # Initialize OSS client if requested
    oss_bucket = None
    if use_oss:
        oss_bucket = get_oss_client()
        if not oss_bucket:
            logger.warning("OSS upload failed to initialize, falling back to local mode")
            use_oss = False
    
    output_path = Path(output_dir)
    
    # Create output directory for local mode or as temp storage for OSS
    output_path.mkdir(parents=True, exist_ok=True)
    
    total_files_processed = 0
    total_files_uploaded = 0
    
    for module_name, web_dir in extension_web_dirs.items():
        logger.info(f"Processing {module_name} from {web_dir}")
        
        module_output_dir = output_path / module_name
        module_output_dir.mkdir(parents=True, exist_ok=True)
        
        files_processed = 0
        files_uploaded = 0
        
        # Process files matching patterns
        for pattern in file_patterns:
            files = glob.glob(os.path.join(glob.escape(web_dir), pattern), recursive=True)
            
            for file_path in files:
                # Calculate relative path from web_dir
                rel_path = os.path.relpath(file_path, web_dir)
                dest_path = module_output_dir / rel_path
                
                # Create parent directories
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                
                try:
                    if use_oss and oss_bucket:
                        # Upload to OSS
                        oss_key = f"{OSS_CONFIG['base_path']}{module_name}/{rel_path}"
                        if upload_to_oss(oss_bucket, file_path, oss_key):
                            files_uploaded += 1
                            logger.debug(f"Uploaded to OSS: {oss_key}")
                        else:
                            # Fallback to local copy if OSS upload fails
                            shutil.copy2(file_path, dest_path)
                            logger.debug(f"OSS upload failed, copied locally: {rel_path}")
                    else:
                        # Local copy
                        shutil.copy2(file_path, dest_path)
                        logger.debug(f"Copied locally: {rel_path}")
                    
                    files_processed += 1
                    
                except Exception as e:
                    logger.error(f"Failed to process {file_path}: {e}")
                    
        if use_oss and oss_bucket:
            logger.info(f"Processed {files_processed} files for {module_name} (uploaded to OSS: {files_uploaded})")
        else:
            logger.info(f"Copied {files_processed} files locally for {module_name}")
            
        total_files_processed += files_processed
        total_files_uploaded += files_uploaded
        
    return total_files_processed, total_files_uploaded


def clean_output_dir(output_dir):
    """Remove existing output directory."""
    if os.path.exists(output_dir):
        logger.info(f"Cleaning output directory: {output_dir}")
        shutil.rmtree(output_dir)


def build_static_extensions(output_dir="./static_extensions", clean=False, use_oss=False, verbose=False):
    """
    Build static extensions from custom nodes. Can be called programmatically.
    Returns (total_files_processed, total_files_uploaded) or None if no extensions found.
    """
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    output_dir = os.path.abspath(output_dir)
    
    logger.info("Starting static extensions build...")
    if use_oss:
        logger.info("OSS upload mode enabled")
    else:
        logger.info(f"Local output directory: {output_dir}")
    
    # Clean output directory if requested
    if clean:
        clean_output_dir(output_dir)
    
    # Scan custom nodes for web directories
    logger.info("Scanning custom nodes...")
    extension_web_dirs = scan_custom_node_web_dirs()
    
    if not extension_web_dirs:
        logger.warning("No custom nodes with web directories found!")
        return None
        
    logger.info(f"Found {len(extension_web_dirs)} custom nodes with web directories:")
    for module_name, web_dir in extension_web_dirs.items():
        logger.info(f"  - {module_name}: {web_dir}")
    
    # Process web files
    logger.info("Processing web files...")
    total_files_processed, total_files_uploaded = copy_web_files(extension_web_dirs, output_dir, use_oss=use_oss)
    
    if use_oss and total_files_uploaded > 0:
        logger.info(f"Build completed! Processed {total_files_processed} files, uploaded {total_files_uploaded} to OSS")
    else:
        logger.info(f"Build completed! Processed {total_files_processed} files to {output_dir}")
    
    # Generate summary
    logger.info("\nStatic extensions structure:")
    for module_name in extension_web_dirs.keys():
        if use_oss:
            logger.info(f"  - {OSS_CONFIG['base_path']}{module_name}/")
        else:
            module_dir = os.path.join(output_dir, module_name)
            if os.path.exists(module_dir):
                file_count = sum(len(files) for _, _, files in os.walk(module_dir))
                logger.info(f"  - /extensions/{module_name}/ ({file_count} files)")
    
    return total_files_processed, total_files_uploaded


def main():
    parser = argparse.ArgumentParser(description="Build static extensions from ComfyUI custom nodes")
    parser.add_argument("--output-dir", default="./static_extensions", 
                       help="Output directory for static extensions (default: ./static_extensions)")
    parser.add_argument("--clean", action="store_true", 
                       help="Clean output directory before building")
    parser.add_argument("--oss", action="store_true",
                       help="Upload files to OSS instead of local directory")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    build_static_extensions(
        output_dir=args.output_dir,
        clean=args.clean,
        use_oss=args.oss,
        verbose=args.verbose
    )


if __name__ == "__main__":
    main()