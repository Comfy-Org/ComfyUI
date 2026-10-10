import triton
import os

if os.environ.get('TRITON_AUTOTUNE_ENBALE', '0') == '1':
    autotune = triton.autotune
else:
    def autotune(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
