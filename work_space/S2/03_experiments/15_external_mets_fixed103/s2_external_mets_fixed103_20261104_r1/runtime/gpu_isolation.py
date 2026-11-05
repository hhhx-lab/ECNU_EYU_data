"""Check network and allocated CUDA identity before external checkpoint loading."""
import ctypes
import json
import os
from pathlib import Path
import uuid as uuid_module


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def runtime_uuid(torch):
    libraries = [*Path(torch.__file__).resolve().parent.glob('lib/libcudart*.so*'),
                 *Path(torch.__file__).resolve().parent.parent.glob('nvidia/cuda_runtime/lib/libcudart*.so*'),
                 *Path('/usr/local/cuda/lib64').glob('libcudart*.so*')]
    for library in libraries:
        try:
            function = ctypes.CDLL(str(library)).cudaDeviceGetUuid
        except (OSError, AttributeError):
            continue
        value = (ctypes.c_ubyte * 16)()
        function.argtypes = [ctypes.c_void_p, ctypes.c_int]
        function.restype = ctypes.c_int
        require(function(ctypes.byref(value), 0) == 0, 'CUDA runtime logical-device UUID query failed')
        return bytes(value).hex()
    raise RuntimeError('CUDA runtime UUID interface unavailable')


def device_identity():
    import torch
    require(os.environ.get('CUDA_VISIBLE_DEVICES'), 'allocated CUDA visibility missing')
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, 'exactly one allocated GPU required')
    properties = torch.cuda.get_device_properties(0)
    require('A100' in properties.name and properties.total_memory >= 35 * 1024**3, 'full A100 allocation required')
    # Torch properties follow CUDA_VISIBLE_DEVICES; the driver API enumerates all physical GPUs.
    value = getattr(properties, 'uuid', None)
    uuid = uuid_module.UUID(str(value).removeprefix('GPU-')).hex if value is not None else runtime_uuid(torch)
    return dict(uuid=uuid, name=properties.name, total_memory=properties.total_memory,
                visible_devices=os.environ['CUDA_VISIBLE_DEVICES'],
                device_order=os.environ.get('CUDA_DEVICE_ORDER', 'FASTEST_FIRST'), device_count=1)


def check_inside():
    from isolation_check import main as network_check
    network_check()
    observed = device_identity()
    expected = json.loads(os.environ['FULL_EXPECTED_GPU_IDENTITY'])
    require(observed == expected, 'host/container allocated GPU mapping differs')
    print(json.dumps(dict(status='allocated_GPU_and_model_isolation_pass', observed=observed)), flush=True)
    return observed
