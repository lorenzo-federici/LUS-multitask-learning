import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import gc
import sys
import json
import time
import argparse
import pandas as pd
from pathlib import Path
import tensorflow as tf
from keras import backend as K
from datetime import datetime, date
from keras import mixed_precision

BASE_PATH = str(Path(__file__).resolve().parent.parent)
if BASE_PATH not in sys.path:
    sys.path.append(BASE_PATH)

# import custom lib
from setting_exp import SettingExp

def _config_accelleration(host_hw):
    # Set the device to GPU 
    cpu_device = tf.config.list_physical_devices('CPU')[0]
    gpu_devices = tf.config.list_physical_devices('GPU')

    if gpu_devices:
        # Limit GPU memory growth
        try:
            for gpu in gpu_devices:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

        device = gpu_devices[0]
        details = tf.config.experimental.get_device_details(device)
        gpu_name = details.get('device_name', 'Unknown')
        gpu_capability = float(".".join(map(str, details.get('compute_capability', '0'))))
        host_hw.update({'gpu_name': gpu_name, 'gpu_capability': gpu_capability})

        tf.config.set_visible_devices([device, cpu_device])
        tf.config.experimental.set_memory_growth(device, True)

        print(f'>> GPU {gpu_name} available (compute capability: {gpu_capability})')
        print(f'\t• Name: {gpu_name}')
        print(f'\t• Compute capability: {gpu_capability}')

    host_hw['device'] = device
    return host_hw

def _check_mixedp_and_xla(mp_xla, host_hw):
    try:
        device_type = host_hw['device'].device_type
        gpu_capability = host_hw['gpu_capability']

        if device_type == 'GPU' and gpu_capability >= 7.0:
            mp, xla = mp_xla
            if mp:
                # Mixed Precision 
                policy = mixed_precision.Policy("mixed_float16")
                mixed_precision.set_global_policy(policy)
                print(f'✔ mixed precision on')

            if xla:
                # XLA (Accelerated Linear Algebra)
                tf.config.optimizer.set_jit(True)
                print(f'✔ XLA on')
        else:
            print(f'✘ mixed precision and XLA off')
    except Exception as e:
        print(f'✘ error enabling Mixed Precision and XLA: {e}')

def main():
    parser = argparse.ArgumentParser(description="LUS Ordinal Classification")
    parser.add_argument("--exps_json", type=str, required=True, help="json file containing the experiments to be performed")
    parser.add_argument("--idx_exp", type=int, required=True, help="experiment index on json")
    parser.add_argument("--grid", action='store_true', help="active grid search")
    
    args = parser.parse_args()

    host_hw = {}
    mp_xla = (False, False)

    print('\n')
    print("~"*50)
    host_hw = _config_accelleration(host_hw)
    _check_mixedp_and_xla(mp_xla, host_hw)
    print("~"*50)

    setting_exp = SettingExp(BASE_PATH, args.exps_json, args.idx_exp, isGrid=args.grid, output_mode=(False,True))

    setting_exp.run()

if __name__ == "__main__":
    main()