#!/usr/bin/env python3
"""
List all available audio devices to help with device selection.
Run this to see which devices you have available.
"""

import sounddevice as sd

devices = sd.query_devices()
print("\n" + "="*80)
print("AVAILABLE AUDIO DEVICES ON YOUR SYSTEM")
print("="*80 + "\n")

for idx, dev in enumerate(devices):
    in_ch = dev["max_input_channels"]
    out_ch = dev["max_output_channels"]
    
    device_type = []
    if in_ch > 0:
        device_type.append(f"INPUT ({in_ch}ch)")
    if out_ch > 0:
        device_type.append(f"OUTPUT ({out_ch}ch)")
    
    device_str = " | ".join(device_type) if device_type else "UNKNOWN"
    
    print(f"[{idx}] {dev['name']}")
    print(f"    {device_str}")
    print()

print("="*80)
print("\n💡 SOLUTIONS TO REDUCE SPEAKER->MIC FEEDBACK:\n")
print("1. BEST: Use separate devices")
print("   - USB microphone + built-in speakers")
print("   - OR headphones + built-in microphone")
print("   - Edit io_devices.py to specify device indices\n")

print("2. GOOD: Use loopback device (if available)")
print("   - Look for 'Loopback', 'BlackHole', 'VB-Cable'")
print("   - These are virtual audio drivers that avoid feedback\n")

print("3. OKAY: Audio settings")
print("   - Lower speaker volume")
print("   - Move mic away from speaker")
print("   - Use microphone with directional pickup pattern\n")

default_in, default_out = sd.default.device
print(f"\nDefault Input Device: [{default_in}] {devices[default_in]['name']}")
print(f"Default Output Device: [{default_out}] {devices[default_out]['name']}\n")
