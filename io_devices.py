import sounddevice as sd

def get_sound_devices(disable_loopback=True):
    devices = sd.query_devices()
    default_input, default_output = sd.default.device

    mic_device = None
    speaker_device = None

    loopback_keywords = ["loopback", "monitor", "blackhole", "virtual", "what u hear"]

    def is_loopback(dev):
        return any(k in dev["name"].lower() for k in loopback_keywords)

    for idx, dev in enumerate(devices):
        if dev["max_input_channels"] > 0:
            if disable_loopback and is_loopback(dev):
                continue
            mic_device = idx
            break

    for idx, dev in enumerate(devices):
        name = dev["name"].lower()
        if dev["max_output_channels"] > 0 and any(k in name for k in ["headphone", "speaker", "output"]):
            if disable_loopback and is_loopback(dev):
                continue
            speaker_device = idx
            break

    if mic_device is None:
        mic_device = default_input
    if speaker_device is None:
        speaker_device = default_output

    print(f"\nSelected Audio Devices:")
    print(f"   Microphone: {devices[mic_device]['name']} (index {mic_device})")
    print(f"   Speaker: {devices[speaker_device]['name']} (index {speaker_device})\n")

    return mic_device, speaker_device


mic_device, speaker_device = get_sound_devices()
devices = sd.query_devices()
mic_device_name = devices[mic_device]["name"]
speaker_device_name = devices[speaker_device]["name"]