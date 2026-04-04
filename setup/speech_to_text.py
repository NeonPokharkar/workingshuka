from faster_whisper import WhisperModel
import faster_whisper
from matplotlib import text
from matplotlib.pylab import rint
import torch
from kokoro_onnx import Kokoro
import soundfile as sf
# from TTS.api import TTS
import os, shutil
from fastapi import UploadFile, File
import uuid
import numpy as np
from pydub import AudioSegment

# --- KOKORO VOICE DIRECTORY (54 VOICES) ---

# [US English]
# Female: af_heart, af_bella, af_nicole, af_sarah, af_sky, af_alloy, af_nova, af_river
# Male:   am_adam, am_michael, am_echo, am_eric, am_liam, am_onyx, am_puck

# [UK English]
# Female: bf_emma, bf_isabella, bf_alice, bf_lily
# Male:   bm_george, bm_lewis, bm_daniel, bm_fable

# [Japanese]
# Female: jf_alpha, jf_gongitsune, jf_nezumi, jf_tebukuro, jf_urashima
# Male:   jm_kintaro

# [Mandarin Chinese]
# Female: zf_xiaobei, zf_xiaoni, zf_xiaoxiao, zf_xiaoyi
# Male:   zm_yunjian, zm_yunxi, zm_yunzhe, zm_zhongkui

# [Spanish]
# Female: ef_dora
# Male:   em_alex, em_santa

# [French]
# Female: ff_siwis

# [Hindi]
# Female: hf_alpha, hf_beta
# Male:   hm_omega

# [Italian]
# Female: if_sara
# Male:   im_nicola

# [Brazilian Portuguese]
# Female: pf_dora
# Male:   pm_alex, pm_santa


os.environ["ONNX_PROVIDER"] = "CUDAExecutionProvider"

device = "cuda" if torch.cuda.is_available() else "cpu"

# tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=True).to(device)

# def speakInVoice(text, language, source_voice_path, file_id):
#     tts.tts_to_file(
#         text=text,
#         speaker_wav=source_voice_path,
#         language=language,
#         file_path=f"audio/output_clone_{file_id}.wav"
#     )
#     audio = AudioSegment.from_wav(f"audio/output_clone_{file_id}.wav")
#     audio.export(f"audio/output_clone_{file_id}.mp3", format="mp3", bitrate="192k")

#     os.remove(f"audio/output_clone_{file_id}.wav")
#     print(f"Saved as audio/output_clone_{file_id}.mp3")

#     return f"audio/output_clone_{file_id}.mp3"

kokoro = Kokoro("kokoro/kokoro-v1.0.onnx", "kokoro/voices-v1.0.bin")

def speakInShukaVoice(text, voice_type):
    id = uuid.uuid4()
    samples, sample_rate = kokoro.create(text, voice=voice_type, speed=1.0)

    audio_data = (samples * 32767).astype(np.int16)

    audio_segment = AudioSegment(
        audio_data.tobytes(), 
        frame_rate=sample_rate,
        sample_width=audio_data.dtype.itemsize, 
        channels=1
    )

    audio_segment.export(f"audio/output_{id}.mp3", format="mp3", bitrate="192k")
    print(f"Saved as audio/output_{id}.mp3")

    return f"audio/output_{id}.mp3"

# faster_whisper.download_model("large-v3-turbo", output_dir="./whisper")

model = WhisperModel("./whisper")

def transcribe_wav(file: UploadFile = File(...)):
    id = uuid.uuid4()
    temp_path = f"audio/user_{id}.wav"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    convert_to_mp3(temp_path)
    temp_path = temp_path.replace(".wav", ".mp3")

    try:
        segments, _ = model.transcribe(temp_path, beam_size=5)
        return "".join([s.text for s in segments])
    finally:
        if os.path.exists(temp_path): os.remove(temp_path)

def convert_to_mp3(input_path):
    # 1. Load whatever the phone sent (M4A, WAV, etc.)
    audio = AudioSegment.from_file(input_path)
    
    # 2. Define the new MP3 path
    mp3_path = os.path.splitext(input_path)[0] + ".mp3"
    
    # 3. Export as MP3 (Mono, 128k is perfect for Whisper)
    audio.export(mp3_path, format="mp3", bitrate="128k", parameters=["-ac", "1"])
    
    return mp3_path