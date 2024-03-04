import sounddevice as sd
import soundfile as sf

def record_audio(filename, duration, samplerate=44100, channels=1):
    print("Recording...")
    audio_data = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=channels, dtype='int16')
    sd.wait()
    sf.write(filename, audio_data, samplerate)

if __name__ == "__main__":
    output_filename = "my_voice.wav"
    recording_duration = 5  # in seconds
    record_audio(output_filename, recording_duration)
    print(f"Recording saved as '{output_filename}'")