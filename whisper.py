from openai import OpenAI
from docx import Document

client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    # api_key="My API Key",
)

def transcribe_audio(audio_file_path):
    with open(audio_file_path, 'rb') as audio_file:
        transcription = client.audio.transcriptions.create(model="whisper-1", 
  file=audio_file,language="ta")
    print(transcription)
        # transcribe_audi = client.audio.transcriptions.create("",file=audio._file,model='whisper-1',language='english')
    return transcription.text


print(transcribe_audio("my_voice.wav"))