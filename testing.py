from openai import OpenAI
from moviepy.editor import VideoFileClip
client = OpenAI()

completion = client.chat.completions.create(
  model="gpt-3.5-turbo",
  messages=[
    {"role": "system", "content": "You are a language translator, skilled in converting english to tamil."},
    # {"role": "system", "content": "You are a good model , s"},
    {"role": "user", "content": "translate the statement to tamil , 'hi my name is shiva prakash' "}
  ]
)

print(completion.choices[0].message.content)