from openai import OpenAI
client = OpenAI()

completion = client.chat.completions.create(
  model="gpt-3.5-turbo-1106",
  messages=[
    {"role": "system", "content": "You are an Indian Official , good at translating multiple indian languages."},
    {"role": "user", "content": "இங்க ரோட்டுல குளியா இருக்கு, convert this to english"}
  ]
)
print(completion.choices[0].message.content)