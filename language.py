from openai import OpenAI
client = OpenAI()

completion = client.chat.completions.create(
  model="gpt-4-0125-preview",
  messages=[
    {"role": "system", "content": "You are an Indian Official , good at translating multiple indian languages."},
    {"role": "user", "content": "இங்க கரண்டு பிரச்சினை இருக்கு, convert this to english"}
  ]
)
print(completion.choices[0].message.content)