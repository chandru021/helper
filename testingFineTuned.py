from openai import OpenAI
import json
client = OpenAI()

data_path = "test.jsonl"

with open(data_path, 'r', encoding='utf-8') as f:
    dataset = [json.loads(line) for line in f]

print(dataset[1]['messages'][1])
total = len(dataset)
correct = 0
wrong = 0
for data in dataset:
# data = dataset[1]
  completion = client.chat.completions.create(
    model="ft:gpt-3.5-turbo-1106:personal::8yZyI2VO",
    messages=[
      data['messages'][0],
  # {"role": "system", "content": "You are an Indian government servant who helps to classify which department should handle the problem of a person."},
      # {"role": "user", "content": "Problem: A guy misbehaved with me"}
      data['messages'][1]
    ]
  )

  if(completion.choices[0].message.content == data['messages'][2]['content']):
      correct+=1
  else:
      wrong +=1 

print(correct)
print(wrong)

print(correct//total)