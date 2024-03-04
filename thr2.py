from openai import OpenAI
import json
client = OpenAI()

# data_path = "test.jsonl"

# with open(data_path, 'r', encoding='utf-8') as f:
#     dataset = [json.loads(line) for line in f]

# print(dataset[1]['messages'][1])
# total = len(dataset)
# correct = 0
# wrong = 0
# for data in dataset:
# data = dataset[1]
completion = client.chat.completions.create(
  model="gpt-3.5-turbo-1106",
  messages=[
    # data['messages'][0],
{"role": "system", "content": "You are a data analyst, good at performing entity extraction"},
    {"role": "user", "content": "Problem: Tap is leaking at kalapatty road, perform entity extraction and give the result in json alone"}
    # data['messages'][1]
  ]
)
print(completion.choices[0].message.content)

#   if(completion.choices[0].message.content == data['messages'][2]['content']):
#       correct+=1
#   else:
#       wrong +=1 

# print(correct)
# print(wrong)

# print(correct//total)