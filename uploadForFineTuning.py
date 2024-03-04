from openai import OpenAI

client = OpenAI()

client.files.create(
  file=open("validate2.jsonl", "rb"),
  purpose="fine-tune"
)