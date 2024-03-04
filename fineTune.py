from openai import OpenAI
client = OpenAI()

client.fine_tuning.jobs.create(
  training_file="file-pNVFwCf8KqXLmOYV7RkY43CJ", 
  model="gpt-3.5-turbo-1106",
  validation_file="file-am4WgMBCjMVRePlsao3PrXif"
)