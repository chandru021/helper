import csv
import json
from sklearn.model_selection import train_test_split

# Path to the CSV file containing problem statements and departments
input_file = "datasetfinal.csv"

# Read dataset from CSV file
dataset = []
with open(input_file, mode='r') as file:
    csv_reader = csv.reader(file)
    for row in csv_reader:
        dataset.append(row)





train_data, test_validate = train_test_split(dataset, test_size=0.3, random_state=42)


test_data, validate_data = train_test_split(test_validate, test_size=0.5, random_state=42)


# Function to format data into JSON
def format_data(data):
    formatted_data = []
    for problem, department in data:
        formatted_data.append({
            "messages": [
                {"role" : "system" , "content" : "You are an Indian government servant who helps to classify which department should handle the problem of a person."},
                {"role": "user", "content": f"Problem: {problem}"},
                {"role": "assistant", "content": f"Department: {department}"}
            ]
        })
    return formatted_data

# Format train and test data
train_formatted_data = format_data(train_data)
test_formatted_data = format_data(test_data)
validate_formatted_data = format_data(validate_data)

# Write the formatted data to JSON Lines (JSONL) files
train_output_file = "train.jsonl"
test_output_file = "test.jsonl"
validate_output_file="validate.jsonl"

with open(train_output_file, "w") as train_jsonl_file:
    for data in train_formatted_data:
        json.dump(data, train_jsonl_file)
        train_jsonl_file.write('\n')  # Add a newline after each JSON object

with open(test_output_file, "w") as test_jsonl_file:
    for data in test_formatted_data:
        json.dump(data, test_jsonl_file)
        test_jsonl_file.write('\n')  # Add a newline after each JSON object

with open(validate_output_file, "w") as validate_output_file:
    for data in validate_formatted_data:
        json.dump(data, validate_output_file)
        validate_output_file.write('\n') 

print(f"Formatted train data has been saved to {train_output_file}")
print(f"Formatted test data has been saved to {test_output_file}")
