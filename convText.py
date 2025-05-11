import json

def convert_file_to_jnlp(input_file_path, output_file_path, chunk_size=1000):
    # Read the content of the input file
    with open(input_file_path, 'r', encoding='utf-8') as input_file:
        input_text = input_file.read()

    # Split the input text into chunks of the specified size
    chunks = [input_text[i:i + chunk_size] for i in range(0, len(input_text), chunk_size)]

    # Write each chunk as a separate line in the JSON Lines file
    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        for i, chunk in enumerate(chunks):
            # Create a JSON object for each chunk
            data = {
                "id": i + 1,
                "text": chunk
            }
            # Write the JSON object as a line in the file
            output_file.write(json.dumps(data, ensure_ascii=False) + '\n')

# Example usage
input_file_path = 'input.txt'  # Replace with your input file path
output_file_path = 'output.jnlp'
convert_file_to_jnlp(input_file_path, output_file_path)

