#!/bin/bash

# Usage: ./chunk_sender.sh "your input text here" [chunk_size]

INPUT_TEXT="$1"
CHUNK_SIZE="${2:-50}"  # Default chunk size is 50 characters

# If first argument is "-", read from stdin
if [ "$1" = "-" ]; then
    INPUT_TEXT=$(cat)
fi

# Check if input text is provided
if [ -z "$INPUT_TEXT" ]; then
    echo "Usage: $0 \"input text\" [chunk_size]"
    echo "   or: echo \"text\" | $0 - [chunk_size]"
    exit 1
fi

# Get length of input text
TEXT_LENGTH=${#INPUT_TEXT}

echo "Input text length: $TEXT_LENGTH characters"
echo "Chunk size: $CHUNK_SIZE characters"
echo "---"

# Loop through text in chunks
for ((i=0; i<TEXT_LENGTH; i+=CHUNK_SIZE)); do
    # Extract chunk
    CHUNK="${INPUT_TEXT:$i:$CHUNK_SIZE}"
    
    # Calculate chunk number
    CHUNK_NUM=$((i/CHUNK_SIZE + 1))
    
    echo "Sending chunk $CHUNK_NUM: \"$CHUNK\""
    
    # Send chunk via curl
    curl -X POST "http://localhost:8000/generate" \
         -H "Content-Type: application/json" \
         -d "{\"prompt\": \"$CHUNK\"}" \
         --silent \
         --show-error
    
    echo -e "\n---"
    
    # Optional: Add delay between requests
    # sleep 1
done

echo "All chunks sent!"
