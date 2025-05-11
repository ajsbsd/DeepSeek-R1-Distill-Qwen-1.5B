from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
import torch
import json
import logging
from threading import Thread

# Set up logging configuration for JSON Lines
json_lines_logger = logging.getLogger('json_lines_logger')
json_lines_logger.setLevel(logging.INFO)
json_lines_handler = logging.FileHandler('training_data.jnlp')
json_lines_logger.addHandler(json_lines_handler)

app = FastAPI()

# Load tokenizer and model
MODEL_NAME = "JetBrains/Mellum-4b-sft-python"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to("cuda" if torch.cuda.is_available() else "cpu")

@app.post("/generate")
async def generate_text(data: dict):
    prompt = data.get("prompt")
    if not prompt:
        json_lines_logger.error(json.dumps({"error": "Missing 'prompt' in request body"}))
        raise HTTPException(status_code=400, detail="Missing 'prompt' in request body")

    json_lines_logger.info(json.dumps({"prompt": prompt}))

    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    def generate():
        try:
            model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=True,
                temperature=0.3,
                top_k=50,
                pad_token_id=tokenizer.eos_token_id,
                streamer=streamer
            )
        except Exception as e:
            error_message = f"Generation error: {str(e)}"
            json_lines_logger.error(json.dumps({"error": error_message}))
            streamer.on_finalized_text(f"\n[{error_message}]")

    thread = Thread(target=generate)
    thread.start()

    def stream_response():
        buffer = ""
        full_response = ""
        for token in streamer:
            buffer += token
            full_response += token
            if len(buffer) > 10 or any(c in buffer for c in [' ', '\n', '.', ',', ';']):
                yield json.dumps({"text": buffer}) + "\n"
                buffer = ""
        if buffer:
            yield json.dumps({"text": buffer}) + "\n"
        thread.join()

        # Log the full response
        json_lines_logger.info(json.dumps({"response": full_response}))

    return StreamingResponse(stream_response(), media_type="application/json-lines")

