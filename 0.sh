python -m venv .
source /home/ai/bin/activate
git clone https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct
git branch -M main
git push -u origin main
git branch -M main
git config --global user.email "ajsbsd@gmail.com"
git config --global user.name "ajsbsd@gmail.com"
git push origin main
pip install --upgrade transformers
pip install --upgrade pip
pip install --upgrade torch
pip install --upgrade accelerate
