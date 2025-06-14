# Research Assistant

A research assistant that searches, downloads, embeds, summarizes, and queries research papers from arXiv and existing papers on your computer. No APIs. Leverages locally downloaded models using the OpenVINO library for searching arXiv, chatting with papers, and generating embeddings.

Works best on Intel CPU/GPU. Only CPU support for ARM platforms (no GPU).

### How to Use this Repo

Clone the repo
Create a virtual environment
Install dependencies using requirements.txt file
Modify the system_prompt.yaml file for your use case. Ex: neuroscience research assistant
Run the gradio interface using the command: `python chatbot.py`

<img width="1233" alt="Screenshot 2025-06-13 at 6 22 01 PM" src="https://github.com/user-attachments/assets/be55b22c-37da-45ff-9ce4-f7bfbbb2bc7f" />
