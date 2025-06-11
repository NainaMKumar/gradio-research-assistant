import os
from pathlib import Path
import re
import yaml
import numpy as np
import pickle
import argparse
import json
import faiss

import huggingface_hub as hf_hub
from huggingface_hub import list_models
import openvino as ov
import openvino.properties as props
import openvino.properties.hint as hints
import openvino.properties.streams as streams

from llama_index.llms.openvino import OpenVINOLLM
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import FunctionTool
from llama_index.core.memory import ChatMemoryBuffer

from llama_index.embeddings.huggingface_openvino import OpenVINOEmbedding
from llama_index.core import SimpleDirectoryReader, load_index_from_storage
from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.core.storage import StorageContext

import gradio as gr

from utils import search_arxiv, download_papers, completion_to_prompt
from paths import Assit_paths

system_prompt_path = "system_prompt.yaml"
keywords_path = "keywords.json"

existing_papers_path = Path("existing_papers")
existing_papers_path.mkdir(exist_ok=True)

output_dir = Path("arxiv_pdfs")
output_dir.mkdir(exist_ok=True)

embedding_cache =  Path("embedding_cache")
embedding_cache.mkdir(exist_ok=True)

memory = ChatMemoryBuffer.from_defaults(token_limit=2048)
ov_config = {hints.performance_mode(): hints.PerformanceMode.LATENCY, streams.num(): "1", props.cache_dir(): ""}
ov_llm = None
ov_embedding = None

index = None
agent = None
chat_engine = None

def load_system_prompt():
    with open(system_prompt_path, "r") as f:
        system_prompt = yaml.safe_load(f)

    return system_prompt

def get_available_devices():
    """Get available devices for OpenVINO."""
    core = ov.Core()
    return {device.split(".")[0] for device in core.available_devices}

def load_cached_embeddings():

    # load index from disk
    vector_store_path = embedding_cache/"default__vector_store.json"
    
    if vector_store_path.exists():
        vector_store = FaissVectorStore.from_persist_path(str(vector_store_path))
        storage_context = StorageContext.from_defaults(
            vector_store=vector_store, persist_dir=embedding_cache
        )
        index = load_index_from_storage(storage_context=storage_context)

        return index
    
    return None

def generate_embeddings(file_path, output_dir): 
    
    try: 

        all_nodes = []

        if file_path:
            gr.Info(f"Generating embeddings for {Path(file_path).stem}")
            reader = SimpleDirectoryReader(input_files=[str(Path(file_path))])

        else:
            gr.Info(f"Generating embeddings for {output_dir}")
            reader = SimpleDirectoryReader(input_dir = output_dir)
        
        documents = reader.load_data()
        splitter = SentenceSplitter(chunk_size=512, chunk_overlap=20)
        nodes = splitter.get_nodes_from_documents(documents)
        all_nodes.extend(nodes)
        
        if all_nodes:

            # Get embedding dimension
            dim = ov_embedding._model.request.outputs[0].get_partial_shape()[2].get_length()

            # Create FAISS index
            faiss_index = faiss.IndexFlatL2(dim)
            vector_store = FaissVectorStore(faiss_index=faiss_index)
            storage_context = StorageContext.from_defaults(vector_store=vector_store)


            index = VectorStoreIndex(
                nodes=all_nodes,
                embed_model=ov_embedding,
                storage_context=storage_context
            )

            index.storage_context.persist(persist_dir=embedding_cache)

            return index

                
    except Exception as e:
        raise gr.Error(f"Embedding generation failed: {e}")  

def arxiv_query(input: str, **kwargs) -> str: 
    """Finds research papers on arxiv based on query."""

    results = search_arxiv(input, max_results = 10)
    return results
    
def load_chat_model(model_type):
    model_name = model_type.split("/")[-1]
    model_path = Assit_paths.models(model_name)
    
    if not os.path.exists(model_path):
        try:
            hf_hub.snapshot_download(repo_id=model_type, local_dir=str(model_path))
        except Exception as e:
            raise gr.Error(f"Download failed: {e}")
    
    else: 
        try: 
            gr.Info(f"Model '{model_name}' already downloaded.")

            llm = OpenVINOLLM(
            model_id_or_path=str(model_type),
            context_window=3900,
            max_new_tokens=2048,
            model_kwargs={"ov_config": ov_config},
            generate_kwargs={"do_sample": False, "temperature": None, "top_p": None},
            completion_to_prompt=completion_to_prompt,
            device_map="GPU" if "GPU" in get_available_devices() else "CPU" 
        )
            Settings.llm = llm
            return llm

        except Exception as e: 
            raise gr.Error(f"Failed to load LLM: {e}")

def load_embedding_model():

    model_path = Assit_paths.models(chat_model_name = None)
    
    if not os.path.exists(model_path):
        OpenVINOEmbedding.create_and_save_openvino_model(
            "BAAI/bge-small-en-v1.5", str(model_path)
        )

    device = "GPU" if "GPU" in get_available_devices() else "CPU" 
    embedding = OpenVINOEmbedding(model_id_or_path=str(model_path), device=device)
    Settings.embed_model = embedding
    
    return embedding

def initialize_chat_engine():
    
    global chat_engine

    if index is None:
        return None
    
    chat_engine = index.as_chat_engine(llm=ov_llm, chat_mode="context", system_prompt=load_system_prompt(), memory=memory)


def find_keywords(message): 

    with open(keywords_path, "r") as f:
        keywords = json.load(f)["keywords"]

    if any(keyword in message for keyword in keywords):
        return agent.chat(message)

    else:
        return chat_engine.chat(message)

def initialize_agent():

    global agent
    
    arxiv_tool = FunctionTool.from_defaults(fn=arxiv_query, 
                                        name = "arxiv_query", 
                                        description = 
                                        "Use this tool ONLY when the user asks you to find new research papers, search arXiv, or look for new literature. "
                                        "DO NOT use this tool to answer questions about already uploaded PDFs."
                                       )

    agent = ReActAgent.from_tools(tools=[arxiv_tool], llm=ov_llm, verbose=True, system_prompt = load_system_prompt(), max_iterations=7)

def process_input(message, history, pdfs):

    global index

    if pdfs is not None:
        for pdf in pdfs:
            file_path = existing_papers_path / pdf.name
            index = generate_embeddings(file_path = file_path, output_dir = None)
        
    response = find_keywords(message)

    # arxiv_ids = re.findall(r"arXiv ID:\s(.+)", response.response, re.IGNORECASE)
    arxiv_ids = re.findall(r"(?i)\barxiv(?:\s*ID)?\s*:\s*([0-9]{4}\.[0-9]{4,5}(?:v\d+)?)", response.response, re.IGNORECASE)
    print(arxiv_ids)
    arxiv_url = re.findall(r'arxiv\.org/abs/(\d{4}\.\d{4,5}(?:v\d+)?)', response.response, re.IGNORECASE)
    print(arxiv_url)
                
    if arxiv_ids or arxiv_url:  

        if arxiv_ids: 
            download_papers(arxiv_ids, output_dir = output_dir)
        else:
            download_papers(arxiv_url, output_dir = output_dir)

        #Generate embeddings
        new_index = generate_embeddings(file_path = None, output_dir = output_dir)
            
        #Insert into index
        if new_index:
            index = new_index
            initialize_chat_engine()
    
    return response.response


def add_user_message(message, history):
    history.append({"role": "user", "content": message})
    return "", history

def respond(_, history, pdfs):

    message = history[-1]["content"]
    response = process_input(message, history, pdfs)
    history.append({"role": "assistant", "content": response})
    return history

def create_UI():
    with gr.Blocks() as demo:
        
        gr.Markdown("# Neuroscience Research Assistant")

        chatbot = gr.Chatbot(type="messages", height=400)
        with gr.Column():
            msg = gr.Textbox(label="Your question", scale=4)
            pdfs = gr.File(file_types=[".pdf"], file_count="multiple", label="Upload PDFs")
            # model_type = gr.Dropdown(choices=model_type_options, label="Model Type", scale=2)

        msg.submit(
            fn=add_user_message,
            inputs=[msg, chatbot],
            outputs=[msg, chatbot]
        ).then(
            fn=respond,
            inputs=[msg, chatbot, pdfs],
            outputs=chatbot
        )
        
        return demo

def run(model_type, is_public, local_network):
    global ov_llm, ov_embedding, index

    #initialize once
    ov_llm = load_chat_model(model_type)
    ov_embedding = load_embedding_model()
    index = load_cached_embeddings()
    initialize_agent()
    initialize_chat_engine()

    demo = create_UI()
    demo.launch(server_name="0.0.0.0" if local_network else None, share=is_public, inbrowser=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--chat_model", type=str, default="OpenVINO/Qwen2-1.5B-Instruct-int4-ov")
    parser.add_argument("--public", default=False, action="store_true")
    parser.add_argument("--local_network", default=False, action="store_true")
    args = parser.parse_args()

    run(args.chat_model, args.public, args.local_network)



    
