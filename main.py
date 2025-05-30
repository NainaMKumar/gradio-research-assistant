import os
from pathlib import Path
import re
import yaml
import numpy as np
import pickle

import huggingface_hub as hf_hub
from huggingface_hub import list_models

import openvino.properties as props
import openvino.properties.hint as hints
import openvino.properties.streams as streams

from llama_index.llms.openvino import OpenVINOLLM
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import FunctionTool
from llama_index.core.memory import ChatMemoryBuffer

from llama_index.embeddings.huggingface_openvino import OpenVINOEmbedding
from llama_index.core import SimpleDirectoryReader
from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.tools import QueryEngineTool, ToolMetadata


from utils import search_arxiv, download_papers, completion_to_prompt
from paths import Assit_paths

import streamlit as st

system_prompt_path = "system_prompt.yaml"

existing_papers_path = Path("existing_papers")
existing_papers_path.mkdir(exist_ok=True)

output_dir = Path("arxiv_pdfs")
existing_papers_path.mkdir(exist_ok=True)

embedding_cache =  Path("embedding_cache")
existing_papers_path.mkdir(exist_ok=True)

llm_device = "CPU"
ov_config = {hints.performance_mode(): hints.PerformanceMode.LATENCY, streams.num(): "1", props.cache_dir(): ""}

ov_llm = None
ov_embedding = None

def load_cached_embeddings():
    all_nodes = []
    for cache_file in Path(embedding_cache).iterdir():
        with open(cache_file, "rb") as f:
            nodes = pickle.load(f)
            all_nodes.extend(nodes)

    if all_nodes:
        index = VectorStoreIndex(
            nodes=all_nodes, 
            embed_model=ov_embedding,
        )
        return index
    return None

def generate_embeddings(file_path, output_dir): 
    
    try: 

        all_nodes = []

        if output_dir is not None:
            files = list(Path(output_dir).iterdir())
        else:
            files = [Path(file_path)]
        
        for file in files:
            cache_path = Path(embedding_cache) / file.stem

            if cache_path.exists():
                st.write(f"Loading cached embeddings for {file.stem}")
                
                with open(cache_path, "rb") as f:
                    nodes = pickle.load(f)

            else:
                st.write(f"Generating embeddings for {file.stem}")
                reader = SimpleDirectoryReader(input_files=[str(file)])
                documents = reader.load_data()

                splitter = SentenceSplitter(chunk_size=512, chunk_overlap=20)
                nodes = splitter.get_nodes_from_documents(documents)

                with open(cache_path, "wb") as f:
                    pickle.dump(nodes, f)
            
            all_nodes.extend(nodes)
        
        if all_nodes:
            index = VectorStoreIndex(
            nodes=all_nodes, 
            embed_model=ov_embedding,
            # transformations=[SentenceSplitter(chunk_size=1024, chunk_overlap=20)],
        )

            return index

                
    except Exception as e:
        st.error(f"Embedding generation failed: {e}")  



def arxiv_query(input: str, **kwargs) -> str: 
    """Finds research papers on arxiv based on query."""

    results = search_arxiv(input)
    return results
    
def load_chat_model(model_type):
    model_name = model_type.split("/")[-1]
    model_path = Assit_paths.models(model_name)
    
    if not os.path.exists(model_path):
        try:
            hf_hub.snapshot_download(repo_id=model_type, local_dir=str(model_path))
        except Exception as e:
            st.error(f"Download failed: {e}")
            st.stop()
    
    else: 
        try: 
            st.write(f"Model '{model_name}' already downloaded.")

            llm = OpenVINOLLM(
            model_id_or_path=str(model_type),
            context_window=3900,
            max_new_tokens=2048,
            model_kwargs={"ov_config": ov_config},
            generate_kwargs={"do_sample": False, "temperature": None, "top_p": None},
            completion_to_prompt=completion_to_prompt,
            device_map=llm_device
        )
            Settings.llm = llm
            return llm

        except Exception as e: 
            st.error(f"Failed to load LLM: {e}")
            st.stop()

def load_embedding_model():

    model_path = Assit_paths.models(chat_model_name = None)
    
    if not os.path.exists(model_path):
        OpenVINOEmbedding.create_and_save_openvino_model(
            "BAAI/bge-small-en-v1.5", str(model_path)
        )

    embedding = OpenVINOEmbedding(model_id_or_path=str(model_path), device="cpu")
    Settings.embed_model = embedding
    
    return embedding

def get_vector_tool():

    if st.session_state.index is None:
        return "Vector index not yet initialized"
    
    return QueryEngineTool(
        st.session_state.index.as_query_engine(streaming=True),
        metadata=ToolMetadata(
            name="vector_search",
            description=
                "Use this tool to ANSWER detailed technical questions using the PDFs already embedded. "
                "Best for summarizing, explaining, or analyzing existing papers."
            )
        )

#Set Model Type

format_type = "int4"
models = list_models(author="OpenVINO")
models_list = [model.modelId for model in models]
model_type_options = np.sort([model for model in models_list if format_type in model])
model_type = st.selectbox("Model Type", model_type_options) #ex. "OpenVINO/Qwen/2-1.5B-Instruct-int4-ov"

if st.checkbox(f"Download model '{model_type}'?"):
    try:
        ov_llm = load_chat_model(model_type)
        ov_embedding = load_embedding_model()

    except Exception as e: 
        st.error(f"Error loading models: {e}")
    
#Set up streamlit
st.title("Neuroscience Research Assistant")

#Set up index
if "index" not in st.session_state: 
    st.session_state.index = load_cached_embeddings()


arxiv_tool = FunctionTool.from_defaults(fn=arxiv_query, 
                                        name = "arxiv_query", 
                                        description = 
                                        "Use this tool ONLY when the user asks you to find new research papers, search arXiv, or look for new literature. "
                                        "DO NOT use this tool to answer questions about already uploaded PDFs."
                                       )
tools = [arxiv_tool]
if st.session_state.index is not None and "vector_tool" not in st.session_state:
    st.session_state.vector_tool = get_vector_tool()
    tools.append(st.session_state.vector_tool)

with open(system_prompt_path, "rb") as f:
    system_prompt = yaml.safe_load(f)
    
#Set up agent
if "agent" not in st.session_state: 
    st.session_state.agent = ReActAgent.from_tools(tools=tools, llm=ov_llm, verbose=True, system_prompt = system_prompt, max_iterations=7)
    
if "messages" not in st.session_state.keys(): # Initialize the chat message history
    st.session_state.messages = [
        {"role": "assistant", "content": "Ask me to find and download research papers and summarize them!"}
    ]

uploaded_files = st.file_uploader(
    "Choose a PDF file", accept_multiple_files=True
)

if uploaded_files:
    for uploaded_file in uploaded_files:
        file_path = existing_papers_path / uploaded_file.name
        st.session_state.index = generate_embeddings(file_path = file_path, output_dir = None)


if prompt := st.chat_input("Your question"): # Prompt for user input and save to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages: # Display the prior chat messages
    with st.chat_message(message["role"]):
        st.write(message["content"])

# If last message is not from assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
                
            response = st.session_state.agent.chat(prompt)

            #check if response contains arxiv IDs
            
            # arxiv_ids = re.findall(r"arXiv ID:\s(.+)", response.response, re.IGNORECASE)
            arxiv_ids = re.findall(r"(?i)\barxiv(?:\s*ID)?\s*:\s*([0-9]{4}\.[0-9]{4,5}(?:v\d+)?)", response.response, re.IGNORECASE)
                
            if arxiv_ids: 
                
                #Download papers
                download_papers(arxiv_ids, output_dir = output_dir)
                st.info("Download complete!")

                #Generate embeddings
                index = generate_embeddings(file_path = None, output_dir = output_dir)
            
                #Insert into index
                if index:

                    st.session_state.index = index

            st.write(response.response)
            message = {"role": "assistant", "content": response.response}
            st.session_state.messages.append(message) # Add response to message history

