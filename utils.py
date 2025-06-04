import json
import os
import arxiv
from pathlib import Path
import openvino as ov
import gradio as gr

client = arxiv.Client()

def get_available_devices():
    """Get available devices for OpenVINO."""
    core = ov.Core()
    return {device.split(".")[0] for device in core.available_devices}

def save_arxiv_ids(ids): 
    
    with open("arxiv_ids.json",'w') as f: 
        json.dump(ids, f)

def open_arxiv_ids():
    
    try: 
        with open("arxiv_ids.json",'r') as f:
            ids = list(json.load(f))
        return ids
    
    except: 
        print("file not created")
        return []
    
def search_arxiv(query, max_results):
    results = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance
    ).results()

    papers = [{"title": result.title, "id": result.entry_id.split('/')[-1]} for result in results]
    output = "\n"
    for index, paper in enumerate(papers): 
        output += f"{index + 1}. {paper['title']}  ArXiv ID:{paper['id']}\n"
    return output

def download_papers(arxiv_ids, output_dir):
    try: 
        os.mkdir(output_dir)
    
    except Exception as e: 
        print(f"An error occured: {e}")
    
    for paper_id in arxiv_ids:
        pdf_path = os.path.join(output_dir, f"{paper_id}.pdf")
    
        if os.path.exists(pdf_path):
            gr.Info(f"Skipping paper... already exists: {paper_id}.pdf")
            continue
    
        try: 
            search = arxiv.Search(id_list = [paper_id])
            paper = next(client.results(search))
            
            # Download the PDF to the directory
            paper.download_pdf(dirpath=output_dir, filename=f"{paper_id}.pdf")
        
        except Exception as e:
            print(f"An error occured: {e}")

def phi_completion_to_prompt(completion):
    return f"<|system|><|end|><|user|>{completion}<|end|><|assistant|>\n"


def llama3_completion_to_prompt(completion):
    return f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{completion}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

def completion_to_prompt(completion):
    return f"<|system|>\n</s>\n<|user|>\n{completion}</s>\n<|assistant|>\n"




    

