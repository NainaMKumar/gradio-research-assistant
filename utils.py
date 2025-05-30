import json
import os
import arxiv
from pathlib import Path
import streamlit as st

client = arxiv.Client()

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
    
def search_arxiv(query, max_results = 5):
    results = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance
    ).results()

    return [
        {"title": result.title, "id": result.entry_id.split('/')[-1]}
        for result in results
    ]

"""uncomment if you want to store IDs in json and avoid duplicates"""
# def search_arxiv(query, max_results = 5): 

#     saved_ids = open_arxiv_ids()
    
#     search = arxiv.Search(
#         query = query, 
#         max_results = max_results, 
#         sort_by = arxiv.SortCriterion.Relevance
#     )

#     results = []

#     for r in client.results(search): 

#         arxiv_id = r.entry_id.split('/')[-1]

#         if arxiv_id in saved_ids: 
#             continue 
            
#         results.append({
#             "title": r.title,
#             "id": arxiv_id,
#             # "authors": [author.name for author in r.authors],
#             # "abstract": r.summary,
#             # "pdf_url": r.pdf_url
#         })

#         saved_ids.append(arxiv_id)
        
#     save_arxiv_ids(saved_ids)
        

#     return results #returns empty list if nothing gets added 



def download_papers(arxiv_ids, output_dir):
    try: 
        os.mkdir(output_dir)
    
    except Exception as e: 
        print(f"An error occured: {e}")
    
    for paper_id in arxiv_ids:
        pdf_path = os.path.join(output_dir, f"{paper_id}.pdf")
    
        if os.path.exists(pdf_path):
            st.write(f"Skipping paper... already exists: {paper_id}.pdf")
            continue
    
        try: 
            search = arxiv.Search(id_list = [paper_id])
            paper = next(client.results(search))
            
            # Download the PDF to the directory
            paper.download_pdf(dirpath=output_dir, filename=f"{paper_id}.pdf")
        
        except Exception as e:
            st.write(f"An error occured: {e}")

def phi_completion_to_prompt(completion):
    return f"<|system|><|end|><|user|>{completion}<|end|><|assistant|>\n"


def llama3_completion_to_prompt(completion):
    return f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{completion}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

def completion_to_prompt(completion):
    return f"<|system|>\n</s>\n<|user|>\n{completion}</s>\n<|assistant|>\n"




    

