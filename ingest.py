import os
import json
from datetime import datetime
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()

def get_file_metadata(path):
    return {
        "path": path,
        "mtime": os.path.getmtime(path) if os.path.exists(path) else None,
        "source_type": "external" if os.path.isabs(path) else "local"
    }

def ingest_notes():
    documents = []
    metadata_file = "external_files.json"
    
    # Read local notes directory
    local_files = []
    for root, _, files in os.walk("notes"):
        for file in files:
            if file.endswith(".txt"):
                path = os.path.join(root, file)
                local_files.append(get_file_metadata(path))

    # Read external files from JSON
    external_files = []
    try:
        with open(metadata_file, "r") as f:
            external_entries = json.load(f)
            for entry in external_entries:
                if os.path.exists(entry["path"]) and entry["path"].endswith(".txt"):
                    external_files.append(get_file_metadata(entry["path"]))
    except FileNotFoundError:
        pass

    # Process all files
    all_files = local_files + external_files
    for file_meta in all_files:
        try:
            with open(file_meta["path"], "r", encoding="utf-8") as f:
                documents.append(Document(
                    page_content=f.read(),
                    metadata={
                        "source": file_meta["path"],
                        "source_type": file_meta["source_type"],
                        "mtime": file_meta["mtime"]
                    }
                ))
        except Exception as e:
            print(f"Skipping {file_meta['path']} due to error: {str(e)}")

    # Split and vectorize documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(documents)
    
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local("vector_store")
    
    # Save ingestion metadata
    ingestion_data = {
        "local_files": [f["path"] for f in local_files],
        "external_files": [f["path"] for f in external_files],
        "timestamp": datetime.now().isoformat()
    }
    with open(os.path.join("vector_store", "metadata.json"), "w") as f:
        json.dump(ingestion_data, f)

if __name__ == "__main__":
    ingest_notes()
    print("Ingestion complete!")