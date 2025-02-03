from datetime import datetime
import os
import json
import time
import streamlit as st
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from ingest import ingest_notes

load_dotenv()

# Configure Streamlit
st.set_page_config(page_title="AI Notes Agent", layout="wide")
st.title("🔍 Smart Notes Search")

def handle_deletion():
    """Universal deletion handler with proper state management"""
    # Local files deletion
    if "pending_local_delete" in st.session_state:
        path = st.session_state.pending_local_delete
        try:
            os.remove(path)
            st.success(f"Deleted: {os.path.basename(path)}")
            time.sleep(0.5)
            del st.session_state.pending_local_delete
            st.rerun()
        except Exception as e:
            st.error(f"Delete failed: {str(e)}")
            del st.session_state.pending_local_delete

    # External files deletion
    if "pending_external_delete" in st.session_state:
        path = st.session_state.pending_external_delete
        try:
            with open("external_files.json", "r") as f:
                externals = json.load(f)
            updated = [e for e in externals if e["path"] != path]
            with open("external_files.json", "w") as f:
                json.dump(updated, f, indent=2)
            st.success(f"Removed tracking for: {os.path.basename(path)}")
            time.sleep(0.5)
            del st.session_state.pending_external_delete
            st.rerun()
        except Exception as e:
            st.error(f"Removal failed: {str(e)}")
            del st.session_state.pending_external_delete

def confirm_dialog():
    """Universal confirmation dialog component"""
    if "pending_local_delete" in st.session_state:
        path = st.session_state.pending_local_delete
        st.warning(f"Delete {os.path.basename(path)} permanently?")
        col1, col2 = st.columns([1, 2])
        if col1.button("Confirm"):
            handle_deletion()
        if col2.button("Cancel"):
            del st.session_state.pending_local_delete
            
    elif "pending_external_delete" in st.session_state:
        path = st.session_state.pending_external_delete
        st.warning(f"Stop tracking {os.path.basename(path)}?")
        col1, col2 = st.columns([1, 2])
        if col1.button("Confirm"):
            handle_deletion()
        if col2.button("Cancel"):
            del st.session_state.pending_external_delete

# ================== Sidebar Components ==================
with st.sidebar:
    st.header("📂 Notes Management")
    
    # Initialize session state for upload tracking
    if 'upload_key' not in st.session_state:
        st.session_state.upload_key = 0
        
    # File uploader with persistent key
    uploaded_file = st.file_uploader(
        "Upload new TXT file",
        type=["txt"],
        help="Only .txt files allowed",
        key=f"file_uploader_{st.session_state.upload_key}"
    )
    
    if uploaded_file:
        try:
            # Validate file type
            if not uploaded_file.name.lower().endswith('.txt'):
                st.session_state.upload_error = "Only .txt files are allowed!"
                st.session_state.upload_key += 1
                st.rerun()
            
            # Handle file saving
            notes_dir = "notes"
            os.makedirs(notes_dir, exist_ok=True)
            
            # Create unique filename if needed
            base_name = os.path.splitext(uploaded_file.name)[0]
            extension = os.path.splitext(uploaded_file.name)[1]
            final_name = f"{base_name}_{int(time.time())}{extension}"
            save_path = os.path.join(notes_dir, final_name)
            
            # Save file
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Set success state
            st.session_state.upload_success = f"File uploaded: {final_name}"
            st.session_state.upload_key += 1  # Reset uploader
            st.rerun()
            
        except Exception as e:
            st.session_state.upload_error = f"Upload failed: {str(e)}"
            st.session_state.upload_key += 1
            st.rerun()
    
    # Show status messages
    if 'upload_success' in st.session_state:
        st.success(st.session_state.upload_success)
        del st.session_state.upload_success  # Clear after showing
        
    if 'upload_error' in st.session_state:
        st.error(st.session_state.upload_error)
        del st.session_state.upload_error  # Clear after showing


    if 'external_path_value' not in st.session_state:
        st.session_state.external_path_value = ""

    st.divider()
    st.subheader("🔗 Add External Files")
    
    # Text input and button in columns
    col1, col2 = st.columns([4, 1])
    
    # Input field with separate value tracking
    new_external = col1.text_input(
        "External file path:", 
        value=st.session_state.external_path_value,
        help="Absolute path to .txt file",
        key="external_path_input",
        on_change=lambda: st.session_state.__setitem__("external_path_value", st.session_state.external_path_input)
    )
    
    add_btn = col2.button("Add", key="add_path_btn")
    
    if add_btn:
        # Store the current value before clearing
        current_value = st.session_state.external_path_value
        
        # Immediately clear the input field through session state
        st.session_state.external_path_value = ""
        
        if not current_value:
            st.error("Please enter a file path!", icon="🚫")
        elif not current_value.endswith('.txt'):
            st.error("Only .txt files can be added!", icon="🚫")
        elif not os.path.exists(current_value):
            st.error("File path does not exist!", icon="❗")
        else:
            try:
                metadata_file = "external_files.json"
                existing = []
                
                # Load existing data safely
                if os.path.exists(metadata_file):
                    try:
                        with open(metadata_file, "r") as f:
                            if os.path.getsize(metadata_file) > 0:
                                existing = json.load(f)
                    except json.JSONDecodeError:
                        st.error("Corrupted file list, resetting...")
                        existing = []
                        with open(metadata_file, "w") as f:
                            json.dump([], f)
                
                # Check for duplicates
                existing_paths = [entry["path"] for entry in existing]
                if current_value in existing_paths:
                    st.error("Path already exists!", icon="⚠️")
                else:
                    # Add new entry
                    existing.append({
                        "path": current_value,
                        "added_at": datetime.now().isoformat()
                    })
                    
                    # Save updated list
                    with open(metadata_file, "w") as f:
                        json.dump(existing, f, indent=2)
                    
                    st.success(f"Added: {os.path.basename(current_value)}", icon="✅")
                    time.sleep(2)
                    st.rerun()
                    
            except Exception as e:
                st.error(f"Error: {str(e)}", icon="❌")
                st.session_state.external_path_value = current_value  # Restore value on error

    # Display existing files
    st.divider()
    refresh_db=st.button("Refresh the Knowledge Base")
    if refresh_db:
        print("Manually refershing the Knowledge Db")
        with st.spinner("🔄 Updating knowledge base ..."):
            ingest_notes()
            st.cache_resource.clear()


    cols=st.columns([4,2])
    st.subheader("📚 All Sources")
    
    # Display local files with delete buttons
    st.markdown("**Local Notes**")
    local_files = []
    for root, dirs, files in os.walk("notes"):
        for file in files:
            if file.endswith(".txt"):
                path = os.path.join(root, file)
                local_files.append((file, path))

    if local_files:
        for filename, filepath in sorted(local_files):
            cols = st.columns([4, 1])
            cols[0].write(f"📄 {filename}")
            
            # Delete button - only stores deletion request internal notes
            if cols[1].button("×", key=f"del_local_{filename}"):
                st.session_state.pending_local_delete = filepath
    else:
        st.info("No local notes found")

    # Display external files with delete buttons
    st.markdown("**External Files**")
    try:
        if os.path.exists("external_files.json"):
            with open("external_files.json", "r") as f:
                if os.path.getsize("external_files.json") > 0:
                    externals = json.load(f)
                    for entry in externals:
                        cols = st.columns([4, 1])
                        path = entry["path"]
                        
                        # File info
                        cols[0].write(f"🔗 `{path}`")
                        if not os.path.exists(path):
                            cols[0].warning("File not found!")
                        
                        # Delete button external files
                        if cols[1].button("×", key=f"del_ext_{hash(path)}"):
                          st.session_state.pending_external_delete = path
                        
                    if not externals:
                        st.info("No external files added")
    except Exception as e:
        st.error(f"Error loading external files: {str(e)}")
       # Show confirmation dialog if needed
    confirm_dialog()
    

# ================== Main App Functionality ==================
@st.cache_resource
def load_retriever():
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.load_local("vector_store", embeddings, allow_dangerous_deserialization=True)
    return vector_store.as_retriever(search_kwargs={"k": 3})

def check_for_updates():
    try:
        current_state = {
            "local": {},
            "external": {}
        }
        
        # Check local files
        for root, _, files in os.walk("notes"):
            for file in files:
                if file.endswith(".txt"):
                    path = os.path.join(root, file)
                    current_state["local"][path] = os.path.getmtime(path)
        
        # Check external files
        with open("external_files.json", "r") as f:
            externals = json.load(f)
            for entry in externals:
                path = entry["path"]
                if os.path.exists(path):
                    current_state["external"][path] = os.path.getmtime(path)
        
        # Compare with last ingestion
        with open(os.path.join("vector_store", "metadata.json"), "r") as f:
            last_state = json.load(f)
            
        local_changed = any(
            path not in last_state["local_files"] or 
            current_state["local"].get(path) != os.path.getmtime(path)
            for path in current_state["local"]
        )
        
        external_changed = any(
            path not in last_state["external_files"] or 
            current_state["external"].get(path) != os.path.getmtime(path)
            for path in current_state["external"]
        )
        
        return local_changed or external_changed
        
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        return True


# Query processing
query = st.text_input("Ask about your notes:")

if query:
    if check_for_updates():
        with st.spinner("🔄 Updating knowledge base with new notes..."):
            ingest_notes()
            st.cache_resource.clear()

    retriever = load_retriever()
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp", temperature=0.3)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    
    result = qa_chain.invoke({"query": query})
    
    st.subheader("Answer:")
    st.write(result["result"])

    st.subheader("Source Files:")
    seen_sources = set()
    for doc in result["source_documents"]:
        source_path = doc.metadata["source"]
        source_file = os.path.basename(source_path)
        if source_file not in seen_sources:
            col1, col2 = st.columns([2, 10])
            col1.write(f"📄 {source_file}")
            col2.caption(f"Path: {os.path.dirname(source_path)}")
            seen_sources.add(source_file)