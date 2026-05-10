# Streamlit UI Specification

## File
app.py — single file, all UI code here

## Layout
LEFT SIDEBAR:
- App title
- PDF drag and drop uploader
- List of uploaded documents

MAIN AREA:
- Chat message history
- Each answer shows citations as expanders below
- Fixed chat input bar at bottom

## User Flow
1. User opens app
2. Uploads PDF via sidebar
3. App runs ingest(pdf_path) with progress bar
4. User types question in chat bar
5. App runs generate(query, chunks, bm25_index)
6. Answer appears in chat with [SOURCE N] citations
7. User expands citations to see source chunks

## Components
- st.file_uploader() — PDF upload
- st.chat_input() — question input
- st.chat_message() — display messages
- st.expander() — citation details
- st.progress() — ingestion progress bar
- st.spinner() — loading state

## State Management
st.session_state:
- messages: list of chat history
- chunks: loaded from ChromaDB
- bm25_index: built after ingestion
- ingested_docs: list of uploaded filenames

## Backend Integration

### On PDF Upload
- save file to data/raw/
- call ingest(pdf_path)
- reload chunks from ChromaDB
- rebuild bm25_index
- show progress bar

### On Query Submit
- call generate(query, chunks, bm25_index)
- display answer in chat
- show sources as expanders

## Error Handling
- PDF upload fails → show error message
- No PDF uploaded → warn before querying
- LLM timeout → show friendly error
- Empty answer → show fallback message