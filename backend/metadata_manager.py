import os
import json

METADATA_FILE = os.path.join("data", "metadata.json")

def load_metadata():
    """Loads document metadata from JSON file."""
    if not os.path.exists(METADATA_FILE):
        return {}
    try:
        with open(METADATA_FILE, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"[ERROR] Failed to load metadata: {e}")
        return {}

def save_metadata(metadata):
    """Saves document metadata to JSON file."""
    os.makedirs(os.path.dirname(METADATA_FILE), exist_ok=True)
    try:
        with open(METADATA_FILE, 'w') as f:
            json.dump(metadata, f, indent=4)
    except Exception as e:
        print(f"[ERROR] Failed to save metadata: {e}")

def add_document(filename, chunk_count, status="Indexed"):
    """Adds or updates a document in the metadata storage."""
    metadata = load_metadata()
    metadata[filename] = {
        "chunks": chunk_count,
        "status": status
    }
    save_metadata(metadata)

def remove_document(filename):
    """Removes a document from the metadata storage."""
    metadata = load_metadata()
    if filename in metadata:
        del metadata[filename]
        save_metadata(metadata)

def get_all_documents():
    """Returns all documents and their metadata."""
    return load_metadata()

def document_exists(filename):
    """Checks if a document exists in metadata."""
    metadata = load_metadata()
    return filename in metadata
