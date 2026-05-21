#!/usr/bin/env python3
"""
scripts/download_model.py
-------------------------
One-time script to download the sentence-transformers embedding model.

RUN THIS ONCE before using the RAG system:
    python scripts/download_model.py

WHY A SEPARATE SCRIPT?
----------------------
Corporate firewalls often block huggingface.co connection.
This script lets you:
  1. Connect to a personal hotspot / home WiFi
  2. Run this script ONCE to download the model (~80MB)
  3. Go back to corporate network — the model is cached locally
  4. All RAG operations work offline forever

The model is cached at: ~/.cache/huggingface/hub/
"""

import sys


def main() -> None:
    print("Downloading sentence-transformers model: all-MiniLM-L6-v2")
    print("This downloads ~80MB on first run, then uses local cache.\n")

    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer("all-MiniLM-L6-v2")
        dim = model.get_sentence_embedding_dimension()
        print(f"\n✅ Model downloaded and cached successfully!")
        print(f"   Embedding dimension: {dim}")
        print(f"   Model is now available offline.")

        # Quick sanity test
        test_embedding = model.encode(["test"])
        print(f"   Test embedding shape: {test_embedding.shape}")
        print(f"\nYou can now run: python -m src.rag.vector_store")

    except Exception as exc:
        print(f"\n❌ Download failed: {exc}")
        print("\nTroubleshooting:")
        print("  1. Connect to a personal hotspot or home WiFi")
        print("  2. Disconnect from VPN if active")
        print("  3. Try again: python scripts/download_model.py")
        sys.exit(1)


if __name__ == "__main__":
    main()
