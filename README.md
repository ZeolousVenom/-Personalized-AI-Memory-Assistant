✨ Personalized AI Memory Assistant

Ask questions and get answers based on stored text memories using a local Mistral LLM + Vector Database.

🚀 Project Overview

The Personalized AI Memory Assistant allows users to store any text as “memories” and later ask questions related to the stored content.
The system retrieves the most relevant memory chunks using vector search and generates responses using a local Mistral LLM.

This works as a mini-RAG (Retrieval-Augmented Generation) system with a simple Streamlit UI.

🧠 Features
✅ Memory Management

Add new memory text

View all stored memories

Store embeddings in a vector database

🔍 Smart Retrieval

Uses vector search to fetch relevant memory chunks

Supports semantic similarity search

💬 Chat Interface

Users ask questions

System retrieves relevant memory

Mistral LLM generates accurate answers based on stored data

📊 Statistics

Total memories stored

Personal memory count

Retrieval logs (optional)

🏗️ Tech Stack
Component	Technology
UI	Streamlit
LLM	Mistral (local or remote)
Vector DB	Milvus / Chroma / FAISS (depending on your implementation)
Embeddings	Sentence Transformers / Mistral Embedding API
Memory Store	Vector database
