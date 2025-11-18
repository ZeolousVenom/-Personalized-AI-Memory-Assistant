import streamlit as st
import numpy as np
from datetime import datetime
import json
from typing import List, Dict
import uuid
import requests

# Install required packages:
# pip install streamlit numpy sentence-transformers requests

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    st.error("Please install sentence-transformers: pip install sentence-transformers")

class EmbeddingModel:
    """Real embedding model using sentence-transformers"""
    def __init__(self):
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
    def encode(self, text: str) -> np.ndarray:
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            return self.model.encode(text)
        else:
            # Fallback to mock
            np.random.seed(hash(text) % (2**32))
            return np.random.randn(384).astype(np.float32)

class VectorDatabase:
    """Simple in-memory vector database"""
    def __init__(self):
        self.memories = []
    
    def insert(self, embedding: np.ndarray, text: str, metadata: Dict) -> str:
        memory_id = str(uuid.uuid4())
        self.memories.append({
            'id': memory_id,
            'embedding': embedding,
            'text': text,
            'metadata': metadata
        })
        return memory_id
    
    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict]:
        if not self.memories:
            return []
        
        # Cosine similarity search
        results = []
        for memory in self.memories:
            similarity = np.dot(query_embedding, memory['embedding']) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(memory['embedding'])
            )
            results.append({
                'id': memory['id'],
                'text': memory['text'],
                'metadata': memory['metadata'],
                'score': float(similarity)
            })
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:top_k]
    
    def delete(self, memory_id: str) -> bool:
        initial_len = len(self.memories)
        self.memories = [m for m in self.memories if m['id'] != memory_id]
        return len(self.memories) < initial_len
    
    def get_all(self) -> List[Dict]:
        return [{
            'id': m['id'],
            'text': m['text'],
            'metadata': m['metadata']
        } for m in self.memories]

class LocalMistralLLM:
    """Local Mistral LLM client (Ollama or local API)"""
    def __init__(self, api_url: str = "http://localhost:11434", model_name: str = "mistral"):
        self.api_url = api_url.rstrip('/')
        self.model_name = model_name
        self.api_type = self.detect_api_type()
    
    def detect_api_type(self):
        """Detect if using Ollama or other API"""
        try:
            response = requests.get(f"{self.api_url}/api/tags", timeout=2)
            if response.status_code == 200:
                return "ollama"
        except:
            pass
        return "generic"
    
    def test_connection(self):
        """Test if the local LLM is accessible"""
        try:
            if self.api_type == "ollama":
                response = requests.get(f"{self.api_url}/api/tags", timeout=5)
                return response.status_code == 200
            else:
                response = requests.post(
                    f"{self.api_url}/v1/chat/completions",
                    json={"model": self.model_name, "messages": [{"role": "user", "content": "test"}]},
                    timeout=5
                )
                return response.status_code == 200
        except Exception as e:
            return False
    
    def generate(self, prompt: str, context: str = "") -> str:
        try:
            # Build the system prompt with context
            system_prompt = """You are a helpful AI assistant with access to stored memories about the user. 
Use the provided context to answer questions accurately. If the context doesn't contain relevant information, 
say so politely and provide a general response. Keep your answers concise and helpful."""
            
            # Build user message
            user_message = prompt
            if context:
                user_message = f"""Context from stored memories:
{context}

User question: {prompt}

Please answer based on the context above."""
            
            if self.api_type == "ollama":
                # Ollama API format
                response = requests.post(
                    f"{self.api_url}/api/chat",
                    json={
                        "model": self.model_name,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_message}
                        ],
                        "stream": False,
                        "options": {
                            "temperature": 0.7,
                            "num_predict": 500
                        }
                    },
                    timeout=60
                )
                
                if response.status_code == 200:
                    return response.json()["message"]["content"]
                else:
                    return f"❌ Error: {response.status_code} - {response.text}"
            
            else:
                # Generic OpenAI-compatible API format
                response = requests.post(
                    f"{self.api_url}/v1/chat/completions",
                    json={
                        "model": self.model_name,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_message}
                        ],
                        "temperature": 0.7,
                        "max_tokens": 500
                    },
                    timeout=60
                )
                
                if response.status_code == 200:
                    return response.json()["choices"][0]["message"]["content"]
                else:
                    return f"❌ Error: {response.status_code} - {response.text}"
                    
        except requests.exceptions.Timeout:
            return "❌ Request timed out. The local LLM might be overloaded or not responding."
        except requests.exceptions.ConnectionError:
            return f"❌ Cannot connect to local LLM at {self.api_url}. Please make sure it's running."
        except Exception as e:
            return f"❌ Error calling local Mistral: {str(e)}"

# Initialize session state
if 'vector_db' not in st.session_state:
    st.session_state.vector_db = VectorDatabase()
if 'embedding_model' not in st.session_state:
    st.session_state.embedding_model = EmbeddingModel()
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'api_url' not in st.session_state:
    st.session_state.api_url = "http://localhost:11434"
if 'model_name' not in st.session_state:
    st.session_state.model_name = "mistral"

# Page configuration
st.set_page_config(
    page_title="🧠 AI Memory Assistant",
    page_icon="🧠",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .user-message {
        background-color: #E3F2FD;
    }
    .assistant-message {
        background-color: #F5F5F5;
    }
    .memory-item {
        background-color: #f8f9fa;
        padding: 0.8rem;
        border-radius: 0.3rem;
        margin: 0.5rem 0;
        border-left: 3px solid #1E88E5;
    }
    .status-connected {
        color: #4CAF50;
        font-weight: bold;
    }
    .status-disconnected {
        color: #f44336;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header">🧠 Personalized AI Memory Assistant</div>', unsafe_allow_html=True)
st.markdown("*Ask questions and get answers based on stored memories using your local Mistral LLM!*")

# Sidebar for memory management and configuration
with st.sidebar:
    st.header("⚙️ LLM Configuration")
    
    # API URL and model configuration
    api_url = st.text_input(
        "Local API URL:",
        value=st.session_state.api_url,
        help="For Ollama: http://localhost:11434\nFor other APIs: http://localhost:8000"
    )
    
    model_name = st.text_input(
        "Model Name:",
        value=st.session_state.model_name,
        help="For Ollama: mistral, mistral:7b, etc."
    )
    
    # Update configuration if changed
    if api_url != st.session_state.api_url or model_name != st.session_state.model_name:
        st.session_state.api_url = api_url
        st.session_state.model_name = model_name
        st.session_state.llm = LocalMistralLLM(api_url, model_name)
        st.rerun()
    
    # Initialize LLM if not exists
    if 'llm' not in st.session_state:
        st.session_state.llm = LocalMistralLLM(api_url, model_name)
    
    # Test connection button
    if st.button("🔌 Test Connection", use_container_width=True):
        with st.spinner("Testing connection..."):
            is_connected = st.session_state.llm.test_connection()
            if is_connected:
                st.success(f"✅ Connected to {st.session_state.llm.api_type.upper()}")
            else:
                st.error(f"❌ Cannot connect to {api_url}")
    
    st.divider()
    st.header("📚 Memory Management")
    
    # Add memory section
    with st.expander("➕ Add New Memory", expanded=False):
        memory_text = st.text_area("Enter information to remember:", height=100)
        category = st.selectbox("Category:", ["Personal", "Work", "Preferences", "Facts", "Other"])
        importance = st.slider("Importance:", 1, 5, 3)
        
        if st.button("💾 Save Memory", use_container_width=True):
            if memory_text.strip():
                # Create embedding
                embedding = st.session_state.embedding_model.encode(memory_text)
                
                # Store in vector database
                metadata = {
                    'category': category,
                    'importance': importance,
                    'timestamp': datetime.now().isoformat()
                }
                memory_id = st.session_state.vector_db.insert(embedding, memory_text, metadata)
                
                st.success("✅ Memory saved successfully!")
                st.rerun()
            else:
                st.error("Please enter some text to remember.")
    
    # View all memories
    with st.expander("👁️ View All Memories", expanded=False):
        all_memories = st.session_state.vector_db.get_all()
        
        if all_memories:
            st.write(f"**Total memories: {len(all_memories)}**")
            
            for memory in all_memories:
                with st.container():
                    st.markdown(f"**{memory['metadata']['category']}** | ⭐ {memory['metadata']['importance']}/5")
                    st.write(memory['text'])
                    st.caption(f"Added: {memory['metadata']['timestamp'][:10]}")
                    
                    if st.button(f"🗑️ Delete", key=f"del_{memory['id']}", use_container_width=True):
                        st.session_state.vector_db.delete(memory['id'])
                        st.success("Memory deleted!")
                        st.rerun()
                    st.divider()
        else:
            st.info("No memories stored yet. Add some information to get started!")
    
    # Statistics
    st.divider()
    st.subheader("📊 Statistics")
    all_memories = st.session_state.vector_db.get_all()
    st.metric("Total Memories", len(all_memories))
    
    if all_memories:
        categories = [m['metadata']['category'] for m in all_memories]
        for cat in set(categories):
            st.metric(f"{cat}", categories.count(cat))

# Main chat interface
st.header("💬 Chat Interface")

# Display chat history
chat_container = st.container()
with chat_container:
    for message in st.session_state.chat_history:
        if message['role'] == 'user':
            st.markdown(f'<div class="chat-message user-message">👤 <b>You:</b> {message["content"]}</div>', 
                       unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-message assistant-message">🤖 <b>Assistant:</b><br>{message["content"]}</div>', 
                       unsafe_allow_html=True)
            
            if 'retrieved_memories' in message and message['retrieved_memories']:
                with st.expander("📖 Retrieved Memories Used", expanded=False):
                    for idx, mem in enumerate(message['retrieved_memories'], 1):
                        st.markdown(f'<div class="memory-item">', unsafe_allow_html=True)
                        st.markdown(f"**{idx}. Relevance: {mem['score']:.1%}**")
                        st.write(mem['text'])
                        st.caption(f"Category: {mem['metadata']['category']} | Importance: {mem['metadata']['importance']}/5")
                        st.markdown('</div>', unsafe_allow_html=True)

# Input area
st.divider()
user_input = st.text_input("💭 Ask me anything about your stored memories:", key="user_input", placeholder="e.g., What's my favorite programming language?")

col1, col2, col3 = st.columns([1, 1, 4])
with col1:
    send_button = st.button("📤 Send", use_container_width=True)
with col2:
    clear_button = st.button("🧹 Clear Chat", use_container_width=True)

# Process user input
if send_button and user_input.strip():
    # Add user message to chat history
    st.session_state.chat_history.append({
        'role': 'user',
        'content': user_input
    })
    
    # Create query embedding
    query_embedding = st.session_state.embedding_model.encode(user_input)
    
    # Search for relevant memories
    retrieved_memories = st.session_state.vector_db.search(query_embedding, top_k=5)
    
    # Filter memories by relevance threshold
    relevant_memories = [m for m in retrieved_memories if m['score'] > 0.3]
    
    # Prepare context for LLM
    context = ""
    if relevant_memories:
        context = "Relevant information from memory:\n\n"
        for idx, mem in enumerate(relevant_memories, 1):
            context += f"{idx}. {mem['text']} (Category: {mem['metadata']['category']}, Importance: {mem['metadata']['importance']}/5)\n"
    
    # Generate response using local Mistral
    with st.spinner("🤔 Thinking..."):
        response = st.session_state.llm.generate(user_input, context)
    
    # Add assistant response to chat history
    st.session_state.chat_history.append({
        'role': 'assistant',
        'content': response,
        'retrieved_memories': relevant_memories
    })
    
    st.rerun()

if clear_button:
    st.session_state.chat_history = []
    st.rerun()

# Quick action buttons
st.divider()
col1, col2 = st.columns(2)

with col1:
    if st.button("🔍 Show All Stored Information", use_container_width=True):
        all_memories = st.session_state.vector_db.get_all()
        if all_memories:
            summary = f"📚 I have **{len(all_memories)}** memories stored:\n\n"
            for idx, mem in enumerate(all_memories, 1):
                summary += f"**{idx}.** {mem['text']} *(Category: {mem['metadata']['category']}, Importance: {mem['metadata']['importance']}/5)*\n\n"
            
            st.session_state.chat_history.append({
                'role': 'assistant',
                'content': summary,
                'retrieved_memories': []
            })
        else:
            st.session_state.chat_history.append({
                'role': 'assistant',
                'content': "I don't have any memories stored yet. Please add some information using the sidebar!",
                'retrieved_memories': []
            })
        st.rerun()

with col2:
    if st.button("⚠️ Delete All Memories", use_container_width=True, type="secondary"):
        if st.session_state.vector_db.get_all():
            st.session_state.vector_db = VectorDatabase()
            st.success("🗑️ All memories deleted!")
            st.rerun()

# Footer
st.divider()
st.caption(f"🔧 **Tech Stack:** Streamlit | Local Mistral ({st.session_state.llm.api_type.upper()}) | Sentence Transformers | Vector Search")
st.caption("💡 **Setup:** Make sure your local Mistral is running (e.g., `ollama run mistral`), then test the connection in the sidebar!")