import os
import streamlit as st
from youtube_transcript_api import TranscriptsDisabled
from langchain_huggingface import HuggingFaceEmbeddings

# Import backend logic from main.py
from main import (
    extract_video_id,
    get_transcript,
    create_vector_store,
    build_rag_chain
)

# ----------------- PAGE CONFIG -----------------
st.set_page_config(page_title="YouTube RAG Assistant", page_icon="🎥", layout="wide")
st.title("🎥 YouTube Video Q&A (RAG Assistant)")

# ----------------- CACHED EMBEDDINGS -----------------
@st.cache_resource
def load_embedding_model():
    """Load and cache the embedding model once."""
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# ----------------- SIDEBAR -----------------
with st.sidebar:
    st.header("⚙️ Configuration")
    default_api_key = os.environ.get("GROQ_API_KEY", "")
    groq_api_key = st.text_input("Groq API Key", value=default_api_key, type="password")
    
    st.markdown("---")
    st.header("📺 Video Source")
    video_input = st.text_input("YouTube URL or Video ID", value="LPZh9BOjkQs")
    
    if st.button("🚀 Load & Index Video", type="primary"):
        if not groq_api_key:
            st.error("Please enter your Groq API Key.")
        elif not video_input:
            st.error("Please enter a YouTube video URL or ID.")
        else:
            video_id = extract_video_id(video_input)
            with st.spinner("Fetching transcript and building vector store..."):
                try:
                    # 1. Fetch transcript
                    transcript = get_transcript(video_id)
                    
                    # 2. Build vector store
                    embeddings = load_embedding_model()
                    vector_store, num_chunks = create_vector_store(transcript, embeddings)
                    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
                    
                    # 3. Build LCEL chain
                    rag_chain = build_rag_chain(retriever, groq_api_key)
                    
                    # Store in session state
                    st.session_state["rag_chain"] = rag_chain
                    st.session_state["retriever"] = retriever
                    st.session_state["video_id"] = video_id
                    st.session_state["messages"] = []  # reset conversation for new video
                    
                    st.success(f"Video indexed successfully! ({num_chunks} chunks)")
                except TranscriptsDisabled:
                    st.error("Captions/transcripts are disabled for this video.")
                except Exception as e:
                    st.error(f"Error processing video: {e}")

    # Video Player preview in sidebar
    if "video_id" in st.session_state:
        st.markdown("---")
        st.video(f"https://www.youtube.com/watch?v={st.session_state['video_id']}")

# ----------------- MAIN CHAT AREA -----------------
if "rag_chain" not in st.session_state:
    st.info("👈 Enter your YouTube URL and Groq API Key in the sidebar, then click **'Load & Index Video'** to begin.")
else:
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    # Display conversation history
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "sources" in msg:
                with st.expander("🔍 View Retrieved Sources"):
                    for idx, doc in enumerate(msg["sources"], 1):
                        st.markdown(f"**Chunk {idx}:**\n> {doc.page_content}")

    # User input
    if user_query := st.chat_input("Ask a question about the video..."):
        st.session_state["messages"].append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        retriever = st.session_state["retriever"]
        rag_chain = st.session_state["rag_chain"]
        
        with st.chat_message("assistant"):
            sources = retriever.invoke(user_query)
            
            # Stream the response in real-time
            response_stream = rag_chain.stream(user_query)
            full_response = st.write_stream(response_stream)
            
            with st.expander("🔍 View Retrieved Sources"):
                for idx, doc in enumerate(sources, 1):
                    st.markdown(f"**Chunk {idx}:**\n> {doc.page_content}")
            
            st.session_state["messages"].append({
                "role": "assistant",
                "content": full_response,
                "sources": sources
            })
