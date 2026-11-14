import re
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

def extract_video_id(url_or_id: str) -> str:
    """Extracts an 11-character video ID from a URL or returns the ID as-is."""
    match = re.search(r'(?:v=|\/|youtu\.be\/)([0-9A-Za-z_-]{11})', url_or_id)
    return match.group(1) if match else url_or_id.strip()

def get_transcript(video_id: str) -> str:
    """Fetches and concatenates the transcript for a YouTube video ID."""
    ytt = YouTubeTranscriptApi()
    transcript_list = ytt.fetch(video_id, languages=["en"])
    return " ".join(chunk.text for chunk in transcript_list)

def create_vector_store(transcript: str, embeddings, chunk_size: int = 1000, chunk_overlap: int = 200):
    """Splits transcript into chunks and returns a FAISS vector store and chunk count."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = splitter.create_documents([transcript])
    vector_store = FAISS.from_documents(docs, embeddings)
    return vector_store, len(docs)

def format_docs(docs):
    """Combines retrieved document contents into a single text block."""
    return "\n\n".join(doc.page_content for doc in docs)

def build_rag_chain(retriever, groq_api_key: str, model_name: str = "openai/gpt-oss-20b"):
    """Constructs and returns the LCEL RAG chain."""
    prompt_template = PromptTemplate(
        template="""You are a helpful assistant. Answer the question based on the provided YouTube video transcript context.
If the context does not provide a formal textbook definition, summarize what the video mentions, explains, or implies about the topic.
Only if the topic is completely absent from the context, say "I don't know based on this video. 

Context:
{context}

Question: {question}
Answer:""",
        input_variables=['context', 'question']
    )
    
    llm = ChatGroq(model=model_name, temperature=0.3, groq_api_key=groq_api_key)
    
    parallel_chain = RunnableParallel({
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    })
    
    return parallel_chain | prompt_template | llm | StrOutputParser()
