import streamlit as st
from pathlib import Path
import hashlib
from urllib.parse import quote
import base64

from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="RBI Regulatory Intelligence",
    page_icon="🏛️",
    layout="wide"
)

st.markdown("""
<style>
    .main { background-color: #f8f9fa; }
    .user-message {
        background-color: #e3f2fd;
        padding: 15px 20px;
        border-radius: 15px;
        margin: 10px 0;
        margin-left: 25%;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .assistant-message {
        background-color: #ffffff;
        padding: 15px 20px;
        border-radius: 15px;
        margin: 10px 0;
        margin-right: 25%;
        border-left: 4px solid #1976d2;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .sources-header {
        color: #424242;
        font-weight: 600;
        margin-top: 15px;
        margin-bottom: 10px;
        font-size: 1.05em;
    }
    .pdf-button {
        background-color: #1976d2;
        color: white;
        padding: 8px 16px;
        border-radius: 5px;
        text-decoration: none;
        display: inline-block;
        margin: 5px;
        font-weight: 500;
        border: none;
        cursor: pointer;
    }
    .pdf-button:hover {
        background-color: #1565c0;
        color: white;
        text-decoration: none;
    }
    .highlight-box {
        background-color: #fff9c4;
        padding: 10px;
        border-left: 3px solid #fbc02d;
        margin: 10px 0;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

def doc_hash(doc):
    return hashlib.md5(doc.page_content.encode("utf-8")).hexdigest()

def build_structured_context(docs):
    context_blocks = []
    source_map = {}
    for i, doc in enumerate(docs, 1):
        metadata = doc.metadata
        document_name = metadata.get("file_name", "Unknown document")
        page_number = metadata.get("page_number", "N/A")
        context_blocks.append(f"""[Source {i}]
Document: {document_name}
Page: {page_number}

Content:
{doc.page_content}""".strip())
        source_map[i] = {
            "file": document_name,
            "page": page_number,
            "content": doc.page_content
        }
    return "\n\n---\n\n".join(context_blocks), source_map

def show_pdf_with_page(pdf_path, page_number):
    """Display PDF at specific page using iframe"""
    try:
        # Read the PDF file
        with open(pdf_path, "rb") as f:
            base64_pdf = base64.b64encode(f.read()).decode('utf-8')
        
        # Create PDF viewer with page parameter
        pdf_display = f"""
        <iframe 
            src="data:application/pdf;base64,{base64_pdf}#page={page_number}" 
            width="100%" 
            height="800px" 
            type="application/pdf"
            style="border: 1px solid #ddd; border-radius: 5px;">
        </iframe>
        """
        
        st.markdown(pdf_display, unsafe_allow_html=True)
        return True
    except Exception as e:
        st.error(f"Could not load PDF: {e}")
        return False

class HybridRetriever:
    def __init__(self, bm25_retriever, vector_retriever, weights=[0.5, 0.5]):
        self.bm25_retriever = bm25_retriever
        self.vector_retriever = vector_retriever
        self.bm25_weight = weights[0]
        self.vector_weight = weights[1]

    def invoke(self, query, k=5):
        bm25_docs = self.bm25_retriever.invoke(query)
        vector_docs = self.vector_retriever.invoke(query)
        rrf_k = 60
        doc_scores = {}
        doc_map = {}
        for rank, doc in enumerate(bm25_docs, 1):
            key = doc_hash(doc)
            doc_scores[key] = self.bm25_weight / (rrf_k + rank)
            doc_map[key] = doc
        for rank, doc in enumerate(vector_docs, 1):
            key = doc_hash(doc)
            score = self.vector_weight / (rrf_k + rank)
            doc_scores[key] = doc_scores.get(key, 0) + score
            doc_map[key] = doc
        ranked = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        return [doc_map[k] for k, _ in ranked[:k]]

def rerank_docs(query, docs, reranker, top_k=5):
    pairs = [(query, doc.page_content) for doc in docs]
    scores = reranker.predict(pairs)
    scored_docs = list(zip(docs, scores))
    scored_docs.sort(key=lambda x: x[1], reverse=True)
    return [doc for doc, score in scored_docs[:top_k]]

def answer_query(query, hybrid_retriever, reranker, rag_chain, top_k_retrieval=20, top_k_rerank=5):
    candidate_docs = hybrid_retriever.invoke(query, k=top_k_retrieval)
    if not candidate_docs:
        return {"answer": "I couldn't find relevant information.", "sources": {}}
    reranked_docs = rerank_docs(query, candidate_docs, reranker, top_k_rerank)
    if not reranked_docs:
        return {"answer": "No relevant documents after reranking.", "sources": {}}
    context_text, source_map = build_structured_context(reranked_docs)
    response = rag_chain.invoke({"context": context_text, "question": query})
    return {"answer": response, "sources": source_map}

@st.cache_resource
def initialize_system():
    progress_text = st.empty()
    
    progress_text.text("🔧 Loading ChromaDB...")
    chroma_path = Path("./notebook/chroma_db")
    if not chroma_path.exists():
        st.error(f"❌ ChromaDB not found at: {chroma_path.absolute()}")
        st.stop()
    
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = Chroma(
        persist_directory="./notebook/chroma_db",
        embedding_function=embeddings,
        collection_name="rbi_master_circulars"
    )
    
    progress_text.text("📄 Loading documents...")
    all_docs = vectorstore.get()
    if not all_docs or 'documents' not in all_docs or len(all_docs['documents']) == 0:
        st.error("❌ No documents in ChromaDB collection!")
        st.stop()
    
    documents = [
        Document(page_content=content, metadata=metadata)
        for content, metadata in zip(all_docs['documents'], all_docs['metadatas'])
    ]
    progress_text.text(f"✅ Loaded {len(documents)} documents")
    
    progress_text.text("🔧 Creating retrievers...")
    bm25_retriever = BM25Retriever.from_documents(documents, bm25_variant="plus")
    bm25_retriever.k = 11
    vectorstore_retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 11})
    hybrid_retriever = HybridRetriever(bm25_retriever, vectorstore_retriever, weights=[0.5, 0.5])
    
    progress_text.text("🔧 Loading reranker...")
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    
    progress_text.text("🔧 Loading LLM...")
    llm = OllamaLLM(model="qwen2.5:1.5b", temperature=0.0, top_p=1.0)
    
    prompt_template = ChatPromptTemplate.from_template("""
You are a document-grounded research assistant.

You must answer the question using ONLY the information explicitly present
in the provided document context.

==============================
CRITICAL RULES (NON-NEGOTIABLE)
==============================

1. You MUST use ONLY the information stated verbatim in the context.
2. Do NOT use prior knowledge, external knowledge, assumptions, inference, or interpretation.
3. EVERY factual sentence MUST end with a citation in the exact format: [Source X].
4. If the question contains multiple sub-questions, ALL sub-questions MUST be answered.
5. If ANY sub-question is missing, unclear, or not explicitly answered in the context,
   you MUST refuse to answer entirely.

==============================
MANDATORY REFUSAL CONDITIONS
==============================

You MUST respond ONLY with ONE of the following sentences if the conditions apply:

- If any required information is missing or unclear:
  "I couldn't find this information in the available documents."

- If the question contains multiple sub-questions and ANY part is not fully answered:
  "I could not find information related to this question according to the provided documents."

Do NOT add explanations.
Do NOT add citations.
Do NOT add summaries.

==============================
PROHIBITED BEHAVIOR
==============================

- Do NOT speculate, infer, or extrapolate.
- Do NOT restate or rephrase the question.
- Do NOT explain reasoning or decision-making.
- Do NOT summarize document structure or metadata.
- Do NOT include phrases such as:
  "it appears", "based on", "suggests", "indicates", or similar.
- Do NOT include a references list or source summary section.
- Do NOT repeat the context text verbatim beyond what is necessary to answer.

==============================
CONTEXT
==============================

Below are retrieved document sections.
Each section is identified by a unique source number.

{context}

==============================
QUESTION
==============================

{question}

==============================
ANSWER FORMAT
==============================

- Use bullet points only if the context explicitly lists items.
- Each sentence MUST end with one or more citations in the form [Source X].
- Do NOT include headings, explanations, or extra commentary.

==============================
ANSWER
==============================
""")
    
    rag_chain = prompt_template | llm | StrOutputParser()
    
    progress_text.text("✅ System ready!")
    progress_text.empty()
    
    return hybrid_retriever, reranker, rag_chain

def main():
    st.markdown("# 🏛️ RBI Regulatory Intelligence System")
    st.caption("AI-Powered Master Circulars Query Assistant")
    
    with st.sidebar:
        st.markdown("### ⚙️ Settings")
        top_k_retrieval = st.slider("Initial Retrieval", 10, 50, 20, 5)
        top_k_rerank = st.slider("Final Documents", 3, 10, 5, 1)
        st.markdown("---")
        st.markdown("### 📊 System Info")
        st.info("**LLM:** Qwen2.5:1.5b\n**Embeddings:** Nomic-Embed-Text\n**Retrieval:** Hybrid (BM25 + Vector)\n**Reranker:** MS-MARCO MiniLM")
        st.markdown("---")
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            if 'show_pdf' in st.session_state:
                del st.session_state['show_pdf']
            st.rerun()
    
    try:
        hybrid_retriever, reranker, rag_chain = initialize_system()
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        st.info("💡 Make sure:\n1. Ollama is running: `ollama serve`\n2. Models are pulled: `ollama pull nomic-embed-text` and `ollama pull qwen2.5:1.5b`\n3. You've run the indexing notebook")
        st.stop()
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display PDF if requested
    if 'show_pdf' in st.session_state and st.session_state.show_pdf:
        pdf_info = st.session_state.show_pdf
        st.markdown(f"### 📄 Viewing: {pdf_info['file']} (Page {pdf_info['page']})")
        
        col1, col2 = st.columns([1, 5])
        with col1:
            if st.button("← Back to Chat"):
                del st.session_state['show_pdf']
                st.rerun()
        
        # Show highlighted text
        st.markdown('<div class="highlight-box">', unsafe_allow_html=True)
        st.markdown("**🔍 Relevant Text from this page:**")
        st.text_area("", value=pdf_info['content'], height=150, label_visibility="collapsed", key="highlight_text")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Show PDF
        pdf_path = Path("data/pdfs") / pdf_info['file']
        if pdf_path.exists():
            show_pdf_with_page(pdf_path, pdf_info['page'])
        else:
            st.error(f"PDF file not found: {pdf_path}")
            st.info(f"Looking for: {pdf_path.absolute()}")
        
        return
    
    # Normal chat interface
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f'<div class="user-message">👤 <strong>You</strong><br/>{message["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="assistant-message">🏛️ <strong>RBI Assistant</strong><br/>{message["content"]}</div>', unsafe_allow_html=True)
            if "sources" in message and message["sources"]:
                st.markdown('<p class="sources-header">📚 Sources</p>', unsafe_allow_html=True)
                for source_id, source_info in message["sources"].items():
                    with st.expander(f"📄 Source {source_id}: {source_info['file']} (Page {source_info['page']})", expanded=False):
                        st.markdown(f"**📁 Document:** {source_info['file']}")
                        st.markdown(f"**📖 Page:** {source_info['page']}")
                        st.markdown("**📝 Relevant Text:**")
                        chunk_preview = source_info['content']
                        if len(chunk_preview) > 500:
                            chunk_preview = chunk_preview[:500] + "..."
                        st.text_area("content", value=chunk_preview, height=150, key=f"src_{message['timestamp']}_{source_id}", label_visibility="collapsed")
                        
                        # Button to view PDF
                        if st.button(f"📖 View PDF at Page {source_info['page']}", key=f"pdf_btn_{message['timestamp']}_{source_id}"):
                            st.session_state.show_pdf = {
                                'file': source_info['file'],
                                'page': source_info['page'],
                                'content': source_info['content']
                            }
                            st.rerun()
    
    if prompt := st.chat_input("Ask about RBI Master Circulars..."):
        import time
        timestamp = time.time()
        st.session_state.messages.append({"role": "user", "content": prompt, "timestamp": timestamp})
        st.markdown(f'<div class="user-message">👤 <strong>You</strong><br/>{prompt}</div>', unsafe_allow_html=True)
        
        with st.spinner("🔍 Searching..."):
            response = answer_query(prompt, hybrid_retriever, reranker, rag_chain, top_k_retrieval, top_k_rerank)
        
        st.session_state.messages.append({"role": "assistant", "content": response["answer"], "sources": response["sources"], "timestamp": timestamp})
        st.rerun()

if __name__ == "__main__":
    main()