import re
from typing import List, Dict, Any, Optional

from app.config import GOOGLE_API_KEY, DEFAULT_TOP_K
from app.vector_store import vector_store_manager
from langchain_core.prompts import PromptTemplate

STRICT_QA_PROMPT = PromptTemplate(
    template=(
        "You are a precise Document Q&A assistant. Answer the user's question using ONLY the provided context snippets below.\n"
        "If the answer cannot be found in the context, respond EXACTLY with:\n"
        "\"I cannot find the answer in the provided context.\"\n\n"
        "Do NOT rely on prior knowledge or make assumptions.\n\n"
        "Context:\n"
        "{context}\n\n"
        "Question: {question}\n"
        "Answer:"
    ),
    input_variables=["context", "question"]
)


def get_llm():
    """Returns Gemini LLM if API key is configured, else returns None (offline fallback)."""
    if GOOGLE_API_KEY and GOOGLE_API_KEY.strip():
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            return ChatGoogleGenerativeAI(
                model="gemini-1.5-flash",
                google_api_key=GOOGLE_API_KEY,
                temperature=0.0
            )
        except Exception as e:
            print(f"Failed to initialize ChatGoogleGenerativeAI ({e}). Falling back to offline context reader.")
    return None


def offline_grounded_answer(question: str, context_docs: List[Any]) -> str:
    """
    Extractive fallback answer generator when running offline without Gemini API key.
    Searches context docs for sentence matches or returns the most relevant context line.
    """
    if not context_docs:
        return "I cannot find the answer in the provided context."

    q_words = set(re.findall(r'\w+', question.lower())) - {"what", "is", "the", "a", "an", "in", "of", "and", "or", "for", "to", "how", "where", "who", "which"}
    
    best_sentence = ""
    best_overlap = 0

    for doc in context_docs:
        text = doc.page_content
        sentences = re.split(r'(?<=[.!?])\s+', text)
        for sentence in sentences:
            s_words = set(re.findall(r'\w+', sentence.lower()))
            overlap = len(q_words.intersection(s_words))
            if overlap > best_overlap:
                best_overlap = overlap
                best_sentence = sentence.strip()

    if best_overlap > 0 and best_sentence:
        return best_sentence
    
    # If no word overlap but context exists, return first concise snippet from top context doc
    first_doc_snippet = context_docs[0].page_content.strip()
    if len(first_doc_snippet) > 300:
        first_doc_snippet = first_doc_snippet[:300] + "..."
    return first_doc_snippet if first_doc_snippet else "I cannot find the answer in the provided context."


def query_rag_pipeline(question: str, document_id: Optional[str] = None, top_k: int = DEFAULT_TOP_K) -> Dict[str, Any]:
    """
    Retrieves context from FAISS and generates a grounded response.
    """
    # Step 1: Retrieve relevant documents
    retrieved_docs = vector_store_manager.similarity_search(
        query=question,
        k=top_k,
        document_id=document_id
    )

    if not retrieved_docs:
        return {
            "answer": "I cannot find the answer in the provided context.",
            "sources": []
        }

    # Format sources for response
    sources = []
    seen_sources = set()
    for doc in retrieved_docs:
        fn = doc.metadata.get("filename", "unknown")
        pg = doc.metadata.get("page", 1)
        snippet = doc.page_content.strip()
        if len(snippet) > 200:
            snippet = snippet[:200] + "..."

        source_key = (fn, pg, snippet[:50])
        if source_key not in seen_sources:
            seen_sources.add(source_key)
            sources.append({
                "filename": fn,
                "page": pg,
                "snippet": snippet
            })

    # Step 2: Combine context
    formatted_context = "\n\n---\n\n".join(
        [f"[Source: {doc.metadata.get('filename')} (Page {doc.metadata.get('page')})]\n{doc.page_content}" for doc in retrieved_docs]
    )

    # Step 3: LLM generation (Gemini or Offline Fallback)
    llm = get_llm()

    if llm is not None:
        try:
            formatted_prompt = STRICT_QA_PROMPT.format(context=formatted_context, question=question)
            response = llm.invoke(formatted_prompt)
            answer_text = response.content.strip() if hasattr(response, 'content') else str(response).strip()
        except Exception as e:
            print(f"Gemini API invocation failed ({e}). Using offline answer engine.")
            answer_text = offline_grounded_answer(question, retrieved_docs)
    else:
        answer_text = offline_grounded_answer(question, retrieved_docs)

    return {
        "answer": answer_text,
        "sources": sources
    }
