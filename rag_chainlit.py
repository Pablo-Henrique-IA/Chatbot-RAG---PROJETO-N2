import chainlit as cl
import os
import logging
import warnings
from together import Together
from typing import List, Dict, Any
from pinecone import Pinecone
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import Pinecone as PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from dotenv import load_dotenv

# Supressão de avisos
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Logging
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

# Carregamento das variáveis de ambiente
load_dotenv()

# Pinecone
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "chatbot-rag-2025")
PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE", "ns1")

# Together API
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
TOGETHER_MODEL = os.getenv("TOGETHER_MODEL", "google/gemma-2b-it")

# Inicializa Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

# Embeddings
def get_embeddings(text: str):
    model = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
    return model.embed_query(text)

# Consulta ao Pinecone
def query_pinecone(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    try:
        vector = get_embeddings(query)
        if not vector:
            logger.error("Erro: Falha ao gerar o embedding.")
            return []

        query_result = index.query(
            vector=[vector],
            top_k=top_k,
            namespace=PINECONE_NAMESPACE,
            include_metadata=True
        )
        return query_result.get("matches", [])
    except Exception as e:
        logger.error(f"Erro na consulta ao Pinecone: {e}")
        return []

# LLM via Together SDK
together_client = Together(api_key=TOGETHER_API_KEY)

def call_together_api(prompt, temperature: float = 0.1) -> str:
    prompt_text = prompt.to_string() if hasattr(prompt, "to_string") else str(prompt)
    try:
        response = together_client.chat.completions.create(
            model=TOGETHER_MODEL,
            messages=[{"role": "user", "content": prompt_text}],
            max_tokens=4096,
            temperature=temperature,
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Erro na Together API: {e}")
        return "Erro na geração da resposta."

# Configuração do retriever
def config_retriever():
    embedding_model = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large-instruct")
    retriever = PineconeVectorStore.from_existing_index(
        index_name=PINECONE_INDEX_NAME,
        embedding=embedding_model,
        namespace=PINECONE_NAMESPACE
    ).as_retriever(
        search_type="mmr",
        search_kwargs={"k": 3, "fetch_k": 4}
    )
    return retriever

# Configuração da RAG Chain
def config_rag_chain(retriever):
    def llm_fn(input) -> str:
        return call_together_api(str(input))

    context_q_prompt = ChatPromptTemplate.from_messages([
        ("system", "Formule uma pergunta autônoma com base no histórico do chat."),
        MessagesPlaceholder("chat_history"),
        ("human", "Pergunta: {input}")
    ])

    history_aware_retriever = create_history_aware_retriever(
        llm=llm_fn,
        retriever=retriever,
        prompt=context_q_prompt
    )

    qa_prompt = PromptTemplate.from_template(
        """
        Você é um assistente virtual especializado em Algoritmo Genético e Algoritmos de Machine Learning.
        Seu objetivo é fornecer respostas precisas e detalhadas com base na documentação disponível.

        1. Responda sempre em **português**, com clareza e objetividade.
        2. Caso a resposta envolva detalhes técnicos, explique de forma didática.
        3. Se não souber a resposta com base na documentação, diga que não possui informações suficientes.

        ### Contexto relevante da documentação:
        {context}

        ### Pergunta do usuário:
        {input}

        ### Responda de forma profissional:
        """
    )

    qa_chain = create_stuff_documents_chain(llm=llm_fn, prompt=qa_prompt)
    rag_chain = create_retrieval_chain(retriever=history_aware_retriever, combine_docs_chain=qa_chain)
    return rag_chain

# Lógica principal do chatbot
def chatbot_rag(question: str):
    try:
        retriever = config_retriever()
        rag_chain = config_rag_chain(retriever)
        response = rag_chain.invoke({"input": question})
        if isinstance(response, dict) and "answer" in response:
            return response["answer"]
        return str(response)
    except Exception as e:
        logger.error(f"Erro no processo RAG: {e}")
        return "Desculpe, não foi possível processar sua pergunta."
