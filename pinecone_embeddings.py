import os
import json
from typing import Any, Dict, List
from uuid import uuid4
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone
from dotenv import load_dotenv
from PyPDF2 import PdfReader

# Carregar variáveis de ambiente do .env
load_dotenv()

# Diretórios locais
PDF_FOLDER = r"C:\Users\Usuário\Desktop\chatbot-rag-2025\arquivos-pdf"
JSON_FOLDER = r"C:\Users\Usuário\Desktop\chatbot-rag-2025\arquivos-json"

# Configurações do Pinecone
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "chatbot-rag-2025")
PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE", "ns1")

# Inicialização do Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

# ETAPA 1: EXTRAÇÃO DOS PDFs PARA JSONs

def extrair_texto_pdf(caminho_pdf: str) -> str:
    try:
        reader = PdfReader(caminho_pdf)
        texto = "\n".join([page.extract_text() or "" for page in reader.pages])
        return texto
    except Exception as e:
        print(f"Erro ao extrair texto de {caminho_pdf}: {e}")
        return ""

def quebrar_em_chunks(texto: str, max_palavras: int = 100) -> List[str]:
    palavras = texto.split()
    chunks = []
    for i in range(0, len(palavras), max_palavras):
        chunk = " ".join(palavras[i:i+max_palavras])
        chunks.append(chunk)
    return chunks

def salvar_chunks_como_json(nome_pdf: str, chunks: List[str]):
    for i, chunk in enumerate(chunks):
        json_doc = {
            "id": str(uuid4()),
            "origem": nome_pdf,
            "paragraph": chunk,
            "page": i + 1
        }
        nome_arquivo = os.path.join(JSON_FOLDER, f"{nome_pdf.replace('.pdf', '')}_chunk{i}.json")
        with open(nome_arquivo, "w", encoding="utf-8") as f:
            json.dump(json_doc, f, ensure_ascii=False, indent=2)

        print(f"Chunk {i+1} salvo: {nome_arquivo}")

def processar_pdfs():
    pdfs = [f for f in os.listdir(PDF_FOLDER) if f.endswith(".pdf")]
    if not pdfs:
        print("Nenhum PDF encontrado.")
        return

    for pdf in pdfs:
        caminho = os.path.join(PDF_FOLDER, pdf)
        print(f"Extraindo texto de: {pdf}")
        texto = extrair_texto_pdf(caminho)
        if texto:
            chunks = quebrar_em_chunks(texto)
            salvar_chunks_como_json(pdf, chunks)

# ETAPA 2: EMBEDDINGS E UPLOAD NO PINECONE

def carregar_json_local(caminho_arquivo: str) -> Dict[str, Any]:
    try:
        with open(caminho_arquivo, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Erro ao carregar JSON {caminho_arquivo}: {e}")
        return {}

model = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")

def get_embeddings(texts: List[str]) -> List[List[float]]:
    return [model.embed_query(text) for text in texts]

def create_embeddings(dados: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    embeddings = []
    texts = [item.get("paragraph", "") for item in dados]
    vectors = get_embeddings(texts)

    for i, item in enumerate(dados):
        try:
            vector = [float(v) for v in vectors[i]]
            embeddings.append({
                "id": item.get("id", str(uuid4())),
                "values": vector,
                "metadata": {
                    "text": item.get("paragraph", ""),
                    "page": str(item.get("page", "")),
                    "origem": item.get("origem", "")
                }
            })
        except Exception as e:
            print(f"Erro ao processar embedding: {e}")

    return embeddings

def upload_to_pinecone(embeddings: List[Dict[str, Any]], batch_size: int = 100):
    try:
        for i in range(0, len(embeddings), batch_size):
            batch = embeddings[i:i + batch_size]
            index.upsert(vectors=batch, namespace=PINECONE_NAMESPACE)
            print(f"Lote {i // batch_size + 1} enviado com sucesso!")
    except Exception as e:
        print(f"Falha ao enviar para Pinecone: {e}")

def processar_jsons():
    arquivos_json = [f for f in os.listdir(JSON_FOLDER) if f.endswith(".json")]
    if not arquivos_json:
        print("Nenhum arquivo JSON encontrado.")
        return

    for arquivo in arquivos_json:
        caminho = os.path.join(JSON_FOLDER, arquivo)
        print(f"Processando: {arquivo}")
        dados = [carregar_json_local(caminho)]
        if not dados:
            continue

        embeddings = create_embeddings(dados)
        if not embeddings:
            continue

        upload_to_pinecone(embeddings)
        print(f"Finalizado: {arquivo}")


# MAIN
def main():
    print("Iniciando extração dos PDFs...")
    processar_pdfs()
    print("Iniciando geração de embeddings...")
    processar_jsons()

if __name__ == '__main__':
    main()
