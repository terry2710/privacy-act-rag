# 1. Import OS, Document Loader, Text Splitter, Bedrock Embeddings, Vector DB, Bedrock-LLM
import os
import time
import boto3
from botocore.config import Config
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_aws import BedrockEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain_aws import ChatBedrock
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


INDEX_DIR = "faiss_index"
INDEX_FILES = ["index.faiss", "index.pkl"]  # the two files FAISS.save_local() writes

# Set PRV_S3_BUCKET to enable S3 backup/restore of the index (e.g. "privacy-act-rag-index").
# Leave unset to use local-disk caching only.
S3_BUCKET = os.environ.get("PRV_S3_BUCKET", "")
S3_PREFIX = "faiss_index"

# No hardcoded profile name: boto3's default credential chain checks env vars first
# (AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY - what Streamlit Cloud secrets get copied into),
# then falls back to the local ~/.aws/credentials [default] profile. Works in both places.
AWS_REGION = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")


def _s3_client():
    return boto3.Session(region_name=AWS_REGION).client('s3')


def _download_index_from_s3(report):
    if not S3_BUCKET:
        return False
    client = _s3_client()
    os.makedirs(INDEX_DIR, exist_ok=True)
    try:
        for filename in INDEX_FILES:
            client.download_file(S3_BUCKET, f"{S3_PREFIX}/{filename}", os.path.join(INDEX_DIR, filename))
        return True
    except Exception as e:
        report(f"No cached index in S3 ({e})")
        return False


def _upload_index_to_s3(report):
    if not S3_BUCKET:
        return
    client = _s3_client()
    try:
        for filename in INDEX_FILES:
            client.upload_file(os.path.join(INDEX_DIR, filename), S3_BUCKET, f"{S3_PREFIX}/{filename}")
    except Exception as e:
        report(f"Failed to upload index to S3 ({e})")


# Bound worst-case wait per Bedrock call, and use "adaptive" retry mode so the client
# backs off its own send rate on ThrottlingException instead of just failing fast.
# New/free-tier accounts have low default Bedrock TPS quotas, so this matters.
BEDROCK_CONFIG = Config(
    connect_timeout=10,
    read_timeout=60,
    retries={"max_attempts": 10, "mode": "adaptive"},
)


# 5b. Wrap within a function
def prv_index(progress_callback=None, max_workers=1, min_interval=1):
    # progress_callback(stage: str, done: int | None, total: int | None) -> None, called at each step.
    def report(stage, done=None, total=None):
        if progress_callback:
            progress_callback(stage, done, total)

    # 4. Create Embeddings -- Client connection
    # v2 has 3x the rate quota of v1 (60 req/min vs 20 req/min, and v1's quota is not adjustable)
    data_embeddings = BedrockEmbeddings(
        region_name=AWS_REGION,
        model_id='amazon.titan-embed-text-v2:0',
        config=BEDROCK_CONFIG)

    # 0a. Local cache first (fastest, free) - if missing, try restoring from S3 before rebuilding.
    if not os.path.isdir(INDEX_DIR):
        report("Checking S3 for cached index")
        if _download_index_from_s3(report):
            report("Downloaded cached index from S3")

    if os.path.isdir(INDEX_DIR):
        report("Loading cached index")
        db_index = FAISS.load_local(INDEX_DIR, data_embeddings, allow_dangerous_deserialization=True)
        report("Loaded cached index", 1, 1)
        return db_index

    # 2. Define the data source and load data with PDFLoader
    report("Loading PDF")
    data_load = PyPDFLoader('https://www.legislation.gov.au/C2004A03712/2026-06-04/2026-06-04/text/original/pdf')
    documents = data_load.load()

    # 3. Split the Text based on Character, Tokens etc. - Recursively split by character - ["\n\n", "\n", " ", ""]
    report("Splitting text")
    data_split = RecursiveCharacterTextSplitter(separators=["\n\n", "\n", " ", ""], chunk_size=1500, chunk_overlap=200)
    chunks = data_split.split_documents(documents)

    # 5a. Embed chunks concurrently (embedding is network I/O, so threads help) and report progress as they land.
    total = len(chunks)
    texts = [chunk.page_content for chunk in chunks]
    metadatas = [chunk.metadata for chunk in chunks]
    vectors = [None] * total
    done = 0
    report("Embedding", 0, total)

    def embed_one(i):
        vector = data_embeddings.embed_query(texts[i])
        time.sleep(min_interval)  # small pacing gap as extra insurance against bursting the rate limit
        return i, vector

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(embed_one, i) for i in range(total)]
        try:
            for future in as_completed(futures):
                i, vector = future.result()
                vectors[i] = vector
                done += 1
                report("Embedding", done, total)
        except Exception as e:
            report(f"Embedding failed after {done}/{total} ({e})")
            raise

    # 5b. Build the Vector DB from the pre-computed embeddings, then cache it to disk and S3.
    db_index = FAISS.from_embeddings(list(zip(texts, vectors)), data_embeddings, metadatas=metadatas)
    report("Saving index to disk")
    db_index.save_local(INDEX_DIR)

    report("Uploading index to S3")
    _upload_index_to_s3(report)

    return db_index


# 6a. Write a function to connect to Bedrock Foundation Model - Claude Foundation Model
def prv_llm():
    # Anthropic models on this account are blocked pending the "Anthropic use case details" form
    # (see Bedrock console -> Model access). Using Amazon Nova Pro as a working stand-in until that
    # clears - swap back to model_id='anthropic.claude-sonnet-5' (or the inference-profile ID
    # 'us.anthropic.claude-sonnet-4-5-20250929-v1:0') once access is approved.
    llm = ChatBedrock(
        region_name=AWS_REGION,
        model_id='amazon.nova-pro-v1:0',
        max_tokens=3000,
        temperature=0.1)
    return llm


# 6b. Write a function which searches the user prompt, searches the best match from Vector DB and sends both to LLM.
def prv_rag_response(index, question):
    rag_llm = prv_llm()
    retriever = index.as_retriever()

    prompt = ChatPromptTemplate.from_template(
        "Answer the question based only on the following context:\n\n{context}\n\nQuestion: {question}"
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | rag_llm
        | StrOutputParser()
    )

    return rag_chain.invoke(question)
# Index creation --> https://python.langchain.com/docs/how_to/vectorstores/