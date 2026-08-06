import time
import rag_backend as demo

print("=== Step 1: Loading PDF ===")
t0 = time.time()
data_load = demo.PyPDFLoader('https://www.legislation.gov.au/C2004A03712/2026-06-04/2026-06-04/text/original/pdf')
documents = data_load.load()
print(f"Loaded {len(documents)} pages in {time.time()-t0:.1f}s")

print("=== Step 2: Splitting ===")
t0 = time.time()
splitter = demo.RecursiveCharacterTextSplitter(separators=["\n\n", "\n", " ", ""], chunk_size=1500, chunk_overlap=200)
chunks = splitter.split_documents(documents)
print(f"Split into {len(chunks)} chunks in {time.time()-t0:.1f}s")

print("=== Step 3: Testing a single embedding call ===")
t0 = time.time()
emb = demo.BedrockEmbeddings(credentials_profile_name='default', model_id='amazon.titan-embed-text-v1')
vec = emb.embed_query(chunks[0].page_content)
print(f"Single embed_query took {time.time()-t0:.1f}s, vector dim={len(vec)}")

print("=== Step 4: Embedding all chunks into FAISS (this is the slow part) ===")
t0 = time.time()
db = demo.FAISS.from_documents(chunks, emb)
print(f"Built FAISS index in {time.time()-t0:.1f}s")
