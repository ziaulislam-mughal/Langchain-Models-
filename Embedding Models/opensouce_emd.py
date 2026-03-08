from langchain_huggingface import HuggingFaceEmbeddings 

emd = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")

text = "Hello world, I am a cat"
result = emd.embed_query(text)