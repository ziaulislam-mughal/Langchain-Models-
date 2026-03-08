from langchain_openai import OpenAIEmbeddings
from  dotenv import load_dotenv 
load_dotenv()


embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
    dimensions=32 
)


#single query 
results = embeddings.embed_query("Hello world", "I am a cat")
print(results)