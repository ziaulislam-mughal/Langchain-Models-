from langchain_openai import OpenAIEmbeddings
from  dotenv import load_dotenv 
load_dotenv()


embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
    dimensions=32 
)


documents = [ 
    "What is the capital of France?",
    "Who is the president of the United States?",
    "What is the largest mammal?"
]

results = embeddings.embed_documents(documents)
print(results)