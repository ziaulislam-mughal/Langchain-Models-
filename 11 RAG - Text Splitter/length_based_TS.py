from langchain_text_splitters import CharacterTextSplitter

text = """
Artificial Intelligence is transforming industries across the world.
Machine learning models are being used in healthcare, finance, and transportation.
Large language models help machines understand human language.
Retrieval Augmented Generation allows models to answer questions using external documents.
Text splitting is a key step in building RAG systems.
"""

#Creat text splitter 

text_splitter = CharacterTextSplitter(
    separator = '\n',
    chunk_size = 100
)

# split text 

chunks = text_splitter.split_text(text)

for i , chunk in enumerate(chunks):
    print(f"chunks {i+1}:")
    print(chunk)
    print("-----------------------------------------")