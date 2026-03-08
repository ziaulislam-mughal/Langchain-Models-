from langchain_huggingface import ChatHuggingFace , HuggingFacePipeline
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

llm = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    pipeline_kwargs={"max_length": 2048, "temperature": 0.7, "top_p": 0.9},
)


model = ChatHuggingFace(llm=llm)