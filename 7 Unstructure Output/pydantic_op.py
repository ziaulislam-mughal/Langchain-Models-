import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

# 1. Load Environment
load_dotenv()

# 2. Check Token
token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if not token:
    print("Error: HUGGINGFACEHUB_API_TOKEN is not set in the .env file.")

# 3. Setup the Endpoint
repo_id = "meta-llama/Llama-3.1-8B-Instruct"

llm = HuggingFaceEndpoint(
    repo_id=repo_id,
    task="text-generation",
    max_new_tokens=512,
    temperature=0.7,
    huggingfacehub_api_token=token,
)

# 4. Wrap it in ChatHuggingFace
model = ChatHuggingFace(llm=llm)

# 5. Create Pydantic Model
class Person(BaseModel):
    name: str = Field(..., description="The person's name")
    age: int = Field(gt=18, description="The person's age")
    city: str = Field(..., description="The city where the person lives")

# 6. Create Output Parser
parser = PydanticOutputParser(pydantic_object=Person)

# 7. Create Prompt Template
template = PromptTemplate(
    template="""
Generate details of a person from {place}.

Return ONLY a valid JSON object.
Do not return python code.
Do not return explanations.

{format_instructions}
""",
    input_variables=["place"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

# 8. Format Prompt
# prompt = template.format(place="Pakistan")

# # 9. Call the Model
# result = model.invoke(prompt)

# # 10. Parse Output
# final_result = parser.parse(result.content)

# print(final_result)


chain = template |model | parser
result = chain.invoke({"place": "Pakistan"})
print(result)