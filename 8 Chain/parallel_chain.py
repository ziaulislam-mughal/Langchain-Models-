import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

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

# 4. Wrap it in ChatHuggingFace (This handles the "conversational" format for you)
model_1 = ChatHuggingFace(llm=llm)
model_2 = ChatHuggingFace(llm=llm)

# 5. Define Prompts
prompt1 = PromptTemplate(
    template="Generate short and simple notes from following text:\n {text}",
    input_variables=["text"],
)

prompt2 = PromptTemplate(
    template="Generate a five short question answers from following : \n {text}",
    input_variables=["text"],
)

prompt3 = PromptTemplate(
    template="""Merge following notes and question answers into a single structured text : \n " \
    "Notes: {notes} \n " \
    "Question Answers: {quiz}""",
    input_variables=["notes", "quiz"],
)

parser = StrOutputParser()

# 6. Build the Parallel Chain
parallel_chain = RunnableParallel({
    "notes": prompt1 | model_1 | parser,
    "quiz": prompt2 | model_2 | parser,
})

# 7. Build the Merge Chain
merge_chain = prompt3 | model_1 | parser

# 8. Combine into a Single Pipeline
chain = parallel_chain | merge_chain

# 9. Test Data
text = """What is Machine Learning?
Last Updated : 13 Sep, 2025
Machine learning is a branch of artificial intelligence that enables algorithms to uncover hidden patterns within datasets. It allows them to predict new, similar data without explicit programming for each task. Machine learning finds applications in diverse fields such as image and speech recognition, natural language processing, recommendation systems, fraud detection, portfolio optimization, and automating tasks.

Machine-Learning-Techniques
Machine Learning Techniques
Handles Massive Data: Machine learning works well with large data and finds patterns that humans might miss.
Adapts Dynamically: Systems evolve with new data, staying relevant in changing environments.
Drives Smarter Decisions: From predicting customer behavior to detecting fraud, ML enhances decision-making with data-driven insights.
Personalizes Experiences: Recommendation systems, like those on Netflix or Amazon, tailor suggestions to individual preferences.
Types of Machine Learning
Machine learning algorithms can be broadly categorized into three main types based on their learning approach and the nature of the data they work with.

Supervised Learning
Involves training models using labeled datasets. Both input and output variables are provided during training.
The aim is to establish a mapping function that predicts outcomes for new, unseen data.
Common applications include classification, regression, and forecasting.
Unsupervised Learning
Works with unlabeled data where outputs are not known in advance.
The model identifies hidden structures, relationships, or groupings in the data.
Useful for clustering, dimensionality reduction, and anomaly detection.
Focuses on discovering inherent patterns within datasets.
Reinforcement Learning
Based on decision-making through interaction with an environment.
An agent performs actions and receives rewards or penalties as feedback.
The goal is to learn an optimal strategy that maximizes long-term rewards.
Widely applied in robotics, autonomous systems, and strategic game playing.
Want to learn Machine Learning from scratch, refer to our guide ML Tutorial.

Real-World Application of Machine Learning
Here are some specific areas where machine learning is being used:

Predictive modelling: Machine learning can be used to build predictive models that can help businesses make better decisions. For example, machine learning can be used to predict which customers are most likely to buy a particular product, or which patients are most likely to develop a certain disease.
Natural language processing: Machine learning is used to build systems that can understand and interpret human language. This is important for applications such as voice recognition, chatbots, and language translation.
Computer vision: Machine learning is used to build systems that can recognize and interpret images and videos. This is important for applications such as self-driving cars, surveillance systems, and medical imaging.
Fraud detection: Machine learning can be used to detect fraudulent behavior in financial transactions, online advertising, and other areas.
Recommendation systems: Machine learning can be used to build recommendation systems that suggest products, services, or content to users based on their past behaviour and preferences.
"""

# 10. Execute the Chain safely using a dictionary
print("Processing data... Please wait.")
result = chain.invoke({"text": text})

print("\n--- Final Result ---\n")
print(result)