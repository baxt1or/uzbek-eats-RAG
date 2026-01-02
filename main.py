from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever

model = OllamaLLM(model="phi3:mini")

template = """ 

You are an expert assistant analyzing restaurant reviews written in Uzbek.

Your task:
1. Use ONLY the provided reviews.
2. Answer the user's question briefly and clearly.
3. First, provide recommendations in Uzbek.
4. Then, provide the same recommendations in Russian.
5. Do NOT add information that is not supported by the reviews.

Formatting rules:
- Uzbek response first
- Russian response second
- Use short bullet points if possible
- Be concise and practical

Relevant reviews (Uzbek):
{reviews}

User question:
{question}

"""

prompt = ChatPromptTemplate.from_template(template=template)
chain = prompt | model



while True:

    question = input("Ask a questions q (quit): ")
    
    if question =="q":
        break
    
    reviews = retriever.invoke(question)
    res = chain.invoke({"reviews": reviews, "question": question})

    print(res)