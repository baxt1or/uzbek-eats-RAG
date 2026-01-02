from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever

model = OllamaLLM(model="phi3:mini")

template = """ 
You are an expert responding restaurant reviews and reviews are in English, but you need to answer briefly in Russian

Here are the relevant reviews: {reviews}

Here is the question to answer: {question}

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