import pandas as pd 
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os

reviews_df = pd.read_csv("./uzbek_restaurant_reviews.csv").iloc[:200, :]

embd = OllamaEmbeddings(model="phi3:mini")

db_location = "./chroma_langchain_db"
add_documents = not os.path.exists(db_location)


if add_documents:
    documents = []
    idxs = []

    for i, row in reviews_df.iterrows():
        document = Document(
            page_content = str(row["review"]),
            metadata = {"rating" : row["rating"]},
            id=str(i)
        )
        idxs.append(str(i))
        documents.append(document)


vector_store = Chroma(
    collection_name = "uzbek_restaurant_reviews",
    persist_directory=db_location,
    embedding_function=embd
)


if add_documents:
    vector_store.add_documents(documents=documents, ids=idxs)


retriever = vector_store.as_retriever(
   search_kwargs={"k":3}
)