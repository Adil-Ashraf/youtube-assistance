from langchain.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI


load_dotenv()

embeddings = OpenAIEmbeddings()
video_url = "https://www.youtube.com/watch?v=-Osca2Zax4Y"


def create_vector_db_from_youtube_url(video_url: str) -> FAISS:
    loader = YoutubeLoader.from_youtube_url(video_url)
    transcript = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(transcript)

    db = FAISS.from_documents(docs, embeddings)
    return db


def get_response_from_query(db, query, k=4):
    docs = db.similarity_search(query, k=k)
    docs_page_content = " ".join([d.page_content for d in docs])

    llm = ChatOpenAI(model_name="gpt-3.5-turbo-16k")

    # prompt = PromptTemplate(
    #     input_variables=["question", "docs"],
    #     template="""
    #     You are a helpful assistant that that can answer questions about youtube videos 
    #     based on the video's transcript.

    #     Answer the following question: {question}
    #     By searching the following video transcript: {docs}

    #     Only use the factual information from the transcript to answer the question.

    #     If you feel like you don't have enough information to answer the question, say "I don't know".

    #     Your answers should be verbose and detailed.
    #     """,
    # )

    prompt = PromptTemplate(
        input_variables=["question", "docs"],
        template="""
        You are an assistant capable of analyzing YouTube video transcripts. Provide insights on the query below:

        Question: {question}
        Utilize the video transcript provided:

        Transcript: {docs}
        Respond with concise information, limiting your response to 80 words. If the transcript lacks sufficient detail for a definitive answer, state "I don't know." Your response should be clear, factual, and directly derived from the provided transcript.
        """,
    )
    chain = LLMChain(llm=llm, prompt=prompt)

    response = chain.run(question=query, docs=docs_page_content)
    # response = response.replace("\n", "")
    return response, docs
# print(create_vector_db_from_youtube_url("https://www.youtube.com/watch?v=-Osca2Zax4Y"))