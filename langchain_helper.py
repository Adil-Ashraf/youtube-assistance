from langchain.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from utils.stream_handler import StreamHandler

load_dotenv()

embeddings = OpenAIEmbeddings()
video_url = "https://www.youtube.com/watch?v=-Osca2Zax4Y"


def create_vector_db_from_youtube_url(video_url: str) -> FAISS:
    loader = YoutubeLoader.from_youtube_url(video_url)
    transcript = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )

    docs = text_splitter.split_documents(transcript)

    db = FAISS.from_documents(docs, embeddings)
    return db


def get_response_from_query(db, query, chat_box, k=4):
    docs = db.similarity_search(query, k=k)
    docs_page_content = " ".join([d.page_content for d in docs])

    stream_handler = StreamHandler(chat_box, display_method='write')

    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo-16k",
        streaming=True,
        callbacks=[stream_handler]
    )

    prompt = PromptTemplate(
        input_variables=["question", "docs"],
        template="""
        You are an assistant capable of analyzing YouTube video transcripts. Provide insights on the query below:

        Question: {question}
        Utilize the video transcript provided:

        Transcript: {docs}
        Respond with bullet-pointed and concise information.
        If the transcript lacks sufficient detail for a definitive answer, state "I don't know." Your response should be clear, factual, and directly derived from the provided transcript.
        """,
    )
    chain = LLMChain(llm=llm, prompt=prompt)

    response = chain.run(question=query, docs=docs_page_content)
    response = response.replace("\n", "")
    return response
