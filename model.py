from ctransformers import AutoModelForCausalLM
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.callbacks.manager import CallbackManager
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import LlamaCpp
from langchain.chains import RetrievalQA, LLMChain
import chainlit as cl

DB_FAISS_PATH = 'vectorstore/db_faiss2'

custom_prompt_template = """Use the all following pieces of information to answer the user's question.
Finish all sentences in your answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""


def set_custom_prompt():
    """
    Prompt template for QA retrieval for each vectorstore
    """
    prompt = PromptTemplate(template=custom_prompt_template,
                            input_variables=['context', 'question'])
    return prompt


# Retrieval QA Chain
def retrieval_qa_chain(llm, prompt, db):
    LLMChain
    qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                           chain_type='stuff',
                                           retriever=db.as_retriever(search_kwargs={'k': 4}),
                                           return_source_documents=True,
                                           chain_type_kwargs={'prompt': prompt},

                                           )
    return qa_chain


# Loading the model
def load_llm():
    # Load the locally downloaded model here
    n_gpu_layers = 43
    n_batch = 2000
    llm = LlamaCpp(
        model_path="data/openbuddy-llama2-13b-v11.1.Q2_K.gguf",
        # model_type="llama",
        n_gpu_layers=n_gpu_layers,
        n_batch=n_batch,
        temperature=0.8,
        top_p=1,
        max_tokens=200000,
        callback_manager=CallbackManager([StreamingStdOutCallbackHandler]),
        verbose=True,
        n_ctx=2048

    )
    return llm


# QA Model Function
def qa_bot():
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
                                       model_kwargs={'device': 'cuda'})
    db = FAISS.load_local(DB_FAISS_PATH, embeddings)
    llm = load_llm()
    qa_prompt = set_custom_prompt()
    qa = retrieval_qa_chain(llm, qa_prompt, db)

    return qa


# output function
def final_result(query):
    qa_result = qa_bot()
    response = qa_result({'query': query})
    return response


# chainlit code
@cl.on_chat_start
async def start():
    chain = qa_bot()
    msg = cl.Message(content="Starting the bot...")
    await msg.send()
    msg.content = "Hi, Welcome to Marketing Bot. What is your query?"
    await msg.update()

    cl.user_session.set("chain", chain)


@cl.on_message
async def main(message):
    chain = cl.user_session.get("chain")
    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"]
    )
    cb.answer_reached = True
    res = await chain.acall(message, callbacks=[cb])
    answer = res["result"]
    sources = res["source_documents"]

    if sources:
        answer += f"\nSources:" + str(sources)
    else:
        answer += "\nNo sources found"

    await cl.Message(content=answer).send()


# print(final_result('кто такой ойбек?'))
