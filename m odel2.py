
from langchain import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA, LLMChain
import chainlit as cl
from langchain import HuggingFacePipeline

DB_FAISS_PATH = 'vectorstore/db_faiss'

custom_prompt_template = """
Context: {context}
Question: {question}


"""


def set_custom_prompt():
    """
    Prompt template for QA retrieval for each vectorstore
    """
    prompt = PromptTemplate(template=custom_prompt_template,
                            input_variables=['context', 'question'])
    print(prompt)
    return prompt


# Retrieval QA Chain
def retrieval_qa_chain(llm, prompt, db):
    LLMChain
    qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                           chain_type='stuff',
                                           retriever=db.as_retriever(search_kwargs={'k': 5}),
                                           return_source_documents=True,
                                           chain_type_kwargs={'prompt': prompt},

                                           )
    return qa_chain


# Loading the model

from transformers import pipeline, GPT2Tokenizer, GPT2Model, GPT2LMHeadModel, LlamaTokenizer, LlamaForCausalLM
import torch
def load_llm():
    # Load the locally downloaded model here
    tokenizer = GPT2Tokenizer.from_pretrained('mGPT')
    model = GPT2LMHeadModel.from_pretrained('mGPT',
                                      # load_in_8bit=True,
                                      device_map="auto",
                                      torch_dtype=torch.float16,
                                      low_cpu_mem_usage=True,
                                      )

    pipe = pipeline(
        'text-generation',
        model=model,
        tokenizer=tokenizer,
        max_length=1024,
        temperature=1.0,
        top_p=0.95,
        repetition_penalty=1.15,
        do_sample=True
    )


    # device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # llm = HuggingFacePipeline.from_model_id(task='text-generation',
    #                       model_id='mGPT',
    #                       model_kwargs={"temperature": 0, "max_length": 2000},
    #                       device=0 if device == 'cuda' else -1,
    #                       )

    llm = HuggingFacePipeline(pipeline=pipe)
    print(llm)
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


print(final_result('виды рекламы?'))
