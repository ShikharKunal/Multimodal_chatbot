import os
from dotenv import load_dotenv
import chainlit as cl
from pdf_processor import preprocess
from vectorize import vectorize_
from chainlit.message import HumanMessage
from langchain.llms import HuggingFaceEndpoint
from langchain.chat_models import ChatHuggingFace
from langchain.schema import RunnableParallel, RunnableLambda, RunnablePassthrough, StrOutputParser

load_dotenv()
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

@cl.on_chat_start
async def on_chat_start():
    print("Chat started")
    await cl.send_message("Hello! I am an AI model that can answer your queries from a PDF. You can ask me questions about the given context.")

    files = None

    while files is None:
        files = await cl.AskFileMessage(
            content="Please upload the PDF file you want me to summarize",
            accept=["application/pdf"],
            max_size_mb=25,
            timeout=180
        ).send()

    file = files[0]

    print("File received")

    msg = await cl.send_message("Please wait while I process the PDF file...")

    text_chunks, images, category_counts = preprocess(file)

    print("Processing done")

    vectorstore = await cl.make_async(vectorize_)(images, text_chunks)

    retriever = vectorstore.as_retriever()

    print("Vectorization done")

    llm = HuggingFaceEndpoint(
        repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
        task="text-generation",
        max_new_tokens=1024,
        max_length=2048,
        do_sample=True,
        repetition_penalty=1.03,
    )

    chat_model = ChatHuggingFace(llm=llm)

    print("Model loaded")

    def prompt_func(inputs):
        context_ = ""

        for doc in inputs["context"]:
            context_ += doc["content"] + "\n"

        prompt = """ 
            You are an advanced AI model that can understand both text and images.
            You have been given summarized text and images from a PDF document.
            You have to generate a response based on the given context.
            Answer only the question asked. Do not provide any additional information.
            The context is:
            {}
            The question is:
            {}
        """.format(context_, inputs["question"])
        return HumanMessage(content=prompt)

    chain = (
        {
            "context": retriever,
            "question": RunnablePassthrough(),
        }
        | RunnableParallel({"response": prompt_func | chat_model | StrOutputParser(),
                            "context": lambda x: x["context"]})
    )

    await msg.update("The PDF file has been processed successfully. You can now ask me questions about the given context.")

    cl.user_session.set("chain", chain)

@cl.on_message
async def main(message: str):
    chain = cl.user_session.get("chain")
    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True, answer_prefix_tokens=["Answer"]
    )
    result = await chain.acall({"question": message}, callback=cb)

    if not cb.answer_reached:
        await cl.send_message(result["response"])

    context = result["context"]

    # Check if the context has an image
    if context["is_img"]:
        print("Sending image")
        img_path = context["id"]
        if img_path:
            with open(img_path, 'rb') as img_file:
                await cl.send_image(content=img_file)