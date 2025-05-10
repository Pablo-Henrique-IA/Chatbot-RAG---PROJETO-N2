import chainlit as cl
from rag_chainlit import chatbot_rag

@cl.on_chat_start
async def welcome():
    await cl.Message(content="Sou o Chatbot Especialista em AG e Algoritmos de Machine Learning. Faça sua pergunta!").send()

@cl.on_message
async def main(message: cl.Message):
    user_input = message.content.strip()
    if user_input:
        resposta = await cl.make_async(chatbot_rag)(user_input)
        await cl.Message(content=resposta).send()
    else:
        await cl.Message(content="Por favor, digite uma pergunta válida.").send()
