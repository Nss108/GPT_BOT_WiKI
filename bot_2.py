import os
from aiogram import Bot, Dispatcher, types
from aiogram.utils import executor
from dotenv import load_dotenv
from llama_index import SimpleDirectoryReader, GPTVectorStoreIndex, LLMPredictor, PromptHelper, ServiceContext, StorageContext, load_index_from_storage
from langchain import OpenAI

load_dotenv() 

API_TOKEN = os.getenv("API_TOKEN")
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")

bot = Bot(token=API_TOKEN)
dp = Dispatcher(bot)

def create_index(work_bot_2):
    max_input = 3000
    tokens = 600
    chunk_size = 600
    max_chunk_overlap = 50
    
    promptHelper = PromptHelper(max_input, tokens, max_chunk_overlap, chunk_size_limit=chunk_size)
    
    llmPredictor = LLMPredictor(llm=OpenAI(temperature=0.9, model_name="text-embedding-ada-002", max_tokens=tokens))
    
    docs = SimpleDirectoryReader(work_bot_2).load_data()

    service_context = ServiceContext.from_defaults(llm_predictor=llmPredictor, prompt_helper=promptHelper)
    
    vectorIndex = GPTVectorStoreIndex.from_documents(documents=docs, service_context=service_context)
    vectorIndex.storage_context.persist(persist_dir = 'Store')
    print("Index has been created and saved successfully")

def answerMe(question):
    storage_context = StorageContext.from_defaults(persist_dir = 'Store')
    index = load_index_from_storage(storage_context)
    print("Index has been loaded successfully")
    query_engine = index.as_query_engine()
    response = query_engine.query(question)
    return response

create_index('/Users/nikolajsutugin/Desktop/GPT_bot/work_bot_2/text')  # Вызываем функцию create_index с путем к текстовым файлам

@dp.message_handler()
async def prompt(message: types.Message):
    prompt_text = message.text
    response = answerMe(prompt_text)
    print(f"Response: {response}")
    await message.reply(response)

if __name__ == '__main__':
    executor.start_polling(dp, skip_updates=True)