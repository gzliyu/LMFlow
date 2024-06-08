from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.llms import HuggingFaceEndpoint
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables import Runnable
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage

# retriever usage
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

import re
import os
import argparse
import logging
logging.getLogger().setLevel(logging.ERROR) # hide warning log


class LangchainChatbot:
    def __init__(self,
                 model_name_or_path: str,
                 provider: str):
        self.prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content="You are a helpful chatbot."),
                MessagesPlaceholder(variable_name="history"),
                HumanMessagePromptTemplate.from_template("{input}"),
                MessagesPlaceholder(variable_name="retriever", optional=True)
            ]
        )
        self.model_name_or_path = model_name_or_path
        self.provider = provider
        self.check_valid_provider()
        self.model = self.get_model()
        self.retriever = None
        self.memory = {}
        self.runnable: Runnable = self.prompt | self.model
        self.llm_chain = RunnableWithMessageHistory(
            self.runnable,
            self.get_session_history,
            input_messages_key="input",
            history_messages_key="history",
        )

    def check_valid_provider(self):
        provider = self.provider
        model_name_or_path = self.model_name_or_path
        if provider == "openai" and 'gpt' in model_name_or_path:
            if os.getenv("OPENAI_API_KEY"):
                return
            raise OSError("OPENAI_API_KEY environment variable is not set.")
        elif provider == "anthropic" and 'claude' in model_name_or_path:
            if os.getenv("ANTHROPIC_API_KEY"):
                return
            raise OSError("ANTHROPIC_API_KEY environment variable is not set.")
        elif provider == "google" and 'gemini' in model_name_or_path:
            if os.getenv("GOOGLE_API_KEY"):
                return
            raise OSError("GOOGLE_API_KEY environment variable is not set.")
        elif provider == "huggingface":
            if os.getenv("HUGGINGFACEHUB_API_TOKEN"):
                return
            raise OSError("HUGGINGFACEHUB_API_TOKEN environment variable is not set.")
        raise ValueError("Invalid provider or model_name_or_path.")

    def set_retriever_url(self, url):
        loader = WebBaseLoader(url)
        data = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
        all_splits = text_splitter.split_documents(data)
        vectorstore = Chroma.from_documents(documents=all_splits, embedding=OpenAIEmbeddings())
        self.retriever = vectorstore.as_retriever(k=4)

    def get_model(self):
        provider = self.provider
        model_name_or_path = self.model_name_or_path
        if provider == "openai":
            model = ChatOpenAI(model=model_name_or_path)
        elif provider == "anthropic":
            model = ChatAnthropic(model=model_name_or_path)
        elif provider == "google":
            model = ChatGoogleGenerativeAI(model=model_name_or_path)
        else:
            model = HuggingFaceEndpoint(repo_id=model_name_or_path)
        return model

    def chat_with_chatbot(self, human_input):
        if self.retriever:
            retriever_search = self.retrieve_by_retriever(human_input)
            response = self.llm_chain.invoke({"input": human_input,
                                              "retriever": [retriever_search]},
                                             config={"configurable": {"session_id": "abc123"}})
        else:
            response = self.llm_chain.invoke({"input": human_input},
                                             config={"configurable": {"session_id": "abc123"}})
        return response if self.provider == "huggingface" else response.content

    def retrieve_by_retriever(self, query):
        return '\n'.join(re.sub('\n+', '\n', dict(result)['page_content']) for result in self.retriever.invoke(query))

    def retrieve_by_memory(self, keyword):
        return [msg.content for msg in self.memory.chat_memory.messages if keyword in msg.content]

    def get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        if session_id not in self.memory:
            self.memory[session_id] = ChatMessageHistory()
        return self.memory[session_id]


def get_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
    )
    parser.add_argument(
        "--model-name-or-path", type=str, help="Model name"
    )
    parser.add_argument(
        "--provider", type=str, help="Provider of the model"
    )
    parser.add_argument(
        "--set-url", action="store_true", help="URL for retrieval"
    )
    parser.add_argument(
        "--save-history", action="store_true", help="Save chat history if enabled"
    )
    return parser


def main(model_name_or_path: str,
         provider: str,
         set_url: bool ,
         save_history: bool
         ):
    chatbot = LangchainChatbot(model_name_or_path=model_name_or_path,
                               provider=provider)
    if set_url:
        url = input("Please set your url: ")
        chatbot.set_retriever_url(url)
    while True:
        human_input = input("user: ")
        if human_input == "exit":
            break
        response = chatbot.chat_with_chatbot(human_input)
        print(f"chatbot: {response}")
    if save_history:
        if '/' in model_name_or_path:
            model_name_or_path = model_name_or_path.split('/')[1]
        if not os.path.exists("chat_history"):
            os.mkdir("chat_history")
        with open(f"chat_history/{model_name_or_path}.txt", 'w') as file:
            file.write(str(chatbot.memory['abc123'].messages))


if __name__ == "__main__":
    args = get_cli().parse_args()
    main(**vars(args))
