from langchain.chat_models import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from configs.settings import Settings

class OpenAIClient:
    """
    A simple wrapper around OpenAI's LLM and embeddings.

    Provides easy methods to:
    - Generate chat responses from a prompt.
    - Generate embeddings for text.
    """

    def __init__(self):
        self.llm = ChatOpenAI(
            model=Settings.OPENAI_LLM_MODEL,
            openai_api_key=Settings.OPENAI_API_KEY
        )
        self.embeddings = OpenAIEmbeddings(
            model=Settings.OPENAI_EMBED_MODEL,
            openai_api_key=Settings.OPENAI_API_KEY
        )

    def chat(self, prompt: str) -> str:
        return self.llm.invoke(prompt).content

    def get_embedding(self, text: str):
        return self.embeddings.embed_query(text)


# Singleton
openai_client = OpenAIClient()
