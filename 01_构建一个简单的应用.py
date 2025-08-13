from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM

model = OllamaLLM(
    model="gemma3:4b",
    base_url="http://localhost:11434",
)

system_template = "Translate the following from English into {language}"
prompt_template = ChatPromptTemplate.from_messages(
    [("system", system_template), ("user", "{text}")]
)
prompt = prompt_template.invoke({"language": "Chinese", "text": "hi!"})

for chunk in model.stream(prompt):
    print(chunk, end="")
