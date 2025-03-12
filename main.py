import os
from dotenv import load_dotenv
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel

load_dotenv()

def main():
    external_client = AsyncOpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
    )


    translation_agent = Agent(name="英文专家",
     instructions="你是一个英文专家，擅长把中文翻译成英文",
        model=OpenAIChatCompletionsModel(
            model="gpt-4o-mini",
            openai_client=external_client,
        ),
    )

    writing_agent = Agent(name="中文专家",
        instructions="你是一个中文专家，擅长回答中文问题",
        model=OpenAIChatCompletionsModel(
            model="gpt-4o-mini",
            openai_client=external_client,
        ),
        handoffs=[translation_agent]
    )

    result = Runner.run_sync(writing_agent, "写一首关于春天的诗,并且翻译成英文")
    print(result.final_output)

if __name__ == "__main__":
    main()
