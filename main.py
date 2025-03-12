import os
from dotenv import load_dotenv
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel

load_dotenv()

def main():
    external_client = AsyncOpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
    )

    english_agent = Agent(name="英文专家",
        instructions="你是一个英文专家，擅长把中文翻译成英文",
        model=OpenAIChatCompletionsModel(
            model="gpt-4o-mini",
            openai_client=external_client,
        )
    )

    french_agent = Agent(name="法语专家",
        instructions="你是一个法语专家，擅长把中文翻译成法语",
        model=OpenAIChatCompletionsModel(
            model="gpt-4o-mini",
            openai_client=external_client,
        ),
        handoffs=[english_agent]
    )

    writing_agent = Agent(name="中文专家",
        instructions="你是一个中文专家，擅长回答中文问题",
        model=OpenAIChatCompletionsModel(
            model="gpt-4o-mini",
            openai_client=external_client,
        ),
        handoffs=[english_agent, french_agent]
    )

    triage_agent = Agent(name="问题分类专家",
        instructions="你是一个问题分类专家，擅长根据问题分类",
        model=OpenAIChatCompletionsModel(
            model="gpt-4o-mini",
            openai_client=external_client,
        ),
        handoffs=[writing_agent, english_agent, french_agent]
    )

    result = Runner.run_sync(triage_agent, "把后面的话翻译成英文： 你好，我是一个人")
    print(result.final_output)

if __name__ == "__main__":
    main()
