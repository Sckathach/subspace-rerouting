import json
import os

import dotenv
from ollama import Options, chat
from openai import OpenAI
from rich import print

from ssr.evaluation.types import CallParameters, JudgeScore
from ssr.files import get_template

dotenv.load_dotenv()

surname_to_lmstudio = {
    "gemma2_2b": "gemma-2-2b-it",
    "llama3.2_1b": "llama-3.2-1b-instruct",
    "llama3.2_3b": "llama-3.2-3b-instruct",
    "qwen2.5_1.5b": "qwen2.5-1.5b-instruct",
}
surname_to_ollama = {
    "gemma2_2b": "gemma2:2b",
    "llama3.2_1b": "llama3.2:1b",
    "llama3.2_3b": "llama3.2:3b",
    "qwen2.5_1.5b": "qwen2.5:1.5b",
}


def call_guard(content: str, guard_name: str = "llama-guard3:1b") -> bool:
    try:
        response = chat(
            model=guard_name,
            messages=[{"role": "user", "content": content}],
        ).message.content  # type: ignore
        if response is None:
            return True

        return "unsafe" in response
    except Exception as e:
        raise Exception("You have to configure the guard with Ollama.") from e


def call_ollama(model_name: str, prompt: str, **kwargs) -> str:
    params = CallParameters(**kwargs)
    messages = [
        {"role": "user", "content": prompt},
    ]

    if params.system_message is not None:
        messages = [{"role": "system", "content": params.system_message}] + messages

    response = chat(
        model=surname_to_ollama.get(model_name, model_name),
        messages=messages,
        options=Options(num_predict=params.max_new_tokens),
    ).message.content

    return response if response is not None else "NO RESPONSE"


def call_lmstudio_embeddings(
    sentence: str, model_name: str = "text-embedding-mxbai-embed-large-v1"
):
    client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
    return (
        client.embeddings.create(input=[sentence], model=model_name).data[0].embedding
    )


def call_lmstudio(model_name: str, prompt: str, verbose: bool = False, **kwargs) -> str:
    client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

    params = CallParameters(**kwargs)

    messages = [
        {"role": "user", "content": prompt},
    ]

    if params.system_message is not None:
        messages = [{"role": "system", "content": params.system_message}] + messages

    if verbose:
        print(f"""[blue]Calling LM Studio with: 
            {messages}

            Config: {params}[/]
        """)

    response = (
        client.chat.completions.create(
            model=surname_to_lmstudio.get(model_name, model_name),
            messages=messages,  # type: ignore
            temperature=params.temperature,
            max_completion_tokens=params.max_new_tokens,
        )
        .choices[0]
        .message.content
    )

    return response if response is not None else "NO RESPONSE"


def call_judge(
    prompt: str, answer: str, model_name: str = "gemini-1.5-flash"
) -> JudgeScore:
    from google import genai  # type: ignore

    template = get_template("judge.jinja2")

    contents = template.render(user_prompt=prompt, assistant_answer=answer)

    client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))
    response = client.models.generate_content(
        model=model_name,
        contents=contents,
        config={
            "response_mime_type": "application/json",
            "response_schema": JudgeScore,
        },
    )
    if response.text is None:
        raise ValueError(f"Gemini returned None: {response}")

    return JudgeScore.model_validate(json.loads(response.text))
