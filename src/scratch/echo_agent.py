class ModelClient:
    def complete(self, prompt: str, max_new_tokens: int = 512) -> str:
        raise NotImplementedError

class EchoModel(ModelClient):
    def complete(self, prompt: str, max_new_tokens: int = 512) -> str:
        return f"<final>{prompt}</final>"

import re


class MiniRuntime:
    def __init__(self, model_client: ModelClient):
        self.model_client = model_client

    def ask(self, user_message: str) -> str:
        raw = self.model_client.complete(user_message)
        return self.parse_final(raw)

    def parse_final(self, raw: str) -> str:
        match = re.search(r"<final>(.*?)</final>", raw, re.DOTALL)
        if not match:
            raise ValueError("model did not return <final>")
        return match.group(1).strip()

if __name__ == "__main__":
    agent = MiniRuntime(EchoModel())
    print(agent.ask("hello pico"))




