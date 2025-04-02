# 测试ollama版本的 nomic-embed-text

import ollama



ollama.embed(
    model="nomic-embed-text:latest",
    input="一些语言"
)