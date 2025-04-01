import ollama

resp = ollama.embed(
    model="nomic-embed-text:latest",
    input="This is a test input sentence.",
)

print(resp)
