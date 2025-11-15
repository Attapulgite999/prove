from huggingface_hub import HfApi
api = HfApi()
models = api.list_models(search="qwen 3")
for model in models:
    print(model.modelId)