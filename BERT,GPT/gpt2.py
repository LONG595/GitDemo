from transformers import BertTokenizer, GPT2LMHeadModel
import torch

# 使用 GPT2 中文模型



model_path = "D:/user/local_models/gpt2-chinese-cluecorpussmall"
tokenizer = BertTokenizer.from_pretrained(model_path)
model = GPT2LMHeadModel.from_pretrained(model_path)

prompt = "如果我拥有一台时间机器"
inputs = tokenizer(prompt, return_tensors="pt")

# 生成文本
output = model.generate(
    inputs["input_ids"],
    max_length=100,
    num_return_sequences=1,
    do_sample=True,
    top_k=50,
    top_p=0.95,
    temperature=0.9
)

result = tokenizer.decode(output[0], skip_special_tokens=True)
print("续写结果：")
print(result)