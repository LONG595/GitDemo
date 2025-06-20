import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline



model_name = "D:/user/roberta-base-finetuned-jd-binary-chinese"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)

def classify_sentiment(text):
    result = classifier(text)
    label = "正面" if result[0]['label'] == "LABEL_1" else "负面"
    score = result[0]['score']
    return f"分类结果: {label} (置信度: {score:.2f})"





classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)

def classify_sentiment(text):
    result = classifier(text)
    label = "正面" if result[0]['label'] == "LABEL_1" else "负面"
    score = result[0]['score']
    return f"分类结果: {label} (置信度: {score:.2f})"

# 影评数据
movie_reviews = [
    "这部电影太精彩了，节奏紧凑毫无冷场，完全沉浸其中！",
    "剧情设定新颖不落俗套，每个转折都让人惊喜。",
    "导演功力深厚，镜头语言非常有张力，每一帧都值得回味。",
    "美术、服装、布景细节丰富，完全是视觉盛宴！",
    "是近年来最值得一看的国产佳作，强烈推荐！",
    "剧情拖沓冗长，中途几次差点睡着。",
    "演员表演浮夸，完全无法让人产生代入感。",
    "剧情老套，充满套路和硬凹的感动。",
    "对白尴尬，像是AI自动生成的剧本。",
    "看完只觉得浪费了两个小时，再也不想看第二遍。"
]

# 外卖评价数据
food_reviews = [
    "食物完全凉了，吃起来像隔夜饭，体验极差。",
    "汤汁洒得到处都是，包装太随便了。",
    "味道非常一般，跟评论区说的完全不一样。",
    "分量太少了，照片看着满满的，实际就几口。",
    "食材不新鲜，有异味，感觉不太卫生。",
    "食物份量十足，性价比超高，吃得很满足！",
    "味道超级赞，和店里堂食一样好吃，五星好评！",
    "这家店口味稳定，已经回购好几次了，值得信赖！",
    "点单备注有按要求做，服务意识很棒。",
    "包装环保、整洁美观，整体体验非常好。"
]


last_digit = 0 # 倒数第一位
second_last_digit = 4 # 倒数第二位

# 获取对应句子
movie_review = movie_reviews[last_digit]
food_review = food_reviews[second_last_digit]

# 进行分类并输出结果
print("影评分类:")
print(f"句子: {movie_review}")
print(classify_sentiment(movie_review))


print("\n外卖评价分类:")
print(f"句子: {food_review}")
print(classify_sentiment(food_review))