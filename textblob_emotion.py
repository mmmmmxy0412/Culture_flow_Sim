import json
from textblob import TextBlob
import csv

def compute_sentiment_from_txt(path):
    polarities = []
    rows = []  # 用于导出明细

    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                print(f"[跳过] 第 {i} 行不是合法 JSON")
                continue

            text = obj.get("rawContent", "")
            if not isinstance(text, str) or not text.strip():
                print(f"[跳过] 第 {i} 行 rawContent 为空或不是字符串")
                continue

            polarity = TextBlob(text).sentiment.polarity
            polarities.append(polarity)

            rows.append({
                "line": i,
                "username": obj.get("user", {}).get("username", ""),
                "date": obj.get("date", ""),
                "polarity": polarity,
                "rawContent": text.replace("\n", " ")
            })

    if not polarities:
        return {
            "count": 0,
            "total_sentiment": 0.0,
            "avg_sentiment": 0.0
        }

    total_sentiment = sum(polarities)
    avg_sentiment = total_sentiment / len(polarities)

    return {
        "count": len(polarities),
        "total_sentiment": total_sentiment,
        "avg_sentiment": avg_sentiment,
        "details": rows
    }

if __name__ == "__main__":
    input_path = ""  # 改成你的文件名/路径
    result = compute_sentiment_from_txt(input_path)

    print("有效推文数量:", result["count"])
    print("总情感值 (sum polarity):", result["total_sentiment"])
    print("平均情感值 (avg polarity):", result["avg_sentiment"])

