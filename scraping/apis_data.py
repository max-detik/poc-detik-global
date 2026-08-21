import json

import requests

ids = [
    8606706,
    8610448,
    8606363,
    8611160,
    8609660,
    8604738,
    8600003,
    8605519,
    8606892,
    8606869,
    8606982,
    8606407,
    8595990,
]

FIELDS = ["id", "title", "content", "resume", "tag", "tags", "keywordauto", "categoryauto"]


def extract_article(data):
    article = {field: data.get(field) for field in FIELDS if field in data}

    image_cover = data.get("image_cover") or []
    if image_cover:
        first_image = image_cover[0]
        article["image_cover"] = {
            "text": first_image.get("text"),
            "alt_image": first_image.get("alt_image"),
        }

    return article


def fetch_article(article_id):
    url = f"https://apis.detik.com/v1/detail/?id={article_id}"
    response = requests.get(url)
    response.raise_for_status()  # Raise an HTTPError for bad responses (4xx or 5xx)
    payload = response.json()
    return extract_article(payload["data"])


def main():
    articles = [fetch_article(article_id) for article_id in ids]

    with open("input/apis-data-all.json", "w", encoding="utf-8") as f:
        json.dump(articles, f, ensure_ascii=False, indent=2)

    print("DONE!")


if __name__ == "__main__":
    main()
