import json
import ipdb

with open("../dataset/REVIEWS_train.json", "r") as f:
    reviews = json.load(f)

ipdb.set_trace()

result = []
for review in reviews:
    if "2025" in review["conference_year_track"]:
        result.append(review)

print(len(result))