import json
from pathlib import Path

path = Path("data/qa/generated_qa.jsonl")

new_items = [
    {
        "qid": "gen004",
        "question": "特別高圧（154kV以上）の連系において、一般送配電事業者が発電等設備設置者に提示すべき標準化された情報の種類は何ですか？",
        "gold": [{"doc_id": "doc008", "page": 12, "chunk_id": "doc008:p012:c001"}]
    },
    {
        "qid": "gen005",
        "question": "電力データ集約システムで特定契約者登録を行う際、LG-WAN（画面）とAPI連携のどちらの方法でも実施可能ですか？",
        "gold": [{"doc_id": "doc002", "page": 33, "chunk_id": "doc002:p033:c001"}]
    },
    {
        "qid": "gen006",
        "question": "配電事業ガイドライン（目次）において、「兼業規制・行為規制」は第何章に記載されていますか、またそのページ数は？",
        "gold": [{"doc_id": "doc004", "page": 3, "chunk_id": "doc004:p003:c001"}]
    },
    {
        "qid": "gen007",
        "question": "電力データ集約システムの利用申請に関する問い合わせ先として、北海道電力ネットワークの担当部署はどこですか？",
        "gold": [{"doc_id": "doc002", "page": 64, "chunk_id": "doc002:p064:c001"}]
    },
    {
        "qid": "gen008",
        "question": "配電事業者が設備を借り受ける際の価格算定において、「託送料金期待収入」から控除される主な費用項目は何ですか？",
        "gold": [{"doc_id": "doc004", "page": 57, "chunk_id": "doc004:p057:c001"}]
    }
]

with path.open("a", encoding="utf-8") as f:
    for item in new_items:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"Appended {len(new_items)} items to {path}")
