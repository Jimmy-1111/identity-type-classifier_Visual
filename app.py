import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch

# === 模型載入（安全版本）===
model = SentenceTransformer("sonoisa/sentence-bert-base-ja-mean-tokens")
model.to(torch.device("cpu"))

# === 分類定義（預設）===
default_definitions = {
    "アイデンティティ挑戦型イノベーション": (
        "私たちはモビリティサービス企業へと転換します。\n"
        "新しい社会課題に挑むイノベーション企業へ生まれ変わります。\n"
        "当社の使命を再定義し、デジタル時代の価値創造企業となります。\n"
        "新しいグローバルプレミアムブランドの確立を目指して、新ブランドを展開します。"
    ),
    "アイデンティティ拡張型イノベーション": (
        "伝統の技術力を新領域エネルギーソリューションへ応用します。\n"
        "コアバリューを保ちながらヘルスケア市場へ事業を広げます。\n"
        "ものづくり精神をIoTサービスに融合させ、新たな価値を提供します。"
    ),
    "アイデンティティ強化型イノベーション": (
        "高品質へのこだわりをさらに磨き既存製品をアップグレードします。\n"
        "環境志向ブランドを強化するため資源循環型工程を導入します。\n"
        "長年培った安全性を核に次世代モデルを開発します。"
    ),
    "伝統的／中立的言語": (
        "生産効率を7%向上させる改善活動を実施しました。\n"
        "コスト最適化により営業利益率が上昇しました。\n"
        "品質管理体制を再構築し不良率を削減しました。"
    )
}

# === Streamlit 設定 ===
st.set_page_config(page_title="日本語句子分類", layout="centered")
st.title("\U0001F4CA 日本語：企業年報文のアイデンティティ分類")

category_inputs = {}
for cat, default in default_definitions.items():
    category_inputs[cat] = default

st.header("\U0001F58B️ 分析対象の文を入力（1 行 1 文）")
sentences_text = st.text_area("ここに文を入力してください", height=220)

EXTERNAL_ONLY_KEYWORDS = [
    "経済", "景気", "インフレ", "金利", "為替", "物価", "政策", "地政学",
    "個人消費", "中央銀行", "消費者心理", "投資環境", "輸出", "輸入", "GDP", "景況感"
]

def is_force_other(sent):
    return "【" in sent or "】" in sent or any(kw in sent for kw in EXTERNAL_ONLY_KEYWORDS)

if st.button("\U0001F680 分析する"):
    sentences = [s.strip() for s in sentences_text.splitlines() if s.strip()]
    if not sentences:
        st.warning("文が入力されていません。")
        st.stop()

    sentence_embeddings = model.encode(sentences, convert_to_tensor=True)

    definition_embeddings = {}
    for label, definition_text in category_inputs.items():
        defs = [t.strip() for t in definition_text.splitlines() if t.strip()]
        if not defs:
            continue
        emb = model.encode(defs, convert_to_tensor=True).mean(dim=0)
        definition_embeddings[label] = emb

    predicted_labels = []
    similarity_scores = []
    explanations = []

    for sent, sent_emb in zip(sentences, sentence_embeddings):
        if is_force_other(sent):
            predicted_labels.append("その他（Other）")
            similarity_scores.append(0.0)
            explanations.append("この文は『【】』または経済・外部環境に関する語句が含まれているため、自動的に『その他』に分類されました。")
            continue

        scores = {
            label: float(util.cos_sim(sent_emb, def_emb))
            for label, def_emb in definition_embeddings.items()
        }
        best_label = max(scores, key=scores.get)
        best_score = scores[best_label]

        predicted_labels.append(best_label)
        similarity_scores.append(best_score)

        top_sentences = category_inputs[best_label].split("\n")[:2]
        explanation = f"この文は『{best_label}』に最も近い定義文と高い類似度（{best_score:.2f}）を示しました。\n例：{top_sentences[0]}"
        explanations.append(explanation)

    result_df = pd.DataFrame({
        "入力文": sentences,
        "分類ラベル": predicted_labels,
        "similarity score": similarity_scores
    })

    st.subheader("\U0001F50D 分析結果")
    st.dataframe(result_df, use_container_width=True)

    st.subheader("\U0001F4AC 分類の説明")
    for i, explanation in enumerate(explanations):
        st.info(f"\n【文 {i+1} の分類理由】\n{explanation}")
