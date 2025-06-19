import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util

# === 模型載入：指定使用 CPU 避免部署問題 ===
model = SentenceTransformer("sonoisa/sentence-bert-base-ja-mean-tokens", device="cpu")

# === 類型定義：每類三句（可自行增減）===
default_definitions = {
    "アイデンティティ挑戦型イノベーション": (
        "私たちはモビリティサービス企業へと転換します。\n"
        "新しい社会課題に挑むイノベーション企業へ生まれ変わります。\n"
        "当社の使命を再定義し、デジタル時代の価値創造企業となります。"
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
    ),
    "その他（Other）": (
        "第２【事業の状況】\n"
        "図１：収益の推移\n"
        "注記：本資料は監査法人の確認を受けています。\n"
        "目次\n"
        "以上"
    )
}

# === 頁面設定 ===
st.set_page_config(page_title="日本語句子分類", layout="centered")
st.title("📊 日本語：企業年報文のアイデンティティ分類")

# === 類型定義輸入（可多行）===
st.header("📝 分類基準の定義文（カテゴリごとに複数行で入力）")
category_inputs = {}
for cat, default in default_definitions.items():
    category_inputs[cat] = st.text_area(
        cat,
        value=default,
        height=90,
        help="改行ごとに別の定義文として扱われます"
    )

# === 句子輸入 ===
st.header("✏️ 分析対象の文を入力（1 行 1 文）")
sentences_text = st.text_area("ここに文を入力してください", height=220)

# === 分析按鈕 ===
if st.button("🚀 分析する"):

    sentences = [s.strip() for s in sentences_text.splitlines() if s.strip()]
    if not sentences:
        st.warning("文が入力されていません。")
        st.stop()

    sentence_embeddings = model.encode(sentences, convert_to_tensor=True)

    # === 類型定義句向量平均 ===
    definition_embeddings = {}
    for label, text_block in category_inputs.items():
        defs = [t.strip() for t in text_block.splitlines() if t.strip()]
        if not defs:
            defs = default_definitions[label].splitlines()
        emb = model.encode(defs, convert_to_tensor=True).mean(dim=0)
        definition_embeddings[label] = emb

    # === 分類主邏輯 ===
    predicted_labels, similarity_scores = [], []
    threshold = 0.75  # 可調整

    def is_force_other(sent):
        return "【" in sent or "】" in sent

    for sent, sent_emb in zip(sentences, sentence_embeddings):
        # 若含有【或】→ 強制歸為「其他」
        if is_force_other(sent):
            predicted_labels.append("その他（Other）")
            similarity_scores.append(0.0)
            continue

        # 否則照常語意分類
        scores = {
            label: float(util.cos_sim(sent_emb, def_emb))
            for label, def_emb in definition_embeddings.items()
        }
        best_label = max(scores, key=scores.get)
        best_score = scores[best_label]

        if best_score < threshold:
            predicted_labels.append("その他（Other）")
        else:
            predicted_labels.append(best_label)
        similarity_scores.append(best_score)

    # === 顯示結果 ===
    result_df = pd.DataFrame({
        "入力文": sentences,
        "分類ラベル": predicted_labels,
        "similarity score": similarity_scores
    })
    st.subheader("🔎 分析結果")
    st.dataframe(result_df, use_container_width=True)
