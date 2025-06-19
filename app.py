import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util

# === 日文 Sentence-BERT ===
model = SentenceTransformer("sonoisa/sentence-bert-base-ja-mean-tokens")

# === 各タイプの定義文（3 行ずつ）===
default_definitions = {
    # アイデンティティ挑戦型
    "アイデンティティ挑戦型イノベーション": (
        "私たちはモビリティサービス企業へと転換します。\n"
        "新しい社会課題に挑むイノベーション企業へ生まれ変わります。\n"
        "当社の使命を再定義し、デジタル時代の価値創造企業となります。"
    ),

    # アイデンティティ拡張型
    "アイデンティティ拡張型イノベーション": (
        "伝統の技術力を新領域エネルギーソリューションへ応用します。\n"
        "コアバリューを保ちながらヘルスケア市場へ事業を広げます。\n"
        "ものづくり精神をIoTサービスに融合させ、新たな価値を提供します。"
    ),

    # アイデンティティ強化型
    "アイデンティティ強化型イノベーション": (
        "高品質へのこだわりをさらに磨き既存製品をアップグレードします。\n"
        "環境志向ブランドを強化するため資源循環型工程を導入します。\n"
        "長年培った安全性を核に次世代モデルを開発します。"
    ),

    # 伝統的／中立的言語
    "伝統的／中立的言語": (
        "生産効率を7%向上させる改善活動を実施しました。\n"
        "コスト最適化により営業利益率が上昇しました。\n"
        "品質管理体制を再構築し不良率を削減しました。"
    ),

    # その他
    "その他（Other）": (
        "今年度の寄付金総額は1億円となりました。\n"
        "新しい福利厚生制度を導入しました。\n"
        "事務所のレイアウトを変更しました。"
    )
}

# === Streamlit レイアウト設定 ===
st.set_page_config(page_title="日本語句子分類", layout="centered")
st.title("📊 日本語：企業年報文のアイデンティティ分類")

# === 1. 定義文編集エリア ===
st.header("📝 分類基準の定義文（複数行入力可・改行区切り）")
category_inputs = {}
for cat, default in default_definitions.items():
    category_inputs[cat] = st.text_area(
        cat,
        value=default,
        height=90,
        help="改行ごとに別の定義文として扱われます（例：3 行）"
    )

# === 2. 解析対象文入力 ===
st.header("✏️ 分析する文を入力（1 行 1 文）")
sentences_text = st.text_area("ここに文を貼り付けてください", height=220)

# === 3. 解析ボタン ===
if st.button("🚀 分析する"):
    # --- 入力チェック ---
    sentences = [s.strip() for s in sentences_text.splitlines() if s.strip()]
    if not sentences:
        st.warning("分析する文がありません。")
        st.stop()

    # --- 入力文のベクトル化 ---
    sentence_embeddings = model.encode(sentences, convert_to_tensor=True)

    # --- 定義文ベクトル（各行→平均） ---
    definition_embeddings = {}
    for label, text_block in category_inputs.items():
        defs = [t.strip() for t in text_block.splitlines() if t.strip()]
        if not defs:  # 空の場合はデフォルトに戻す
            defs = default_definitions[label].splitlines()
        emb = model.encode(defs, convert_to_tensor=True).mean(dim=0)
        definition_embeddings[label] = emb

    # --- 分類 ---
    predicted_labels, similarity_scores = [], []
    for sent_emb in sentence_embeddings:
        scores = {
            label: float(util.cos_sim(sent_emb, def_emb))
            for label, def_emb in definition_embeddings.items()
        }
        best_label = max(scores, key=scores.get)
        predicted_labels.append(best_label)
        similarity_scores.append(scores[best_label])

    # --- 結果表示 ---
    result_df = pd.DataFrame({
        "入力文": sentences,
        "分類ラベル": predicted_labels,
        "similarity score": similarity_scores
    })
    st.subheader("🔎 分析結果")
    st.dataframe(result_df, use_container_width=True)

