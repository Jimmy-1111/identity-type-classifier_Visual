import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch

# === 模型載入 ===
model = SentenceTransformer("sonoisa/sentence-bert-base-ja-mean-tokens")
model.to(torch.device("cpu"))

# === 分類定義文 ===
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
label_options = list(default_definitions.keys()) + ["その他（Other）"]

EXTERNAL_ONLY_KEYWORDS = [
    "経済", "景気", "インフレ", "金利", "為替", "物価", "政策", "地政学",
    "個人消費", "中央銀行", "消費者心理", "投資環境", "輸出", "輸入", "GDP", "景況感"
]

def is_force_other(sent):
    return "【" in sent or "】" in sent or any(kw in sent for kw in EXTERNAL_ONLY_KEYWORDS)

# === Streamlit 頁面設定 ===
st.set_page_config(page_title="アイデンティティ分類", layout="centered")
st.title("📊 日本語：企業年報文のアイデンティティ分類")

# === 初始化狀態 ===
if "results" not in st.session_state:
    st.session_state.results = None

st.header("🖊️ 分析対象の文を入力（1 行 1 文）")
sentences_text = st.text_area("ここに文を入力してください", height=220)

# === 分析按鈕 ===
if st.button("🚀 分析する"):
    sentences = [s.strip() for s in sentences_text.splitlines() if s.strip()]
    if not sentences:
        st.warning("文が入力されていません。")
        st.stop()

    sentence_embeddings = model.encode(sentences, convert_to_tensor=True)
    definition_embeddings = {
        label: model.encode(
            [t.strip() for t in text.splitlines() if t.strip()],
            convert_to_tensor=True
        ).mean(dim=0)
        for label, text in default_definitions.items()
    }

    data = []
    for i, (sent, emb) in enumerate(zip(sentences, sentence_embeddings)):
        if is_force_other(sent):
            pred_label = "その他（Other）"
            score = 0.0
        else:
            scores = {k: float(util.cos_sim(emb, v)) for k, v in definition_embeddings.items()}
            sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            pred_label, score = sorted_scores[0]

        data.append({
            "入力文": sent,
            "分類ラベル": pred_label,
            "similarity score": score,
            "修正後ラベル": pred_label
        })

    st.session_state.results = data  # 儲存分析結果

# === 顯示結果與分類修正 ===
if st.session_state.results:
    st.subheader("✏️ 分類の修正（必要に応じて）")
    for i, row in enumerate(st.session_state.results):
        st.markdown(f"**文 {i+1}：** {row['入力文']}")
        new_label = st.selectbox(
            "分類ラベルを修正する（またはそのまま）",
            label_options,
            index=label_options.index(row["修正後ラベル"]),
            key=f"label_select_{i}"
        )
        st.session_state.results[i]["修正後ラベル"] = new_label

    # 匯出表格
    result_df = pd.DataFrame(st.session_state.results)
    st.subheader("📥 修正後の結果一覧")
    st.dataframe(result_df[["入力文", "分類ラベル", "修正後ラベル", "similarity score"]], use_container_width=True)

    csv = result_df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 修正結果をCSVでダウンロード", csv, "classified_results.csv", "text/csv")
