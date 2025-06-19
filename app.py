import streamlit as st
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util

# 模型載入
model = SentenceTransformer("sonoisa/sentence-bert-base-ja-mean-tokens")

# 分類定義文（已簡化為4類）
category_definitions = {
    "アイデンティティ挑戦型イノベーション": (
        "私たちはモビリティサービス企業へと転換します。\n"
        "新しい社会課題に挑むイノベーション企業へ生まれ変わります。\n"
        "当社の使命を再定義し、デジタル時代の価値創造企業となります。"
    ),
    "アイデンティティ拡張型イノベーション": (
        "伝統の技術力を新領域へ応用します。\n"
        "コアバリューを保ちながらヘルスケア市場へ事業を広げます。"
    ),
    "アイデンティティ強化型イノベーション": (
        "高品質へのこだわりをさらに磨き既存製品をアップグレードします。\n"
        "安全性を核に次世代モデルを開発します。"
    ),
    "その他（Other）": ""
}
label_options = list(category_definitions.keys())

# 初始化
if "data" not in st.session_state:
    st.session_state.data = None
if "current_index" not in st.session_state:
    st.session_state.current_index = 0
if "annotations" not in st.session_state:
    st.session_state.annotations = []

st.title("📊 日本語：企業年報文のアイデンティティ分類（標註協作版）")

# 新增：標註人員輸入欄
annotator = st.text_input("✍️ 標註人員名稱（將記錄在每筆資料中）", value="匿名")

# 上傳 Excel
uploaded_file = st.file_uploader("Excel ファイルをアップロードしてください", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    st.session_state.data = df
    col_name = st.selectbox("▶️ 分類対象の列を選択", df.columns.tolist())

    if col_name:
        current = st.session_state.current_index
        total = len(df)

        # 顯示進度條
        st.markdown(f"⏳ 標註進度：{current + 1} / {total}")
        st.progress((current + 1) / total)

        if current >= total:
            st.success("✅ すべての文を分類しました！")
            result_df = pd.DataFrame(st.session_state.annotations)
            st.dataframe(result_df)
            csv = result_df.to_csv(index=False).encode("utf-8")
            st.download_button("📥 分類結果をCSVでダウンロード", csv, "classified_results.csv", "text/csv")
        else:
            row = df.iloc[current]
            sentence = str(row[col_name])

            st.markdown("### 🧾 参考情報")
            for k, v in row.items():
                if k != col_name:
                    st.write(f"**{k}**: {v}")

            st.markdown("### ✏️ 分類対象の文")
            st.info(sentence)

            # 模型分類預測
            sentence_emb = model.encode(sentence, convert_to_tensor=True)
            definition_embs = {
                label: model.encode(
                    [t.strip() for t in text.splitlines() if t.strip()],
                    convert_to_tensor=True
                ).mean(dim=0)
                for label, text in category_definitions.items() if text.strip()
            }

            if sentence.strip() == "" or any(kw in sentence for kw in ["【", "】", "景気", "為替", "GDP"]):
                predicted_label = "その他（Other）"
                best_score = 0.0
                explanation = "外部環境や構造語句が含まれているため、自動的に『その他』と分類されました。"
            else:
                scores = {k: float(util.cos_sim(sentence_emb, v)) for k, v in definition_embs.items()}
                sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
                predicted_label, best_score = sorted_scores[0]
                second_label, second_score = sorted_scores[1]
                example = "\n".join(f"・{s}" for s in category_definitions[predicted_label].splitlines()[:2])
                explanation = (
                    f"この文は『{predicted_label}』に最も高い類似度（{best_score:.2f}）を示しました。\n"
                    f"次に近いのは『{second_label}』（{second_score:.2f}）でした。\n\n"
                    f"《参考：『{predicted_label}』の定義文例》\n{example}"
                )

            with st.expander("🧠 分類理由を見る", expanded=True):
                st.markdown(explanation)

            selected_label = st.selectbox("📌 分類ラベルを選択してください", label_options, index=label_options.index(predicted_label))

            if st.button("✅ この文を保存して次へ"):
                annotated = {
                    "index": current,
                    "文": sentence,
                    "モデル分類": predicted_label,
                    "相似度スコア": best_score,
                    "修正後ラベル": selected_label,
                    "標註人員": annotator,
                }
                for k, v in row.items():
                    annotated[k] = v
                st.session_state.annotations.append(annotated)
                st.session_state.current_index += 1
                st.rerun()
