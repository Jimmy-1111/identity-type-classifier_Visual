import streamlit as st
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util

# 模型載入
model = SentenceTransformer("sonoisa/sentence-bert-base-ja-mean-tokens")
model.to(torch.device("cpu"))

# 分類定義文
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
    "伝統的／中立的言語": (
        "コスト最適化により営業利益率が上昇しました。\n"
        "品質管理体制を再構築し不良率を削減しました。"
    ),
    "その他（Other）": ""
}
label_options = list(category_definitions.keys())

# 初始化 session state
if "data" not in st.session_state:
    st.session_state.data = None
if "current_index" not in st.session_state:
    st.session_state.current_index = 0
if "annotations" not in st.session_state:
    st.session_state.annotations = []

# 上傳檔案
st.title("📊 日本語：企業年報文のアイデンティティ分類（Excel標註モード）")
uploaded_file = st.file_uploader("Excel ファイルをアップロードしてください", type=["xlsx"])

# 檔案載入與欄位選擇
if uploaded_file:
    df = pd.read_excel(uploaded_file)
    st.session_state.data = df
    st.markdown("✅ 検出された列名：")
    col_name = st.selectbox("▶️ 分類対象の列を選択", df.columns.tolist())

    if col_name:
        current = st.session_state.current_index
        if current >= len(df):
            st.success("すべての文を分類しました。")
            result_df = pd.DataFrame(st.session_state.annotations)
            st.dataframe(result_df)
            csv = result_df.to_csv(index=False).encode("utf-8")
            st.download_button("📥 分類結果をCSVでダウンロード", csv, "classified_results.csv", "text/csv")
        else:
            row = df.iloc[current]
            sentence = str(row[col_name])

            # 顯示其他欄位
            st.markdown("### 🧾 参考情報")
            for k, v in row.items():
                if k != col_name:
                    st.write(f"**{k}**: {v}")

            st.markdown("### ✏️ 分類対象の文")
            st.info(sentence)

            # 類別預測
            sentence_emb = model.encode(sentence, convert_to_tensor=True)
            definition_embs = {
                label: model.encode([t.strip() for t in text.splitlines() if t.strip()], convert_to_tensor=True).mean(dim=0)
                for label, text in category_definitions.items() if text.strip()
            }

            if sentence.strip() == "" or any(kw in sentence for kw in ["【", "】", "景気", "為替", "GDP"]):
                predicted_label = "その他（Other）"
                explanation = "外部環境や構造語句が含まれているため、自動的に『その他』と分類されました。"
            else:
                scores = {label: float(util.cos_sim(sentence_emb, emb)) for label, emb in definition_embs.items()}
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

            # 類別標註
            selected_label = st.selectbox("📌 分類ラベルを選択してください", label_options, index=label_options.index(predicted_label))

            if st.button("✅ この文を保存して次へ"):
                annotated = {
                    "index": current,
                    "文": sentence,
                    "モデル分類": predicted_label,
                    "修正後ラベル": selected_label,
                }
                for k, v in row.items():
                    annotated[k] = v
                st.session_state.annotations.append(annotated)
                st.session_state.current_index += 1
                st.experimental_rerun()
