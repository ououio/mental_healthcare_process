"""
トピック別のオーケストレータ選択分析
- トピックごとの主導感情
- トピックごとの各persona選択確率・比率
"""
import ast
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from config import DATA_DIR

PERSONAS = ["mochiko", "pen_sensei", "muchiko"]
PERSONA_LABELS = {
    "mochiko": "Empathic",
    "pen_sensei": "Dr",
    "muchiko": "Exercise",
}

EMOTION_COLS = [
    "joy", "sadness", "anticipation", "surprise",
    "anger", "fear", "disgust", "trust",
]
EMOTION_JP = {
    "joy": "喜び",
    "sadness": "悲しみ",
    "anticipation": "期待",
    "surprise": "驚き",
    "anger": "怒り",
    "fear": "恐怖",
    "disgust": "嫌悪",
    "trust": "信頼",
}


def parse_orchestrator(raw: str) -> dict:
    import json
    s = str(raw).strip()
    # CSVによる二重クォート除去: "...""key"": val..." → { "key": val }
    if s.startswith('"') and s.endswith('"'):
        s = s[1:-1]
    s = s.replace('""', '"').replace('\\n', '\n')
    for parser in [json.loads, ast.literal_eval]:
        try:
            result = parser(s)
            if isinstance(result, dict):
                return result
        except Exception:
            continue
    return None


def main():
    # --- データ読み込み ---
    df = pd.read_csv(DATA_DIR / "data_with_id.csv")
    topic_df = pd.read_csv(DATA_DIR / "topic_modeling" / "combined_userInput_doc_topics.csv")
    sent_df = pd.read_csv(DATA_DIR / "sentiment" / "sentiment.csv")

    # --- データ結合 ---
    # topic_df は document_index が data_with_id の行indexに対応
    df["topic_id"] = topic_df["topic_id"].values

    input_emo_cols = [f"input_{c}" for c in EMOTION_COLS]
    reply_emo_cols = [f"reply_{c}" for c in EMOTION_COLS]
    for col in input_emo_cols + reply_emo_cols:
        df[col] = sent_df[col].values

    diff_df = pd.read_csv(DATA_DIR / "sentiment" / "sentiment_all_diff.csv")
    diff_emo_cols = [f"diff_{c}" for c in EMOTION_COLS]
    for col in diff_emo_cols:
        df[col] = diff_df[col].values

    # --- オーケストレータJSON解析 ---
    orch_probs = []
    for raw in df["orchestratorRaw"]:
        parsed = parse_orchestrator(str(raw))
        if parsed and all(p in parsed for p in PERSONAS):
            orch_probs.append({p: float(parsed[p]) for p in PERSONAS})
        else:
            orch_probs.append(None)

    df["orch_prob"] = orch_probs
    df = df[df["orch_prob"].notna()].copy()

    for p in PERSONAS:
        df[f"prob_{p}"] = df["orch_prob"].apply(lambda x: x[p])
        df[f"selected_{p}"] = df["orch_prob"].apply(lambda x: 1 if max(x, key=x.get) == p else 0)

    # --- 主導感情の特定 ---
    def get_dominant(row, prefix):
        scores = {c: row[f"{prefix}_{c}"] for c in EMOTION_COLS}
        best = max(scores, key=scores.get)
        return EMOTION_JP[best]

    def get_max_abs_diff_emotion(row):
        scores = {c: abs(row[f"diff_{c}"]) for c in EMOTION_COLS}
        best = max(scores, key=scores.get)
        return EMOTION_JP[best]

    df["input_emotion"] = df.apply(lambda r: get_dominant(r, "input"), axis=1)
    df["reply_emotion"] = df.apply(lambda r: get_dominant(r, "reply"), axis=1)
    df["emotion_diff"] = df.apply(get_max_abs_diff_emotion, axis=1)

    # --- トピック別の集計 ---
    # 各personaの全体選択回数
    total_selected = {p: int(df[f"selected_{p}"].sum()) for p in PERSONAS}

    # --- 合計行 ---
    total_n = len(df)
    total_avg_probs = {p: round(df[f"prob_{p}"].mean(), 3) for p in PERSONAS}
    total_sel_ratios = {p: round(df[f"selected_{p}"].mean(), 3) for p in PERSONAS}
    total_input_dom = df["input_emotion"].value_counts().index[0]
    total_reply_dom = df["reply_emotion"].value_counts().index[0]
    total_diff_dom = df["emotion_diff"].value_counts().index[0]

    results = [{
        "topic_id": "合計",
        "count": total_n,
        "input_emotion": total_input_dom,
        "reply_emotion": total_reply_dom,
        "emotion_diff": total_diff_dom,
        **{f"prob_{PERSONA_LABELS[p]}": total_avg_probs[p] for p in PERSONAS},
        **{f"ratio_{PERSONA_LABELS[p]}": total_sel_ratios[p] for p in PERSONAS},
        **{f"count_{PERSONA_LABELS[p]}": total_selected[p] for p in PERSONAS},
        **{f"share_{PERSONA_LABELS[p]}": 1.0 for p in PERSONAS},
    }]

    for tid, grp in df.groupby("topic_id"):
        n = len(grp)
        avg_probs = {p: round(grp[f"prob_{p}"].mean(), 3) for p in PERSONAS}
        sel_ratios = {p: round(grp[f"selected_{p}"].mean(), 3) for p in PERSONAS}
        sel_counts = {p: int(grp[f"selected_{p}"].sum()) for p in PERSONAS}
        # 该persona在此topic的输出次数占该persona总输出次数的比例
        sel_shares = {
            p: round(sel_counts[p] / total_selected[p], 3) if total_selected[p] > 0 else 0
            for p in PERSONAS
        }
        input_dom = grp["input_emotion"].value_counts().index[0]
        reply_dom = grp["reply_emotion"].value_counts().index[0]
        diff_dom = grp["emotion_diff"].value_counts().index[0]

        results.append({
            "topic_id": int(tid),
            "count": n,
            "input_emotion": input_dom,
            "reply_emotion": reply_dom,
            "emotion_diff": diff_dom,
            **{f"prob_{PERSONA_LABELS[p]}": avg_probs[p] for p in PERSONAS},
            **{f"ratio_{PERSONA_LABELS[p]}": sel_ratios[p] for p in PERSONAS},
            **{f"count_{PERSONA_LABELS[p]}": sel_counts[p] for p in PERSONAS},
            **{f"share_{PERSONA_LABELS[p]}": sel_shares[p] for p in PERSONAS},
        })

    result_df = pd.DataFrame(results[1:]).sort_values("topic_id")
    result_df = pd.concat([pd.DataFrame([results[0]]), result_df], ignore_index=True)

    # --- 出力 ---
    header = (
        "Topic\t件数\t入力感情\t出力感情\t感情差分\t"
        + "\t".join(f"prob({PERSONA_LABELS[p]})" for p in PERSONAS)
        + "\t"
        + "\t".join(f"ratio({PERSONA_LABELS[p]})" for p in PERSONAS)
        + "\t"
        + "\t".join(f"count({PERSONA_LABELS[p]})" for p in PERSONAS)
        + "\t"
        + "\t".join(f"share({PERSONA_LABELS[p]})" for p in PERSONAS)
    )
    print(header)
    for _, row in result_df.iterrows():
        line = (
            f"{row['topic_id']}\t{row['count']}\t{row['input_emotion']}\t{row['reply_emotion']}\t{row['emotion_diff']}\t"
            + "\t".join(str(row[f"prob_{PERSONA_LABELS[p]}"]) for p in PERSONAS)
            + "\t"
            + "\t".join(str(row[f"ratio_{PERSONA_LABELS[p]}"]) for p in PERSONAS)
            + "\t"
            + "\t".join(str(row[f"count_{PERSONA_LABELS[p]}"]) for p in PERSONAS)
            + "\t"
            + "\t".join(str(row[f"share_{PERSONA_LABELS[p]}"]) for p in PERSONAS)
        )
        print(line)

    # CSV保存
    out_path = DATA_DIR / "orchestrator_topic_summary.csv"
    result_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n保存先: {out_path}")


if __name__ == "__main__":
    main()
