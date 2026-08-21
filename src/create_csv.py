"""
CSV前処理スクリプト
チャット履歴を会話単位に整形し、分析用CSVを生成する。
"""
import re
import json
import csv
from collections import deque
import pandas as pd
import config

# --- 定数 ---
WEIGHTS = {
    'kyokansei': 0.1, 'igakuseikakusei': 0.25, 'anzensei': 0.35,
    'yuugaisei': 0.2, 'aiirai': 0.05, 'shikafanshi': 0.05,
}
NEGATIVE_COLS = ['igakuseikakusei', 'anzensei', 'yuugaisei', 'aiirai', 'shikafanshi']

FIXED_KEYWORD_TEXTS = [
    "もちこが共感", "ほっこりおはなし", "もちことモヤモヤまとめ",
    "励ましもちこ", "一緒に調べる", "ぺん先生にきいてみる", "むちことおしゃべり",
]
TOOL_COMMAND_TEXTS = [
    "うちのこいくつ？", "子育てまんだら", "AIたちの楽しい使い方",
    "研究同意の確認", "ストレッチを始める", "スクワットを始める",
    "筋トレ動画をみる", "ストレスBOMB", "ID",
]
EXCLUDED_KEYWORDS = FIXED_KEYWORD_TEXTS + TOOL_COMMAND_TEXTS
CONSENT_KEYWORD = "その機能を使うには、まず研究利用への同意が必要"
MEANINGLESS_PHRASES = [
    "そうかも", "そうかもね", "かもしれない", "かも", "かもね",
    "なんか", "なんだか", "ちょっとどうしよう", "どうしよう",
    "たぶん", "おそらく", "きっと大丈夫", "大丈夫", "ありがとう", "分かりました",
]
PERSONA_SWITCH_PATTERN = re.compile(
    r'(ペン先生|もちこ|むちこ).*?(お話しする|きいてみる|相談する|に交代する|相談してみる|相談します)',
    re.IGNORECASE,
)


# --- ユーティリティ ---
def write_csv(df, filepath):
    """DataFrameをCSVファイルに書き出す（BOM付きUTF-8）"""
    with open(filepath, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(df.columns.tolist())
        for row in df.itertuples(index=False, name=None):
            writer.writerow(row)


def clean_text_columns(df):
    """userInput/replyTextの改行・タブ文字を空白に置換"""
    for col in ['userInput', 'replyText']:
        df[col] = df[col].str.replace('\\n', ' ').str.replace('\\t', ' ').str.replace('\\r', ' ')
    return df


def calc_satisfaction(df_point):
    """satisfactionスコアを計算してdictで返す"""
    risk = df_point[NEGATIVE_COLS].mul([WEIGHTS[c] for c in NEGATIVE_COLS]).sum(axis=1)
    bonus = df_point['kyokansei'] * WEIGHTS['kyokansei']
    df_point['satisfaction'] = 100 - (risk * 10) + (bonus * 10)
    return dict(zip(df_point['userId'], df_point['satisfaction']))


def is_meaningless_input(text):
    """無意味な入力かどうかを判定"""
    if not isinstance(text, str):
        return True
    s = text.strip()
    if not s:
        return True
    if PERSONA_SWITCH_PATTERN.search(s):
        return True
    if s in EXCLUDED_KEYWORDS or CONSENT_KEYWORD in s or s in MEANINGLESS_PHRASES:
        return True
    return False


def create_record(s, reply_row, count):
    """1会話分のレコードを作成"""
    if reply_row['action'] == 'ReplyInterruptPersona':
        persona = s['suggested']
    elif reply_row['action'] == 'ReplyCurrentPersona':
        persona = s['current']
    else:
        persona = 'Unknown'

    raw = s.get('orchestrator_raw', '')
    if isinstance(raw, str) and raw.strip():
        try:
            raw = json.dumps(json.loads(raw), ensure_ascii=False)
        except (json.JSONDecodeError, ValueError):
            pass

    return {
        'session_id': f'S{count:03d}',
        'userId': s['user_id'],
        'point': point_result_dict.get(s['user_id']),
        'persona': persona,
        'replyType': reply_row['action'],
        'userInput': s['input'],
        'replyText': reply_row['replyText'],
        'orchestratorRaw': raw,
    }


# --- メイン処理 ---
def main():
    df_chat = pd.read_csv(config.DATA_DIR / 'mochiko-line-bot-prod-20260318 (1).csv')
    df_point = pd.read_csv(config.DATA_DIR / 'real_research.csv')
    print(f"[Step 1] 生データ: chat={len(df_chat)}行, point={len(df_point)}行")

    global point_result_dict
    point_result_dict = calc_satisfaction(df_point)

    # --- セッション配對 ---
    orchestrator_queue = deque()
    session_buffer = {}
    processed_rows = []
    session_count = 1

    for i in range(len(df_chat)):
        row = df_chat.iloc[i]

        if row['action'] == 'OrchestratorResult':
            p_id = row['createdAt']
            session_buffer[p_id] = {
                'input': row['userInput'],
                'suggested': row['suggestedParent'],
                'current': row['currentParent'],
                'current_reply': None,
                'interrupt_reply': None,
                'user_id': row['userId'],
                'orchestrator_raw': row['orchestratorRaw'],
            }
            orchestrator_queue.append(p_id)

        elif row['action'] in ('ReplyCurrentPersona', 'ReplyInterruptPersona'):
            if orchestrator_queue:
                target = orchestrator_queue[0]
                if row['action'] == 'ReplyCurrentPersona':
                    session_buffer[target]['current_reply'] = row
                    if session_buffer[target]['suggested'] == session_buffer[target]['current']:
                        orchestrator_queue.popleft()
                else:
                    session_buffer[target]['interrupt_reply'] = row
                    orchestrator_queue.popleft()

        keys_to_delete = []
        for p_id, s in session_buffer.items():
            has_interrupt = (s['suggested'] != s['current'])
            if (s['current_reply'] is not None) and ((not has_interrupt) or (s['interrupt_reply'] is not None)):
                reply = s['interrupt_reply'] if s['interrupt_reply'] is not None else s['current_reply']
                processed_rows.append(create_record(s, reply, session_count))
                session_count += 1
                keys_to_delete.append(p_id)

        for p_id in keys_to_delete:
            del session_buffer[p_id]

    df_final = pd.DataFrame(processed_rows)
    print(f"[Step 2] 会话配对完成: {len(df_final)}行")

    # --- ID照合で分割 ---
    df_cant_find_id = df_final[df_final['point'].isna()].copy()
    df_final = df_final[df_final['point'].notna()].copy()
    print(f"[Step 2.5] 排除无法对照id: 残り{len(df_final)}行, 排除{len(df_cant_find_id)}行")

    # --- クリーニング（両方適用） ---
    df_final = clean_text_columns(df_final)
    df_cant_find_id = clean_text_columns(df_cant_find_id)

    # --- 無意味入力フィルタ + 重複除去 ---
    mask_valid = (df_final['userInput'].str.len() >= 4) & (~df_final['userInput'].apply(is_meaningless_input))
    df_clean = df_final[mask_valid].copy()
    print(f"[Step 3] 过滤后: {len(df_clean)}行 (排除{len(df_final)-len(df_clean)}行)")

    df_clean = df_clean.drop_duplicates(subset=['userId', 'replyType', 'userInput', 'persona'], keep='first')
    print(f"[Step 4] 去重后: {len(df_clean)}行")

    # --- ID照合テーブル ---
    ids_chat = pd.DataFrame(df_chat['userId'].unique(), columns=['chat_userId'])
    ids_point = pd.DataFrame(df_point['userId'].unique(), columns=['point_userId'])
    df_id_comparison = pd.merge(ids_chat, ids_point, left_on='chat_userId', right_on='point_userId', how='outer')

    # --- 出力 ---
    write_csv(df_with_id := df_clean, config.DATA_DIR / 'data_with_id.csv')
    write_csv(df_cant_find_id, config.DATA_DIR / 'data_cant_find_id.csv')
    write_csv(df_id_comparison, config.DATA_DIR / 'data_id_comparison.csv')

    print(f"[Step 5] 出力: with_id={len(df_with_id)}行, cant_find_id={len(df_cant_find_id)}行")
    print(f"[Done] 共导出3个CSV文件")


if __name__ == '__main__':
    main()
