"""
CSV前処理スクリプト
チャット履歴を会話単位に整形し、分析用CSVを生成する。
"""
import re
import json
from collections import deque
import pandas as pd
import config

# データ読み込み
df_chat = pd.read_csv(config.DATA_DIR / 'mochiko-line-bot-prod-20260318 (1).csv')
df_point = pd.read_csv(config.DATA_DIR / 'real_research.csv')
print(f"[Step 1] 生データ: chat={len(df_chat)}行, point={len(df_point)}行")

orchestrator_queue = deque()
session_buffer = {}
processed_rows = []
session_count = 1

weights = {'kyokansei': 0.1, 'igakuseikakusei': 0.25, 'anzensei': 0.35, 'yuugaisei': 0.2, 'aiirai': 0.05, 'shikafanshi': 0.05}
negative_cols = ['igakuseikakusei', 'anzensei', 'yuugaisei', 'aiirai', 'shikafanshi']
risk_score = df_point[negative_cols].mul([weights[col] for col in negative_cols]).sum(axis=1)
bonus_score = df_point['kyokansei'] * weights['kyokansei']
df_point['satisfaction'] = (100 - (risk_score * 10) + (bonus_score * 10))
point_result_dict = dict(zip(df_point['userId'], df_point['satisfaction']))


def create_record(s, reply_row, count):
    if reply_row['action'] == 'ReplyInterruptPersona':
        effective_parent = s['suggested']
    elif reply_row['action'] == 'ReplyCurrentPersona':
        effective_parent = s['current']
    else:
        effective_parent = 'Unknown'

    orchestrator_result = ''
    raw_val = s.get('orchestrator_raw')
    if isinstance(raw_val, str) and raw_val.strip():
        try:
            parsed = json.loads(raw_val)
            orchestrator_result = json.dumps(parsed, ensure_ascii=False)
        except (json.JSONDecodeError, ValueError):
            orchestrator_result = raw_val

    return {
        'session_id': f'S{count:03d}',
        'userId': s['user_id'],
        'point': point_result_dict.get(s['user_id']),
        'persona': effective_parent,
        'replyType': reply_row['action'],
        'userInput': s['input'],
        'replyText': reply_row['replyText'],
        'orchestratorRaw': orchestrator_result
    }


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
            'orchestrator_raw': row['orchestratorRaw']
        }
        orchestrator_queue.append(p_id)

    elif row['action'] in ['ReplyCurrentPersona', 'ReplyInterruptPersona']:
        if orchestrator_queue:
            target_pid = orchestrator_queue[0]
            if row['action'] == 'ReplyCurrentPersona':
                session_buffer[target_pid]['current_reply'] = row
                if session_buffer[target_pid]['suggested'] == session_buffer[target_pid]['current']:
                    orchestrator_queue.popleft()
            else:
                session_buffer[target_pid]['interrupt_reply'] = row
                orchestrator_queue.popleft()

    keys_to_delete = []
    for p_id, s in session_buffer.items():
        has_interrupt_needed = (s['suggested'] != s['current'])
        if (s['current_reply'] is not None) and ((not has_interrupt_needed) or (s['interrupt_reply'] is not None)):
            if s['interrupt_reply'] is not None:
                processed_rows.append(create_record(s, s['interrupt_reply'], session_count))
            else:
                processed_rows.append(create_record(s, s['current_reply'], session_count))
            session_count += 1
            keys_to_delete.append(p_id)

    for p_id in keys_to_delete:
        del session_buffer[p_id]

df_final = pd.DataFrame(processed_rows)
print(f"[Step 2] 会话配对完成: {len(df_final)}行")

FIXED_KEYWORD_TEXTS = ["もちこが共感", "ほっこりおはなし", "もちことモヤモヤまとめ", "励ましもちこ", "一緒に調べる", "ぺん先生にきいてみる", "むちことおしゃべり"]
TOOL_COMMAND_TEXTS = ["うちのこいくつ？", "子育てまんだら", "AIたちの楽しい使い方", "研究同意の確認", "ストレッチを始める", "スクワットを始める", "筋トレ動画をみる", "ストレスBOMB", "ID"]
CONSENT_KEYWORDS = "その機能を使うには、まず研究利用への同意が必要"
ALL_EXCLUDED_TEXTS = FIXED_KEYWORD_TEXTS + TOOL_COMMAND_TEXTS
MEANINGLESS_PHRASES = ["そうかも", "そうかもね", "かもしれない", "かも", "かもね", "なんか", "なんだか", "ちょっとどうしよう", "どうしよう", "たぶん", "おそらく", "きっと大丈夫", "大丈夫", "ありがとう", "分かりました"]

persona_switch_pattern = re.compile(r'(ペン先生|もちこ|むちこ).*?(お話しする|きいてみる|相談する|に交代する|相談してみる|相談します)', re.IGNORECASE)


def is_meaningless_input(text):
    if not isinstance(text, str):
        return True
    stripped = text.strip()
    if not stripped:
        return True
    if persona_switch_pattern.search(stripped):
        return True
    if stripped in ALL_EXCLUDED_TEXTS:
        return True
    if CONSENT_KEYWORDS in stripped:
        return True
    if stripped in MEANINGLESS_PHRASES:
        return True
    return False


mask_valid = (df_final['userInput'].str.len() >= 4) & (~df_final['userInput'].apply(is_meaningless_input))
df_clean = df_final[mask_valid].copy()
print(f"[Step 3] 过滤后: {len(df_clean)}行 (排除{len(df_final)-len(df_clean)}行)")

df_clean = df_clean.drop_duplicates(subset=['userId', 'replyType', 'userInput', 'persona'], keep='first')
print(f"[Step 4] 去重后: {len(df_clean)}行")

df_clean['userInput'] = df_clean['userInput'].str.replace('\\n', ' ').str.replace('\\t', ' ').str.replace('\\r', ' ')
df_clean['replyText'] = df_clean['replyText'].str.replace('\\n', ' ').str.replace('\\t', ' ').str.replace('\\r', ' ')

df_with_id = df_clean[df_clean['point'].notna()].copy()
df_cant_find_id = df_clean[df_clean['point'].isna()].copy()
df_chatbot_mo = df_with_id[df_with_id['persona'] == 'mochiko'].copy()
df_chatbot_p = df_with_id[df_with_id['persona'] == 'pen_sensei'].copy()
print(f"[Step 5] 分组: with_id={len(df_with_id)}行, cant_find_id={len(df_cant_find_id)}行")
print(f"[Step 6] 人格分离: chatbot_mo={len(df_chatbot_mo)}行, chatbot_p={len(df_chatbot_p)}行")

ids_chat = pd.DataFrame(df_chat['userId'].unique(), columns=['chat_userId'])
ids_point = pd.DataFrame(df_point['userId'].unique(), columns=['point_userId'])
df_id_comparison = pd.merge(ids_chat, ids_point, left_on='chat_userId', right_on='point_userId', how='outer')

df_id_comparison.to_csv(config.DATA_DIR / 'id_comparison.csv', index=False, encoding='utf-8-sig')
df_final.to_csv(config.DATA_DIR / 'data.csv', index=False, encoding='utf-8-sig')
df_clean.to_csv(config.DATA_DIR / 'cleaned_data.csv', index=False, encoding='utf-8-sig')
df_with_id.to_csv(config.DATA_DIR / 'data_with_id.csv', index=False, encoding='utf-8-sig')
df_cant_find_id.to_csv(config.DATA_DIR / 'data_cant_find_id.csv', index=False, encoding='utf-8-sig')
df_chatbot_mo.to_csv(config.DATA_DIR / 'data_chatbot_mo.csv', index=False, encoding='utf-8-sig')
df_chatbot_p.to_csv(config.DATA_DIR / 'data_chatbot_p.csv', index=False, encoding='utf-8-sig')

print(f"[Done] 共导出7个CSV文件")
