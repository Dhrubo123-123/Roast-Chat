import os
import re
import json
from datetime import datetime
from collections import Counter, defaultdict
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from groq import Groq
import statistics

app = FastAPI(title="RoastChat API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

def parse_whatsapp_chat(text: str):
    patterns = [
        r'(\d{1,2}/\d{1,2}/\d{2,4}),?\s+(\d{1,2}:\d{2}(?::\d{2})?(?:\s?[AP]M)?)\s*[-\u2013]\s*([^:]+?):\s*(.*)',
        r'\[(\d{1,2}/\d{1,2}/\d{2,4}),?\s+(\d{1,2}:\d{2}(?::\d{2})?(?:\s?[AP]M)?)\]\s*([^:]+?):\s*(.*)',
    ]
    messages = []
    lines = text.split('\n')
    current_msg = None
    for line in lines:
        matched = False
        for pat in patterns:
            m = re.match(pat, line.strip())
            if m:
                if current_msg:
                    messages.append(current_msg)
                date_str, time_str, sender, content = m.groups()
                sender = sender.strip()
                if any(x in content.lower() for x in ['messages and calls are end', 'created group', 'added', 'left', 'changed the subject', '<media omitted>']):
                    current_msg = None
                    matched = True
                    break
                current_msg = {
                    'date': date_str,
                    'time': time_str,
                    'sender': sender,
                    'content': content.strip(),
                    'timestamp': parse_timestamp(date_str, time_str)
                }
                matched = True
                break
        if not matched and current_msg and line.strip():
            current_msg['content'] += ' ' + line.strip()
    if current_msg:
        messages.append(current_msg)
    return messages


def parse_timestamp(date_str, time_str):
    try:
        formats = [
            '%m/%d/%y %I:%M %p', '%d/%m/%y %I:%M %p',
            '%m/%d/%Y %I:%M %p', '%d/%m/%Y %I:%M %p',
            '%m/%d/%y %H:%M', '%d/%m/%y %H:%M',
            '%m/%d/%Y %H:%M', '%d/%m/%Y %H:%M',
        ]
        combined = f"{date_str} {time_str}".strip()
        for fmt in formats:
            try:
                return datetime.strptime(combined, fmt)
            except:
                continue
    except:
        pass
    return None


def analyze_chat(messages: list, chat_type: str = "relationship"):
    if not messages:
        return None
    senders = list(set(m['sender'] for m in messages))
    if len(senders) < 2:
        return None
    sender_counts = Counter(m['sender'] for m in messages)
    top2 = [s for s, _ in sender_counts.most_common(2)]
    p1, p2 = top2[0], top2[1]
    msgs = [m for m in messages if m['sender'] in [p1, p2]]
    total = len(msgs)
    p1_msgs = [m for m in msgs if m['sender'] == p1]
    p2_msgs = [m for m in msgs if m['sender'] == p2]
    p1_count = len(p1_msgs)
    p2_count = len(p2_msgs)
    p1_pct = round(p1_count / total * 100)
    p2_pct = 100 - p1_pct
    p1_words = sum(len(m['content'].split()) for m in p1_msgs)
    p2_words = sum(len(m['content'].split()) for m in p2_msgs)
    avg_p1_len = round(p1_words / p1_count) if p1_count else 0
    avg_p2_len = round(p2_words / p2_count) if p2_count else 0
    short_words = {'k', 'ok', 'okay', 'hmm', 'hm', 'oh', 'ha', 'haha', 'yes', 'no', 'sure', 'fine', 'nice', 'good', 'seen', 'noted'}
    p1_short = sum(1 for m in p1_msgs if len(m['content'].split()) <= 2 or m['content'].lower().strip() in short_words)
    p2_short = sum(1 for m in p2_msgs if len(m['content'].split()) <= 2 or m['content'].lower().strip() in short_words)
    p1_short_pct = round(p1_short / p1_count * 100) if p1_count else 0
    p2_short_pct = round(p2_short / p2_count * 100) if p2_count else 0
    p1_questions = sum(1 for m in p1_msgs if '?' in m['content'])
    p2_questions = sum(1 for m in p2_msgs if '?' in m['content'])
    p1_q_pct = round(p1_questions / p1_count * 100) if p1_count else 0
    p2_q_pct = round(p2_questions / p2_count * 100) if p2_count else 0
    unanswered_p1 = 0
    for i, m in enumerate(msgs[:-1]):
        if m['sender'] == p1 and '?' in m['content']:
            next_msgs = [msgs[j] for j in range(i+1, min(i+4, len(msgs))) if msgs[j]['sender'] == p2]
            if next_msgs:
                if not any('?' in nm['content'] or len(nm['content'].split()) > 3 for nm in next_msgs):
                    unanswered_p1 += 1
    double_texts_p1 = 0
    double_texts_p2 = 0
    for i in range(1, len(msgs)):
        if msgs[i]['sender'] == msgs[i-1]['sender']:
            if msgs[i]['sender'] == p1:
                double_texts_p1 += 1
            else:
                double_texts_p2 += 1
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"
        u"\U0001F300-\U0001F5FF"
        u"\U0001F680-\U0001F1FF"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        "]+", flags=re.UNICODE)
    p1_emojis = sum(len(emoji_pattern.findall(m['content'])) for m in p1_msgs)
    p2_emojis = sum(len(emoji_pattern.findall(m['content'])) for m in p2_msgs)
    def is_late_night(m):
        if m['timestamp']:
            h = m['timestamp'].hour
            return h >= 22 or h <= 3
        return False
    p1_late = sum(1 for m in p1_msgs if is_late_night(m))
    p2_late = sum(1 for m in p2_msgs if is_late_night(m))
    day_counts = Counter()
    for m in msgs:
        if m['timestamp']:
            day_counts[m['timestamp'].strftime('%A')] += 1
    long_p1 = sorted(p1_msgs, key=lambda m: len(m['content']), reverse=True)[:3]
    long_p2 = sorted(p2_msgs, key=lambda m: len(m['content']), reverse=True)[:3]
    convo_starts = {'p1': 0, 'p2': 0}
    sorted_msgs = sorted([m for m in msgs if m['timestamp']], key=lambda x: x['timestamp'])
    for i in range(1, len(sorted_msgs)):
        gap = (sorted_msgs[i]['timestamp'] - sorted_msgs[i-1]['timestamp']).total_seconds()
        if gap > 7200:
            if sorted_msgs[i]['sender'] == p1:
                convo_starts['p1'] += 1
            else:
                convo_starts['p2'] += 1
    total_starts = convo_starts['p1'] + convo_starts['p2']
    p1_start_pct = round(convo_starts['p1'] / total_starts * 100) if total_starts else 50
    risk_score = 50
    imbalance = abs(p1_pct - p2_pct)
    if imbalance > 30: risk_score += 15
    elif imbalance > 20: risk_score += 8
    elif imbalance > 10: risk_score += 4
    if p2_short_pct > 40: risk_score += 12
    elif p2_short_pct > 25: risk_score += 6
    if p1_start_pct > 70: risk_score += 12
    elif p1_start_pct > 60: risk_score += 6
    if unanswered_p1 > 5: risk_score += 8
    elif unanswered_p1 > 2: risk_score += 4
    if double_texts_p1 > double_texts_p2 * 1.5: risk_score += 5
    if p1_questions > p2_questions * 2: risk_score += 5
    if p2_emojis < p1_emojis * 0.3: risk_score += 5
    risk_score = min(95, max(15, risk_score))
    return {
        "p1": p1, "p2": p2, "total_messages": total,
        "p1_pct": p1_pct, "p2_pct": p2_pct,
        "p1_words": p1_words, "p2_words": p2_words,
        "avg_p1_len": avg_p1_len, "avg_p2_len": avg_p2_len,
        "p1_short_pct": p1_short_pct, "p2_short_pct": p2_short_pct,
        "p1_questions": p1_questions, "p2_questions": p2_questions,
        "p1_q_pct": p1_q_pct, "p2_q_pct": p2_q_pct,
        "unanswered_p1": unanswered_p1,
        "double_texts_p1": double_texts_p1, "double_texts_p2": double_texts_p2,
        "p1_emojis": p1_emojis, "p2_emojis": p2_emojis,
        "p1_late": p1_late, "p2_late": p2_late,
        "p1_start_pct": p1_start_pct, "convo_starts": convo_starts,
        "risk_score": risk_score,
        "sample_quotes_p1": [m['content'][:120] for m in long_p1],
        "sample_quotes_p2": [m['content'][:120] for m in long_p2],
        "chat_type": chat_type,
        "peak_days": dict(day_counts.most_common(3)),
    }


def generate_roast(stats: dict) -> dict:
    client = Groq(api_key=GROQ_API_KEY)
    chat_type = stats.get("chat_type", "relationship")
    p1 = stats["p1"]
    p2 = stats["p2"]
    risk = stats["risk_score"]
    if chat_type == "relationship":
        system_prompt = """You are RoastChat — a brutally honest, savage but insightful WhatsApp chat analyzer. 
You roast with zero sugarcoating. Indian desi Gen Z slang welcome (bhai, yaar, bro, fr, ngl). 
Be psychologically sharp, funny, and hit where it hurts but stays real. No toxic positivity.
You MUST respond with ONLY valid JSON, no markdown, no extra text."""
        user_prompt = f"""Analyze this WhatsApp relationship chat data and generate a SAVAGE roast report.

DATA:
- Person being analyzed: {p1}
- Other person: {p2}
- Total messages: {stats['total_messages']}
- {p1} sent: {stats['p1_pct']}% of messages ({stats['p1_words']} words total, avg {stats['avg_p1_len']} words/msg)
- {p2} sent: {stats['p2_pct']}% of messages ({stats['p2_words']} words total, avg {stats['avg_p2_len']} words/msg)
- {p1} short replies: {stats['p1_short_pct']}% | {p2} short replies: {stats['p2_short_pct']}%
- {p1} asked questions: {stats['p1_questions']} times | {p2} asked: {stats['p2_questions']} times
- {p1} questions unanswered: {stats['unanswered_p1']} times
- {p1} double-texted: {stats['double_texts_p1']} times | {p2}: {stats['double_texts_p2']} times
- {p1} emojis: {stats['p1_emojis']} | {p2} emojis: {stats['p2_emojis']}
- {p1} started {stats['p1_start_pct']}% of conversations
- Risk score: {risk}/100
- Sample quotes from {p2}: {stats['sample_quotes_p2'][:2]}

Respond ONLY with this JSON:
{{
  "headline": "One savage 8-10 word verdict about this relationship dynamic",
  "risk_label": "One brutal 3-5 word label",
  "opening_roast": "2-3 sentence savage opening. Hit hard immediately.",
  "red_flags": ["flag1", "flag2", "flag3", "flag4"],
  "pattern_insights": ["insight1", "insight2", "insight3"],
  "brutal_truth": "One final paragraph of harsh psychological truth.",
  "silver_lining": "One genuinely useful piece of advice (slightly sarcastic)",
  "roast_lines": ["punchy line 1", "punchy line 2", "punchy line 3"]
}}"""
    else:
        system_prompt = """You are RoastChat — a brutally honest WhatsApp workplace chat analyzer.
Corporate BS detector. Sharp, professional but savage. Expose power dynamics and gaslighting patterns.
You MUST respond with ONLY valid JSON, no markdown, no extra text."""
        user_prompt = f"""Analyze this WhatsApp workplace chat and generate a brutal honest report.

DATA:
- Employee: {p1}
- Boss/Colleague: {p2}
- Total messages: {stats['total_messages']}
- {p1} sent: {stats['p1_pct']}% | {p2} sent: {stats['p2_pct']}%
- {p1} avg msg length: {stats['avg_p1_len']} words | {p2}: {stats['avg_p2_len']} words
- {p1} questions asked: {stats['p1_questions']} | ignored: {stats['unanswered_p1']} times
- {p1} double-texted: {stats['double_texts_p1']} times
- {p1} initiated {stats['p1_start_pct']}% of conversations
- Risk score: {risk}/100

Respond ONLY with this JSON:
{{
  "headline": "One sharp 8-10 word verdict about this workplace dynamic",
  "risk_label": "3-5 word brutal label",
  "opening_roast": "2-3 sentence sharp opening about this workplace dynamic",
  "red_flags": ["flag1", "flag2", "flag3", "flag4"],
  "pattern_insights": ["insight1", "insight2", "insight3"],
  "brutal_truth": "Honest assessment of what is really happening",
  "silver_lining": "One practical career advice",
  "roast_lines": ["observation 1", "observation 2", "observation 3"]
}}"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.85,
        max_tokens=1200,
    )
    raw = response.choices[0].message.content.strip()
    raw = re.sub(r'^```json\s*', '', raw)
    raw = re.sub(r'\s*```$', '', raw)
    return json.loads(raw)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
async def serve_index():
    return FileResponse("static/index.html")


@app.post("/analyze")
async def analyze(file: UploadFile = File(...), chat_type: str = "relationship"):
    if not file.filename.endswith('.txt'):
        raise HTTPException(400, "Only .txt WhatsApp exports supported")
    content = await file.read()
    try:
        text = content.decode('utf-8')
    except:
        try:
            text = content.decode('latin-1')
        except:
            raise HTTPException(400, "Could not read file. Ensure it's a WhatsApp .txt export.")
    messages = parse_whatsapp_chat(text)
    if len(messages) < 20:
        raise HTTPException(400, "Too few messages found. Make sure this is a valid WhatsApp chat export.")
    stats = analyze_chat(messages, chat_type)
    if not stats:
        raise HTTPException(400, "Could not identify two participants in this chat.")
    if not GROQ_API_KEY:
        raise HTTPException(500, "GROQ_API_KEY not configured.")
    try:
        roast = generate_roast(stats)
    except json.JSONDecodeError:
        raise HTTPException(500, "AI response parsing failed. Try again.")
    except Exception as e:
        raise HTTPException(500, f"AI error: {str(e)}")
    return JSONResponse({"stats": stats, "roast": roast})


@app.get("/health")
async def health():
    return {"status": "roasting", "groq": bool(GROQ_API_KEY)}
