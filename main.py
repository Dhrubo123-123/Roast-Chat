import os
import re
import json
from datetime import datetime
from collections import Counter
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from groq import Groq

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
    # Covers ALL known WhatsApp export formats including [DD/MM/YY, HH:MM:SS]
    patterns = [
        r'\[(\d{1,2}/\d{1,2}/\d{2,4}),\s*(\d{1,2}:\d{2}(?::\d{2})?(?:\s?[AP]M)?)\]\s*([^:]+?):\s*(.*)',
        r'(\d{1,2}/\d{1,2}/\d{2,4}),\s*(\d{1,2}:\d{2}(?::\d{2})?(?:\s?[AP]M)?)\s*[-\u2013]\s*([^:]+?):\s*(.*)',
    ]
    skip_phrases = [
        'messages and calls are end', 'created this group', 'created group',
        'added', 'left', 'changed the subject', 'changed this group',
        'you were added', 'security code changed', 'media omitted',
        '\u200e', 'null'
    ]
    messages = []
    current_msg = None
    for line in text.split('\n'):
        line = line.strip()
        if not line:
            continue
        matched = False
        for pat in patterns:
            m = re.match(pat, line)
            if m:
                if current_msg:
                    messages.append(current_msg)
                date_str, time_str, sender, content = m.groups()
                sender = sender.strip()
                content = content.strip()
                if any(x in content.lower() for x in skip_phrases) or any(x in sender.lower() for x in skip_phrases):
                    current_msg = None
                    matched = True
                    break
                current_msg = {
                    'date': date_str,
                    'time': time_str,
                    'sender': sender,
                    'content': content,
                    'timestamp': parse_timestamp(date_str, time_str)
                }
                matched = True
                break
        if not matched and current_msg and line:
            current_msg['content'] += ' ' + line
    if current_msg:
        messages.append(current_msg)
    return messages


def parse_timestamp(date_str, time_str):
    formats = [
        '%d/%m/%y %I:%M:%S %p', '%d/%m/%Y %I:%M:%S %p',
        '%m/%d/%y %I:%M:%S %p', '%m/%d/%Y %I:%M:%S %p',
        '%d/%m/%y %H:%M:%S', '%d/%m/%Y %H:%M:%S',
        '%m/%d/%y %H:%M:%S', '%m/%d/%Y %H:%M:%S',
        '%d/%m/%y %I:%M %p', '%d/%m/%Y %I:%M %p',
        '%m/%d/%y %I:%M %p', '%m/%d/%Y %I:%M %p',
        '%d/%m/%y %H:%M', '%d/%m/%Y %H:%M',
        '%m/%d/%y %H:%M', '%m/%d/%Y %H:%M',
    ]
    combined = f"{date_str} {time_str}".strip()
    for fmt in formats:
        try:
            return datetime.strptime(combined, fmt)
        except:
            continue
    return None


def analyze_chat(messages: list, chat_type: str = "relationship"):
    if not messages:
        return None

    sender_counts = Counter(m['sender'] for m in messages)

    # For group chats, pick top 2. For 1-on-1, it's already 2.
    top_senders = [s for s, _ in sender_counts.most_common(10) if s]
    if len(top_senders) < 2:
        return None

    p1, p2 = top_senders[0], top_senders[1]
    is_group = len(top_senders) > 2
    total_all = len(messages)

    # Filter to top 2 for deep analysis
    msgs = [m for m in messages if m['sender'] in [p1, p2]]
    total = len(msgs)
    if total < 10:
        return None

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

    short_words = {'k', 'ok', 'okay', 'hmm', 'hm', 'oh', 'ha', 'haha', 'yes', 'no',
                   'sure', 'fine', 'nice', 'good', 'seen', 'noted', 'lol', 'hehe'}
    p1_short = sum(1 for m in p1_msgs if len(m['content'].split()) <= 2 or m['content'].lower().strip() in short_words)
    p2_short = sum(1 for m in p2_msgs if len(m['content'].split()) <= 2 or m['content'].lower().strip() in short_words)
    p1_short_pct = round(p1_short / p1_count * 100) if p1_count else 0
    p2_short_pct = round(p2_short / p2_count * 100) if p2_count else 0

    p1_questions = sum(1 for m in p1_msgs if '?' in m['content'])
    p2_questions = sum(1 for m in p2_msgs if '?' in m['content'])

    unanswered_p1 = 0
    for i, m in enumerate(msgs[:-1]):
        if m['sender'] == p1 and '?' in m['content']:
            next_msgs = [msgs[j] for j in range(i+1, min(i+4, len(msgs))) if msgs[j]['sender'] == p2]
            if next_msgs:
                if not any(len(nm['content'].split()) > 3 for nm in next_msgs):
                    unanswered_p1 += 1

    double_texts_p1 = sum(1 for i in range(1, len(msgs)) if msgs[i]['sender'] == p1 == msgs[i-1]['sender'])
    double_texts_p2 = sum(1 for i in range(1, len(msgs)) if msgs[i]['sender'] == p2 == msgs[i-1]['sender'])

    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F\U0001F300-\U0001F5FF"
        u"\U0001F680-\U0001F1FF\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        "]+", flags=re.UNICODE)
    p1_emojis = sum(len(emoji_pattern.findall(m['content'])) for m in p1_msgs)
    p2_emojis = sum(len(emoji_pattern.findall(m['content'])) for m in p2_msgs)

    sorted_msgs = sorted([m for m in msgs if m['timestamp']], key=lambda x: x['timestamp'])
    convo_starts = {'p1': 0, 'p2': 0}
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

    long_p2 = sorted(p2_msgs, key=lambda m: len(m['content']), reverse=True)[:2]

    return {
        "p1": p1, "p2": p2,
        "is_group": is_group,
        "total_participants": len(top_senders),
        "total_messages": total_all,
        "p1_pct": p1_pct, "p2_pct": p2_pct,
        "p1_words": p1_words, "p2_words": p2_words,
        "avg_p1_len": avg_p1_len, "avg_p2_len": avg_p2_len,
        "p1_short_pct": p1_short_pct, "p2_short_pct": p2_short_pct,
        "p1_questions": p1_questions, "p2_questions": p2_questions,
        "unanswered_p1": unanswered_p1,
        "double_texts_p1": double_texts_p1, "double_texts_p2": double_texts_p2,
        "p1_emojis": p1_emojis, "p2_emojis": p2_emojis,
        "p1_start_pct": p1_start_pct,
        "risk_score": risk_score,
        "sample_quotes_p2": [m['content'][:120] for m in long_p2],
        "chat_type": chat_type,
    }


def extract_json(text: str) -> dict:
    """Robustly extract JSON from AI response even if wrapped in garbage."""
    text = text.strip()
    # Strip markdown code fences
    text = re.sub(r'^```(?:json)?\s*', '', text)
    text = re.sub(r'\s*```$', '', text)
    text = text.strip()
    # Try direct parse first
    try:
        return json.loads(text)
    except:
        pass
    # Find first { to last } 
    start = text.find('{')
    end = text.rfind('}')
    if start != -1 and end != -1:
        try:
            return json.loads(text[start:end+1])
        except:
            pass
    # Give up — return a fallback
    raise ValueError("Could not parse JSON from AI response")


def generate_roast(stats: dict) -> dict:
    client = Groq(api_key=GROQ_API_KEY)
    chat_type = stats.get("chat_type", "relationship")
    p1 = stats["p1"]
    p2 = stats["p2"]
    risk = stats["risk_score"]
    is_group = stats.get("is_group", False)

    if is_group:
        context = f"This is a GROUP CHAT with {stats['total_participants']} participants. Focus on the dynamic between the two most active members: {p1} and {p2}."
    else:
        context = f"This is a private 1-on-1 chat between {p1} and {p2}."

    if chat_type == "relationship":
        system_prompt = """You are RoastChat — brutally honest WhatsApp chat analyzer. 
Savage, sharp, psychologically accurate. Indian Gen Z slang welcome. No sugarcoating.
CRITICAL: You MUST respond with ONLY a valid JSON object. No explanation before or after. No markdown. Start your response with { and end with }"""

        user_prompt = f"""Roast this WhatsApp chat. {context}

STATS:
- {p1} sent {stats['p1_pct']}% of messages, avg {stats['avg_p1_len']} words each
- {p2} sent {stats['p2_pct']}% of messages, avg {stats['avg_p2_len']} words each
- {p1} short replies: {stats['p1_short_pct']}% | {p2} short replies: {stats['p2_short_pct']}%
- {p1} asked {stats['p1_questions']} questions, {stats['unanswered_p1']} ignored by {p2}
- {p1} double-texted {stats['double_texts_p1']} times | {p2} double-texted {stats['double_texts_p2']} times
- {p1} started {stats['p1_start_pct']}% of conversations
- Risk score: {risk}/100
- Sample quotes from {p2}: {stats['sample_quotes_p2']}

Return ONLY this JSON object, nothing else:
{{"headline":"savage 8-10 word verdict","risk_label":"3-5 word brutal label","opening_roast":"2-3 savage sentences","red_flags":["flag1","flag2","flag3","flag4"],"pattern_insights":["insight1","insight2","insight3"],"brutal_truth":"harsh 2-3 sentence psychological truth","silver_lining":"one sarcastic but useful advice","roast_lines":["punchy line 1","punchy line 2","punchy line 3"]}}"""

    else:
        system_prompt = """You are RoastChat — brutally honest workplace chat analyzer. Expose power dynamics.
CRITICAL: Respond with ONLY a valid JSON object. Start with { end with }. No other text."""

        user_prompt = f"""Roast this workplace WhatsApp chat. {context}

STATS:
- {p1} sent {stats['p1_pct']}% of messages
- {p2} sent {stats['p2_pct']}% of messages
- {p1} avg msg: {stats['avg_p1_len']} words | {p2}: {stats['avg_p2_len']} words
- {p1} questions asked: {stats['p1_questions']} | ignored: {stats['unanswered_p1']} times
- {p1} initiated {stats['p1_start_pct']}% of conversations
- Risk score: {risk}/100

Return ONLY this JSON object, nothing else:
{{"headline":"sharp 8-10 word workplace verdict","risk_label":"3-5 word brutal label","opening_roast":"2-3 sharp sentences","red_flags":["flag1","flag2","flag3","flag4"],"pattern_insights":["insight1","insight2","insight3"],"brutal_truth":"honest 2-3 sentence assessment","silver_lining":"one practical career advice","roast_lines":["observation 1","observation 2","observation 3"]}}"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.8,
        max_tokens=1000,
    )
    raw = response.choices[0].message.content
    return extract_json(raw)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
async def serve_index():
    return FileResponse("static/index.html")


@app.post("/analyze")
async def analyze(file: UploadFile = File(...), chat_type: str = "relationship"):
    if not file.filename.endswith('.txt'):
        raise HTTPException(400, "Only .txt WhatsApp exports supported")

    content = await file.read()
    text = None
    for enc in ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252']:
        try:
            text = content.decode(enc)
            break
        except:
            continue
    if not text:
        raise HTTPException(400, "Could not read file encoding.")

    messages = parse_whatsapp_chat(text)
    if len(messages) < 10:
        raise HTTPException(400, f"Only found {len(messages)} messages. Need at least 10. Make sure this is a WhatsApp .txt export.")

    stats = analyze_chat(messages, chat_type)
    if not stats:
        raise HTTPException(400, "Could not analyze chat. Need at least 2 participants with messages.")

    if not GROQ_API_KEY:
        raise HTTPException(500, "GROQ_API_KEY not configured on server.")

    try:
        roast = generate_roast(stats)
    except ValueError as e:
        raise HTTPException(500, f"AI returned invalid response. Please try again.")
    except Exception as e:
        raise HTTPException(500, f"AI error: {str(e)[:200]}")

    return JSONResponse({"stats": stats, "roast": roast})


@app.get("/health")
async def health():
    return {"status": "roasting", "groq": bool(GROQ_API_KEY)}
