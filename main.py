import os
import re
import json
from collections import Counter
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from groq import Groq

app = FastAPI(title="RoastChat — Desi Savage Edition")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

# 1. THE FIX: Serve your actual index.html file on the root URL
@app.get("/")
async def home():
    # This looks for index.html in your 'static' folder and serves it as the UI
    return FileResponse("static/index.html")

# 2. PARSER: Fixed for bracket format [14/08/16] and narrow-space \u202f
def parse_whatsapp_chat(text: str):
    pattern = r'\[(\d{1,2}/\d{1,2}/\d{2,4}),\s+(\d{1,2}:\d{2}(?::\d{2})?[\s\u202f]?[AP]M)\]\s*([^:]+?):\s*(.*)'
    
    messages = []
    current_msg = None
    
    for line in text.split('\n'):
        line = line.strip()
        if not line: continue
        
        m = re.match(pattern, line)
        if m:
            if current_msg:
                messages.append(current_msg)
            
            date_str, time_str, sender, content = m.groups()
            
            if any(x in content for x in ["Messages and calls", "created", "added", "omitted"]):
                current_msg = None
                continue
                
            current_msg = {
                'sender': sender.strip(),
                'content': content.strip(),
                'timestamp': f"{date_str} {time_str}".replace('\u202f', ' ')
            }
        elif current_msg:
            current_msg['content'] += ' ' + line

    if current_msg:
        messages.append(current_msg)
    return messages

# 3. STATS: Fixed Regex range crash for Python 3.14
def get_chat_stats(messages):
    sender_counts = Counter(m['sender'] for m in messages)
    top_senders = sender_counts.most_common(2)
    
    if not top_senders: return None
    
    p1 = top_senders[0][0]
    p2 = top_senders[1][0] if len(top_senders) > 1 else "The Ghost"
    
    p1_msgs = [m['content'] for m in messages if m['sender'] == p1]
    link_count = sum(1 for c in p1_msgs if "http" in c or "x.com" in c)
    
    emoji_pattern = re.compile(r'[\U00010000-\U0010ffff]', flags=re.UNICODE)
    emoji_total = sum(len(emoji_pattern.findall(m['content'])) for m in messages)
    
    return {
        "p1": p1, "p2": p2,
        "p1_link_spam": link_count,
        "emoji_count": emoji_total,
        "total_msgs": len(messages),
        "risk_score": 75 if link_count > 10 else 40
    }

# 4. ROAST: Desi Savage Personality
@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    if not GROQ_API_KEY:
        raise HTTPException(500, "GROQ_API_KEY missing.")

    content = await file.read()
    try:
        text = content.decode('utf-8-sig')
    except:
        text = content.decode('latin-1')
        
    messages = parse_whatsapp_chat(text)
    if len(messages) < 10:
        raise HTTPException(400, "Too few messages found.")

    stats = get_chat_stats(messages)
    
    system_instruction = (
        "You are 'The Boy Next Door' from Kolkata who is a comic timing legend. "
        "Roast these alumni group stats savagely. Use Hinglish/Benglish. "
        "Respond ONLY with a JSON object."
    )
    
    client = Groq(api_key=GROQ_API_KEY)
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": f"Stats: {json.dumps(stats)}"}
        ],
        response_format={"type": "json_object"}
    )
    
    return {"stats": stats, "roast": json.loads(response.choices[0].message.content)}
