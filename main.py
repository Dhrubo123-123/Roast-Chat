import os
import re
import json
from collections import Counter
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from groq import Groq

app = FastAPI(title="RoastChat — Desi Savage Edition")

# CORS setup for frontend-backend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

# 1. FIXED ROOT ROUTE: Prevents the 404 errors in your Render logs
@app.get("/", response_class=HTMLResponse)
async def home():
    return """
    <body style="font-family: sans-serif; background: #121212; color: white; display: flex; align-items: center; justify-content: center; height: 100vh;">
        <div style="text-align: center; border: 1px solid #333; padding: 40px; border-radius: 10px;">
            <h1 style="color: #ff4b2b;">🔥 RoastChat API is Online</h1>
            <p>Ready to destroy your alumni group's ego.</p>
            <p style="color: #666; font-size: 0.8rem;">Endpoint: <code>POST /analyze</code></p>
        </div>
    </body>
    """

# 2. UPDATED PARSER: Handles the [DD/MM/YY, HH:MM:SS PM] format and narrow spaces (\u202f)
def parse_whatsapp_chat(text: str):
    # Regex for iOS/Android bracket format [cite: 23, 26]
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
            
            # Filter system logs and media omissions [cite: 24, 27, 47]
            if any(x in content for x in ["Messages and calls", "created this group", "added", "image omitted"]):
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

# 3. STATS ENGINE: Calculates the 'Main Character' and 'Link Spam' [cite: 44, 107]
def get_chat_stats(messages):
    sender_counts = Counter(m['sender'] for m in messages)
    top_senders = sender_counts.most_common(2)
    
    if not top_senders: return None
    
    p1 = top_senders[0][0]
    p2 = top_senders[1][0] if len(top_senders) > 1 else "The Ghost"
    
    p1_msgs = [m['content'] for m in messages if m['sender'] == p1]
    link_count = sum(1 for c in p1_msgs if "http" in c or "facebook.com" in c or "youtu.be" in c) [cite: 28, 46, 80]
    
    # FIX: Safe Emoji Detection for Python 3.14 (Universal Range)
    emoji_pattern = re.compile(r'[\U00010000-\U0010ffff]', flags=re.UNICODE)
    emoji_total = sum(len(emoji_pattern.findall(m['content'])) for m in messages)
    
    return {
        "p1": p1,
        "p2": p2,
        "p1_link_spam": link_count,
        "emoji_count": emoji_total,
        "total_msgs": len(messages),
        "vibe": "Political Talkshow" if link_count > 10 else "Nostalgia Trip"
    }

# 4. ROAST ENDPOINT: The 'Desi Boy Next Door' AI logic
@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    if not GROQ_API_KEY:
        raise HTTPException(500, "GROQ_API_KEY missing in Render config.")

    content = await file.read()
    try:
        text = content.decode('utf-8-sig') # Best for WhatsApp exports
    except:
        text = content.decode('latin-1')
        
    messages = parse_whatsapp_chat(text)
    if len(messages) < 10:
        raise HTTPException(400, "At least give me 10 messages to roast. This is too small.")

    stats = get_chat_stats(messages)
    
    # The "Angy but Fantastic Comic Timing" Persona
    system_instruction = (
        "You are the 'Savage Boy Next Door' from Kolkata. You're funny, sharp, and absolutely "
        "tired of people using alumni groups as their personal political newsletters. "
        "Roast the chat stats provided. Use terms like 'WhatsApp University,' 'Middle-class trap,' "
        "and 'Digital Abhimanyu.' If one guy is spamming news links, call out his 'unpaid internship' at a news channel. "
        "Be brutally honest but hilarious. "
        "Respond ONLY in a JSON object with these keys: headline, risk_label, opening_roast, red_flags, brutal_truth, roast_lines."
    )
    
    # Give the AI a taste of the actual conversation [cite: 52, 107]
    recent_context = [f"{m['sender']}: {m['content'][:60]}" for m in messages[-20:]]
    
    client = Groq(api_key=GROQ_API_KEY)
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": f"Stats: {json.dumps(stats)}. Recent Messages: {recent_context}"}
            ],
            response_format={"type": "json_object"}
        )
        return {"stats": stats, "roast": json.loads(response.choices[0].message.content)}
    except Exception as e:
        raise HTTPException(500, f"AI Error: {str(e)}")
