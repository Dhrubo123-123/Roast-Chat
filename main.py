import os
import re
import json
from collections import Counter
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from groq import Groq

app = FastAPI(title="RoastChat — New Town Edition")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

def parse_whatsapp_chat(text: str):
    """Parses the specific [DD/MM/YY, HH:MM:SS] format with Bengali support."""
    # This pattern specifically catches the narrow-space \u202f before PM/AM [cite: 23, 27]
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
            
            # Skip system noise [cite: 23, 24, 81]
            if any(x in content for x in ["Messages and calls", "created", "added", "omitted", "deleted"]):
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

def get_chat_vibe(messages):
    """Mechanical analysis of who is the 'Group Captain' vs the 'Ghost'."""
    sender_counts = Counter(m['sender'] for m in messages)
    top_two = sender_counts.most_common(2)
    
    p1 = top_two[0][0] # The "Dr. Debu" type (The Spammer) [cite: 27, 45, 107]
    p2 = top_two[1][0] if len(top_two) > 1 else "The Rest of the Ghosts"
    
    p1_msgs = [m['content'] for m in messages if m['sender'] == p1]
    link_count = sum(1 for m in p1_msgs if "http" in m or "x.com" in m or "youtu.be" in m) # [cite: 27, 45, 82]
    
    return {
        "p1": p1,
        "p2": p2,
        "p1_link_spam": link_count,
        "total": len(messages),
        "vibe": "Political Talkshow" if link_count > 5 else "Birthday Bot Factory" # [cite: 26, 122]
    }

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    if not GROQ_API_KEY:
        raise HTTPException(500, "Groq Key missing. Red Alert.")

    content = await file.read()
    try:
        text = content.decode('utf-8')
    except:
        text = content.decode('latin-1')
        
    messages = parse_whatsapp_chat(text)
    if len(messages) < 5:
        raise HTTPException(400, "Brother, this chat is shorter than a government promise. Send more.")

    stats = get_chat_vibe(messages)
    
    # AI SAVAGE PROMPT
    client = Groq(api_key=GROQ_API_KEY)
    system_msg = (
        "You are 'The Boy Next Door' from Kolkata who is tired of life but has peak comic timing. "
        "You are roasting an alumni group chat. Use terms like 'WhatsApp University,' 'Desi,' 'Middle-class trap.' "
        "Your goal is to make the user 'LOL' while feeling the burn. "
        "If someone is spamming links like Dr. Debu, tell them to start a newsletter instead of killing the group vibe. "
        "Mention 'Birati' or 'Forty-year-old mid-life crisis' if applicable. "
        "Respond ONLY in JSON with keys: headline, risk_label, opening_roast, red_flags, brutal_truth, roast_lines."
    )
    
    # We send the stats and the last few spicy messages for context 
    user_context = f"Stats: {json.dumps(stats)}. Last 10 messages: {messages[-10:]}"
    
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_context}
        ],
        response_format={"type": "json_object"}
    )
    
    return {"stats": stats, "roast": json.loads(response.choices[0].message.content)}
