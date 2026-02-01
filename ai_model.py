# ai_model.py
import os
import json
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

GROQ_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_KEY:
    raise Exception("GROQ_API_KEY missing")

client = Groq(api_key=GROQ_KEY)

PROMPT_TEMPLATE = """
أنت مساعد ذكي لتحليل المعاملات المالية باللغة العربية.
النص قد يحتوي على عدة معاملات في نفس الجملة.
قم بتحليل كل عملية وارجع JSON List فقط، مثال:

[
  {{"amount":300,"category":"ملابس","item":"فستان","place":"المول","type":"shopping"}},
  {{"amount":50,"category":"مشروبات","item":"عصير","place":"القهوة","type":"food"}}
]

احرص على:
- اقتراح category ذكي لكل عملية.
- تحديد type ذكي لكل عملية (مثل: shopping, food, health, education, entertainment, bills).
- استخراج المبلغ بدقة.
- item و place إذا موجودين.
- لا تخرج عن JSON List.
- اجعل كل object مكتمل بدون قيم null إلا إذا فعلاً غير موجودة.

النص:
"{text}"
"""

def analyze_text(text: str) -> list:
    """
    يحلل نص عربي ويخرج قائمة JSON لكل عملية، مع اقتراح type ذكي.
    """
    prompt = PROMPT_TEMPLATE.format(text=text)

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )

        output = response.choices[0].message.content.strip()

        # محاولة استخراج قائمة JSON
        start = output.find("[")
        end = output.rfind("]") + 1
        parsed_list = json.loads(output[start:end])

        # ضمان القيم لكل عملية
        for parsed in parsed_list:
            parsed["amount"] = parsed.get("amount")
            parsed["category"] = parsed.get("category") or "Other"
            parsed["item"] = parsed.get("item")
            parsed["place"] = parsed.get("place")
            parsed["type"] = parsed.get("type") or "expense"

        return parsed_list

    except Exception as e:
        return [{"error": str(e)}]
