import json
import re
from pathlib import Path
from datetime import datetime
import ollama

MEMORY_FILE = Path(__file__).parent / "memory.json"


# ── Storage helpers ───────────────────────────────────────────────────────────
def _load() -> list:
    try:
        if MEMORY_FILE.exists():
            return json.loads(MEMORY_FILE.read_text(encoding="utf-8"))
    except Exception:
        pass
    return []


def _save(memories: list) -> None:
    MEMORY_FILE.write_text(
        json.dumps(memories, ensure_ascii=False, indent=2), encoding="utf-8"
    )


# ── CRUD ──────────────────────────────────────────────────────────────────────
def load_all() -> list:
    return _load()


def add(key: str, value: str) -> dict:
    memories = _load()
    new_id = max((m["id"] for m in memories), default=0) + 1
    entry = {
        "id": new_id,
        "key": key.strip(),
        "value": value.strip(),
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }
    memories.append(entry)
    _save(memories)
    return entry


def update(mem_id: int, key: str, value: str) -> bool:
    memories = _load()
    for m in memories:
        if m["id"] == mem_id:
            m["key"]        = key.strip()
            m["value"]      = value.strip()
            m["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M")
            _save(memories)
            return True
    return False


def delete(mem_id: int) -> bool:
    memories = _load()
    filtered = [m for m in memories if m["id"] != mem_id]
    if len(filtered) < len(memories):
        _save(filtered)
        return True
    return False


def clear_all() -> None:
    _save([])


# ── Context for LLM ──────────────────────────────────────────────────────────
def get_context() -> str:
    memories = _load()
    if not memories:
        return ""
    lines = [f"- {m['key']}: {m['value']}" for m in memories]
    return "📝 معلومات مهمة يجب تذكرها دائماً عن المستخدم:\n" + "\n".join(lines)


# ── Auto-extraction ───────────────────────────────────────────────────────────
def extract_from_message(message: str, model: str) -> list:
    """Ask the LLM to extract personal facts from a user message."""
    try:
        resp = ollama.chat(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "أنت مساعد يحلل النص ويستخرج الكيانات المذكورة فيه. "
                        "أجب بـ JSON فقط لا غير، بهذا الشكل: "
                        '[{"key": "نوع", "value": "قيمة"}] '
                        "أمثلة للأنواع: الاسم، اسم المدير، اسم الابن الكبير، اسم الابن الصغير، "
                        "رقم الهاتف، العنوان، المهنة، اسم الصديق. "
                        "إذا لم يذكر شيء يستحق الحفظ، أجب بـ: []"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"حلل هذه الجملة واستخرج الكيانات منها كـ JSON:\n{message}"
                    ),
                },
            ],
            options={"temperature": 0, "num_predict": 300},
        )
        content = resp["message"]["content"].strip()
        match   = re.search(r"\[.*?\]", content, re.DOTALL)
        if match:
            extracted = json.loads(match.group())
            if isinstance(extracted, list):
                return [
                    e for e in extracted
                    if isinstance(e, dict) and e.get("key") and e.get("value")
                ]
    except Exception:
        pass
    return []
