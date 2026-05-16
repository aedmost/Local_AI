import os
import threading
from pathlib import Path
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import rag
import memory

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024  # 100 MB


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    files     = sorted(f.name for f in rag.FILES_DIR.iterdir() if f.is_file())
    doc_count = rag.get_collection().count()
    models    = rag.list_models()
    return render_template("index.html", files=files, doc_count=doc_count, models=models)


@app.route("/chat", methods=["POST"])
def chat():
    data        = request.json or {}
    messages    = data.get("messages", [])
    model       = data.get("model", "llama3.2:3b")
    use_rag     = data.get("use_rag", True)
    auto_memory = data.get("auto_memory", False)
    if not messages:
        return jsonify({"error": "لا توجد رسائل"}), 400
    try:
        reply = rag.chat(messages, model=model, use_rag=use_rag)
    except Exception as e:
        err = str(e)
        if "connection" in err.lower() or "refused" in err.lower():
            return jsonify({"error": "Ollama غير مشغّل — شغّل Ollama أولاً"}), 503
        return jsonify({"error": err}), 500

    saved_memories = []
    if auto_memory:
        user_msg = messages[-1]["content"]
        try:
            extracted = memory.extract_from_message(user_msg, model)
            for item in extracted:
                entry = memory.add(item["key"], item["value"])
                saved_memories.append(entry)
        except Exception:
            pass

    return jsonify({"reply": reply, "saved_memories": saved_memories})


# ── Memory routes ─────────────────────────────────────────────────────────────
@app.route("/memories", methods=["GET"])
def list_memories():
    return jsonify(memory.load_all())


@app.route("/memories/add", methods=["POST"])
def add_memory():
    data = request.json or {}
    key  = data.get("key", "").strip()
    val  = data.get("value", "").strip()
    if not key or not val:
        return jsonify({"error": "key و value مطلوبان"}), 400
    entry = memory.add(key, val)
    return jsonify(entry)


@app.route("/memories/update", methods=["POST"])
def update_memory():
    data = request.json or {}
    ok   = memory.update(int(data.get("id", 0)), data.get("key", ""), data.get("value", ""))
    return jsonify({"ok": ok})


@app.route("/memories/delete", methods=["POST"])
def delete_memory():
    mid = int((request.json or {}).get("id", 0))
    ok  = memory.delete(mid)
    return jsonify({"ok": ok})


@app.route("/memories/clear", methods=["POST"])
def clear_memories():
    memory.clear_all()
    return jsonify({"ok": True})


@app.route("/upload", methods=["POST"])
def upload():
    allowed = {".txt", ".pdf", ".docx", ".md", ".xlsx", ".xls"}
    files   = request.files.getlist("files")
    saved   = []
    for f in files:
        ext = Path(f.filename).suffix.lower()
        if f and ext in allowed:
            fname = secure_filename(f.filename)
            f.save(rag.FILES_DIR / fname)
            saved.append(fname)
    return jsonify({"saved": saved})


@app.route("/index_docs", methods=["POST"])
def index_docs():
    try:
        result = rag.index_files()
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/delete_file", methods=["POST"])
def delete_file():
    fname = (request.json or {}).get("name", "")
    path  = rag.FILES_DIR / secure_filename(fname)
    if path.exists() and path.is_file():
        path.unlink()
    return jsonify({"ok": True})


@app.route("/clear_index", methods=["POST"])
def clear_index():
    rag.clear_index()
    return jsonify({"ok": True})


@app.route("/status")
def status():
    models = rag.list_models()
    return jsonify({"ollama": bool(models), "models": models, "chunks": rag.get_collection().count()})


@app.errorhandler(Exception)
def handle_exception(e):
    import traceback
    tb = traceback.format_exc()
    try:
        with open("error.log", "a", encoding="utf-8") as f:
            f.write(tb + "\n---\n")
    except Exception:
        pass
    return jsonify({"error": str(e), "traceback": tb}), 500


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print("\nLocal AI ready: http://localhost:5050\n")
    app.run(host="0.0.0.0", port=5050, debug=True, use_reloader=False)
