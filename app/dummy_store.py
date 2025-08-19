# super-simple, not thread-safe – good enough for local testing
_history = {}  # Change structure to: {uid: {mode: [ {role, content} ... ]}}

# Update to support language-specific histories
def get_history(uid, mode="default", limit=20):
    """Get history for specific uid and mode (mode can include language)"""
    user_history = _history.get(uid, {})
    mode_history = user_history.get(mode, [])[-limit:]
    print(f"📚 get_history for uid '{uid}', mode '{mode}': {len(mode_history)} messages")
    if mode_history:
        print(f"   Last message: {mode_history[-1]['role']}: {mode_history[-1]['content'][:50]}...")
    else:
        print(f"   No history found for uid '{uid}', mode '{mode}' - this is a NEW SESSION")
    return mode_history

def write_turns(uid, user_msg, assistant_msg, mode="default"):
    """Write conversation turns for specific uid and mode (mode can include language)"""
    if uid not in _history:
        _history[uid] = {}
        print(f"🆕 Creating new history structure for uid '{uid}'")
    
    if mode not in _history[uid]:
        _history[uid][mode] = []
        print(f"🆕 Creating new mode history for uid '{uid}', mode '{mode}'")
    else:
        print(f"📝 Adding to existing history for uid '{uid}', mode '{mode}' (was {len(_history[uid][mode])} messages)")
    
    _history[uid][mode].extend([
        {"role": "user", "content": user_msg},
        {"role": "assistant", "content": assistant_msg}
    ])
    
    print(f"📊 Total messages for uid '{uid}', mode '{mode}': {len(_history[uid][mode])}")
# Keep other functions unchanged...
_topic_counts: dict[str, dict[str, int]] = {}

def inc_topic(uid: str, topic: str):
    _topic_counts.setdefault(uid, {})
    _topic_counts[uid][topic] = _topic_counts[uid].get(topic, 0) + 1

def topic_hits(uid: str, topic: str) -> int:
    return _topic_counts.get(uid, {}).get(topic, 0)

_user_mem: dict[str, dict] = {}

def remember_resource(uid: str, title: str) -> None:
    _user_mem.setdefault(uid, {})["last_resource"] = title

def last_resource(uid: str) -> str | None:
    return _user_mem.get(uid, {}).get("last_resource")



# Simple in-memory store for “pending” calendar events
_pending_calendar: dict[str, dict] = {}

def save_pending_calendar_event(uid: str, ev: dict) -> None:
    """
    ev should contain keys:
     - summary (str)
     - start (datetime)
     - duration (int, minutes)
     - description (str, optional)
    """
    _pending_calendar[uid] = ev

def get_pending_calendar_event(uid: str) -> dict | None:
    return _pending_calendar.get(uid)

def clear_pending_calendar_event(uid: str) -> None:
    _pending_calendar.pop(uid, None)
