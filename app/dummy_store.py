# super-simple, not thread-safe – good enough for local testing
_history = {}          # {uid: [ {role, content} ... ]}

def get_history(uid, limit=20):
    return _history.get(uid, [])[-limit:]

def write_turns(uid, user_msg, assistant_msg):
    _history.setdefault(uid, []).extend([
        {"role": "user", "content": user_msg},
        {"role": "assistant", "content": assistant_msg}
    ])


_topic_counts: dict[str, dict[str, int]] = {}  # uid -> {topic: count}

def inc_topic(uid: str, topic: str):
    _topic_counts.setdefault(uid, {})
    _topic_counts[uid][topic] = _topic_counts[uid].get(topic, 0) + 1

def topic_hits(uid: str, topic: str) -> int:
    return _topic_counts.get(uid, {}).get(topic, 0)


# add near the existing in-memory dicts
_user_mem: dict[str, dict] = {}

def remember_resource(uid: str, title: str) -> None:
    _user_mem.setdefault(uid, {})["last_resource"] = title

def last_resource(uid: str) -> str | None:
    return _user_mem.get(uid, {}).get("last_resource")
