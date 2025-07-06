# super-simple, not thread-safe – good enough for local testing
_history = {}          # {uid: [ {role, content} ... ]}

def get_history(uid, limit=20):
    return _history.get(uid, [])[-limit:]

def write_turns(uid, user_msg, assistant_msg):
    _history.setdefault(uid, []).extend([
        {"role": "user", "content": user_msg},
        {"role": "assistant", "content": assistant_msg}
    ])
