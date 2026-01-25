# TanchoAI

flowchart TB

%% =========================
%% BOOTSTRAP / SHARED
%% =========================
A0[FastAPI App Boot\n- init FastAPI + CORS\n- mount /static\n- init OpenAI client\n- init Firestore via ADC (firebase_admin.initialize_app)\n- set collections for crossword/wordathon] --> A1[Prompt Store\nPROMPTS / DICTIONARY_PROMPTS / ANALYSIS_PROMPTS / CONJUGATION_PROMPTS / CROSSWORD_PROMPTS\n(central prompt templates)] 
A1 --> A2[History + Memory Helpers (dummy_store)\n- get_history(uid, mode)\n- write_turns(uid, mode, turns)\n- inc_topic(uid, topic)\n- topic_hits(uid, topic)\n- remember_resource(uid, resource)\n- last_resource(uid)]
A0 --> A3[Resource Matcher\nmatch_resources(topic, limit)\n- retrieve best learning resources for a topic]

%% =========================
%% 1) CHAT (Conversation mode)
%% =========================
subgraph CHAT[Conversation Mode: /chat]
C0[/POST /chat\nchat(body)/] --> C1[_conversation_reply(...)\n- build prompt (JP/KR, casual/formal)\n- attach recent turns via _history_context()\n- call LLM\n- parse required JSON fields (wrong/fix/reply/explanation)]
C1 --> C2[_generate_chat_title(...)\n- generate short title for chat session (LLM)]
C1 --> C3[_extract_topic_with_language(...)\n- detect language\n- extract topic keyword(s)]
C3 --> C4[inc_topic(uid, topic)\n- increment topic hit counter]
C4 --> C5{Should recommend a resource?\n(_wants_resources OR hits >= threshold)}
C5 -- Yes --> C6[match_resources(topic)\n- choose top resource snippet]
C6 --> C7[remember_resource(uid, resource)\n- store last recommended resource]
C5 -- No --> C8[No recommendation]
C1 --> C9[write_turns(uid, mode, turns)\n- persist latest user+assistant turns]
C7 --> C10[Return JSON response\nwrong/fix/reply/explanation\n(+ optional resource context)]
C8 --> C10
end

%% =========================
%% 2) MENTOR
%% =========================
subgraph MENTOR[Mentor Mode: /mentor + /mentor/confirm]
M0[/POST /mentor\nmentor(body)/] --> M1[_mentor_reply(...)\n- extract topic\n- optionally include AVAILABLE_RESOURCES + RESOURCE_CONTEXT\n- produce JSON: {answer, recommendation}]
M1 --> M2[_ensure_calendar_prompt(...)\n- if FREE_SLOT exists + recommendation exists:\n  append 'add to calendar?' prompt]
M1 --> M3[write_turns(uid, mentor, turns)\n- store mentor conversation]
M2 --> M4[Return JSON\nanswer + recommendation]
M0b[/POST /mentor/confirm\nmentor_confirm(body)/] --> M5[CalendarManager integration\n- create calendar event if user confirms]
end

%% =========================
%% 3) VOICE CHAT
%% =========================
subgraph VOICE[Voice Mode: /voice_chat]
V0[/POST /voice_chat\nvoice_chat(body)/] --> V1[_voice_reply(...)\n- match register (casual/polite)\n- generate short JP/KR reply + EN translation\n- optional pronunciation correction (kana/hangul only)]
V1 --> V2[write_turns(uid, voice, turns)\n- store voice conversation text context]
V1 --> V3[Return JSON\n{jp/en/correction} or equivalent]
end

%% =========================
%% 4) DICTIONARY
%% =========================
subgraph DICT[Dictionary: /dictionary + cache ops]
D0[/POST /dictionary\ndictionary_lookup(body)/] --> D1[_dict_cache_key(...)\n- key by language + normalized word]
D1 --> D2{Cache hit?}
D2 -- Yes --> D3[Return cached DictionaryResponse]
D2 -- No --> D4[LLM Dictionary Call\n- use DICTIONARY_PROMPTS\n- produce structured JSON (meanings, POS, examples, conjugations)]
D4 --> D5[_increment_dictionary_stat(...)\n- stats tracking]
D4 --> D6[Store in in-memory cache]
D6 --> D7[Return DictionaryResponse]
D8[/GET /dictionary/cache/stats/] --> D9[Return cache stats]
D10[/DELETE /dictionary/cache/] --> D11[Clear dictionary cache]
end

%% =========================
%% 5) SENTENCE ANALYSIS
%% =========================
subgraph SA[Sentence Analysis: /analyze_sentence + /sentence_analysis + enhanced + cache ops]
S0[/POST /analyze_sentence\nanalyze_sentence(body)/] --> S1[LLM Grammar Analysis\n- basic analysis endpoint]
S2[/POST /sentence_analysis\nanalyze_sentence(body)/] --> S3[_sentence_cache_key(...)\n- key by language + normalized sentence]
S3 --> S4{Cache hit?}
S4 -- Yes --> S5[Return cached SentenceAnalysisResponse]
S4 -- No --> S6[LLM Analysis Call\n- use SENTENCE_ANALYSIS_PROMPTS\n- return scores + word-by-word analysis + improvements]
S6 --> S7[Store in cache + update stats]
S7 --> S8[Return SentenceAnalysisResponse]
S9[/POST /sentence_analysis_enhanced/] --> S10[Enhanced Analysis\n- richer structure (more fields/word breakdown)\n- still prompt-driven JSON output]
S11[/GET /sentence_analysis/cache/stats/] --> S12[Return cache stats]
S13[/DELETE /sentence_analysis/cache/] --> S14[Clear sentence analysis cache]
end

%% =========================
%% 6) CONJUGATION
%% =========================
subgraph CONJ[Conjugation: /conjugation + cache ops]
G0[/POST /conjugation\nconjugation(body)/] --> G1[Cache key by language+word]
G1 --> G2{Cache hit?}
G2 -- Yes --> G3[Return cached ConjugationResponse]
G2 -- No --> G4[LLM Conjugation Call\n- use CONJUGATION_PROMPTS\n- return verb_info + conjugation tables]
G4 --> G5[Store in cache + stats]
G5 --> G6[Return ConjugationResponse]
G7[/GET /conjugation/cache/stats/] --> G8[Return cache stats]
G9[/DELETE /conjugation/cache/] --> G10[Clear conjugation cache]
end

%% =========================
%% 7) CROSSWORD (Daily)
%% =========================
subgraph CW[Crossword: /crossword/daily]
W0[/POST /crossword/daily\nget_daily_crossword(body)/] --> W1[load_cached_puzzle(date, language)\n- fetch daily crossword from Firestore/cache]
W1 --> W2{Cache hit?}
W2 -- Yes --> W3[Return cached CrosswordPuzzle]
W2 -- No --> W4[generate_crossword_vocabulary(language, date)\n- pick vocab set for that day]
W4 --> W5[generate_crossword_clues(words, language)\n- LLM generates clue JSON via CROSSWORD_PROMPTS]
W5 --> W6[create_crossword_grid(word_data,...)\n- place words into grid + intersections]
W6 --> W7[normalize_grid_chars + trim_grid\n- clean grid chars\n- crop empty borders]
W7 --> W8[save_cached_puzzle(date, language, puzzle)\n- persist to Firestore/cache]
W8 --> W9[Return CrosswordPuzzle]
end

%% =========================
%% 8) WORDATHON (Daily)
%% =========================
subgraph WA[Word-a-thon: /wordathon/daily]
H0[/POST /wordathon/daily\nget_daily_wordathon(body)/] --> H1[load_cached_wordathon(date, language)\n- fetch daily target word from Firestore]
H1 --> H2{Cache hit?}
H2 -- Yes --> H3[Return cached WordathonResponse]
H2 -- No --> H4[Pick/Generate target word\n- choose by language + difficulty constraints]
H4 --> H5[Generate meaning/clue\n- LLM creates clue/meaning for target]
H5 --> H6[save_wordathon(date, language, puzzle)\n- persist to Firestore]
H6 --> H7[Return WordathonResponse]
end
