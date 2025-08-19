import React, { useState, useRef } from 'react';
import axios from 'axios';
import { ReactMic } from 'react-mic';

const API_BASE = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8080';
const TEST_UID = 'demo';                          


/* ─── Google-Sheets webhook ─── */
const LOG_URL = "https://script.google.com/macros/s/AKfycbxBZ1hcCm2q5G7AdsrwErQe93ugrQoi4KMRx53jOe4jeAPHljAj11BVojzZEQHeYkc/exec";

/** Fire-and-forget row writer */
async function logTurn(mode, subMode, uid, payload) {
  fetch(LOG_URL, {
    method : 'POST',
    headers: { 'Content-Type': 'application/json' },
    body   : JSON.stringify({
      sheet : mode,               // conversation | mentor | voice
      uid,
      sub   : subMode || '',      // casual / formal or ''
      payload
    })
  }).catch(() => {});             // never block UI on log errors
}

/* ─────────────────────────────────────────────────────────── */

function formatCorrection(wrong, fix, explanation) {
  const strip = txt => txt?.replace(/<[^>]+>/g, '') ?? '';
  const explain = strip(explanation);
  if (!wrong && !fix && (!explain || explain.toLowerCase() === 'en: no errors found.' || explain.includes('오류가 발견되지 않았습니다')))
    return null;

  return (
    <div className="bg-gray-100 border-l-4 border-yellow-400 p-3 mt-2 mb-1 text-sm">
      <div className="mb-1 font-semibold">📝 Correction</div>
      {wrong && (
        <div>
          <span className="font-medium text-red-700">Wrong:</span>{' '}
          <span className="text-gray-700">{strip(wrong)}</span>
        </div>
      )}
      {fix && (
        <div>
          <span className="font-medium text-green-700">Fix:</span>{' '}
          <span className="text-gray-800">{strip(fix)}</span>
        </div>
      )}
      {explain && <div className="mt-1 italic text-gray-700">{explain}</div>}
    </div>
  );
}

/* ─────────────────────────────────────────────────────────── */

export default function App() {
  const [mode, setMode]         = useState('conversation');
  const [convType, setConvType] = useState('convCasual'); // convCasual / convFormal
  const [language, setLanguage] = useState('japanese');   // NEW: japanese / korean
  const [input, setInput]       = useState('');
  const [file, setFile]         = useState(null);
  const [loading, setLoading]   = useState(false);
  const [messages, setMessages] = useState([]);

  // voice recording
  const [isRecording, setIsRecording] = useState(false);
  const [recordedWav, setRecordedWav] = useState(null);

  const bottomRef = useRef(null);

  /* helpers */
  const pushMsg = (role, text) => {
    setMessages(prev => [...prev, { role, text }]);
    setTimeout(() => bottomRef.current?.scrollIntoView({ behavior: 'smooth' }), 10);
  };
  const updateLastAssistantMsg = updater =>
    setMessages(prev => {
      const idx = [...prev].reverse().findIndex(m => m.role === 'assistant');
      if (idx === -1) return prev;
      const realIdx = prev.length - 1 - idx;
      const copy = [...prev];
      const cur  = copy[realIdx].text || '';
      copy[realIdx].text =
        typeof updater === 'function' ? updater(cur) : updater;
      return copy;
    });

  const handleModeChange = val => {
    setMode(val);
    const label =
      val === 'conversation'
        ? `Conversation (${convType})`
        : val.charAt(0).toUpperCase() + val.slice(1);
    pushMsg('system', `―― Switched to ${label} Mode (${language}) ――`);
    setFile(null); setRecordedWav(null); setIsRecording(false);
  };

  const handleLanguageChange = val => {
    setLanguage(val);
    pushMsg('system', `―― Language changed to ${val.charAt(0).toUpperCase() + val.slice(1)} ――`);
  };

  /* ───────── Submit ───────── */
  async function handleSubmit() {
    if (mode !== 'voice' && !input) return;
    if (mode === 'voice' && !file && !recordedWav)
      return alert('Please record or select a file!');

    if (mode !== 'voice') pushMsg('user', input);
    setLoading(true);

    try {
      /* ─── Conversation ─── */
      if (mode === 'conversation') {
        const { data } = await axios.post(`${API_BASE}/chat`, {
          uid: TEST_UID,
          mode: convType,
          userMessage: input,
          language: language  // NEW: include language
        });
        const { reply, wrong, fix, explanation, chatTitle } = data;

        pushMsg(
          'assistant',
          <>
            <div>{reply}</div>
            {formatCorrection(wrong, fix, explanation)}
            {chatTitle && (
              <div className="text-xs text-gray-500 mt-1 italic">
                💡 Chat title: {chatTitle}
              </div>
            )}
          </>
        );

        await logTurn(
          'conversation',
          `${convType}_${language}`,          // Include language in subMode
          TEST_UID,
          { wrong, fix, reply, explanation, chatTitle, language }
        );
      }

      /* ─── Mentor (stream) ─── */
      else if (mode === 'mentor') {
        pushMsg('assistant', ''); // placeholder
        await doMentorStream(input);
        setLoading(false);
        setInput('');
        return;
      }

      /* ─── Voice ─── */
      else {
        const audioBlob = recordedWav || file;
        const form = new FormData();
        form.append('uid', TEST_UID);
        form.append('voiceMode', 'conversation');
        form.append('convSubMode', convType);
        form.append('language', language);  // NEW: include language
        form.append('audio', audioBlob,
          audioBlob instanceof File ? audioBlob.name : 'audio.wav');

        const { data } = await axios.post(`${API_BASE}/voice_chat`, form);
        const { jp, en, correction, transcript, ttsUrl, chatTitle } = data;

        /* user bubble */
        pushMsg('user', transcript || en || jp);

        /* assistant bubble */
        pushMsg(
          'assistant',
          <>
            <div>{jp}{en ? `\n${en}` : ''}</div>
            {formatCorrection(null, null, correction)}
            {chatTitle && (
              <div className="text-xs text-gray-500 mt-1 italic">
                💡 Chat title: {chatTitle}
              </div>
            )}
            {ttsUrl && (
              <audio controls className="mt-2">
                <source src={API_BASE + ttsUrl} type="audio/mpeg" />
                Your browser does not support audio.
              </audio>
            )}
          </>
        );

        await logTurn('voice', `${language}`, TEST_UID, { jp, en, correction, chatTitle, language });
        setFile(null); setRecordedWav(null);
      }
    } catch (err) {
      console.error(err);
      alert(err.message || err);
    } finally {
      setLoading(false);
      setInput('');
    }
  }

  /* ───────── Mentor stream helper ───────── */
  async function doMentorStream(question) {
    try {
      const res = await fetch(`${API_BASE}/mentor`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          uid: TEST_UID, 
          question,
          language: language  // NEW: include language
        })
      });
      if (!res.ok) throw new Error('HTTP ' + res.status);

      const reader  = res.body.getReader();
      const dec     = new TextDecoder();
      let buffer = '';
      let streamedContent = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += dec.decode(value, { stream: true });

        let sep;
        while ((sep = buffer.indexOf('\n\n')) !== -1) {
          const block = buffer.slice(0, sep).trim(); 
          buffer = buffer.slice(sep + 2);
          if (!block) continue;

          if (block.startsWith('event: done')) {
            // Parse the final JSON payload
            const dataMatch = block.match(/data:\s*(\{.*\})/s);
            if (dataMatch) {
              try {
                const finalData = JSON.parse(dataMatch[1]);
                const { answer, recommendation, chatTitle } = finalData;
                
                // Replace the streamed content with the final parsed answer
                let finalAnswer = answer || streamedContent;
                if (chatTitle) {
                  finalAnswer += `\n\n💡 Chat title: ${chatTitle}`;
                }
                
                updateLastAssistantMsg(() => finalAnswer);
                await logTurn('mentor', `${language}`, TEST_UID, { answer, recommendation, chatTitle, language });
              } catch (parseErr) {
                console.error('Failed to parse final JSON:', parseErr);
                // Keep the streamed content if JSON parsing fails
              }
            }
            break; // Exit the streaming loop
          } else if (block.startsWith('data:')) {
            // Stream individual tokens but try to avoid JSON fragments
            const token = block.replace(/^data:\s*/, '');
            
            // Skip obvious JSON fragments (starting with { or containing ")
            if (!token.startsWith('{') && !token.includes('"answer"') && !token.includes('"recommendation"')) {
              streamedContent += token;
              updateLastAssistantMsg(txt => txt + token);
            }
          }
        }
      }
    } catch (err) {
      updateLastAssistantMsg('[Error receiving mentor reply]');
      console.error(err);
    }
  }

  /* ───────── UI ───────── */
  return (
    <div className="flex flex-col h-screen bg-gray-100">
      <header className="p-4 bg-blue-600 text-white text-xl font-semibold">
        Tancho AI Tester - {language === 'korean' ? '한국어' : '日本語'}
      </header>

      <main className="flex-1 overflow-auto p-4 space-y-4">
        {messages.map((m, i) => (
          <div
            key={i}
            className={
              m.role === 'user'
                ? 'self-end bg-blue-500 text-white'
                : m.role === 'assistant'
                ? 'self-start bg-gray-200 text-gray-900'
                : 'self-center bg-yellow-100 text-gray-700 text-sm'
            }
            style={{ maxWidth: '40rem' }}
          >
            <div className="px-4 py-2 rounded-lg whitespace-pre-wrap">
              {typeof m.text === 'string' ? m.text : m.text}
            </div>
          </div>
        ))}
        <div ref={bottomRef} />
      </main>

      <footer className="p-4 border-t bg-white space-y-2">
        {/* mode and language pickers */}
        <div className="flex items-center space-x-3 flex-wrap">
          <select
            value={mode}
            onChange={e => handleModeChange(e.target.value)}
            className="p-2 border rounded"
          >
            <option value="conversation">Conversation</option>
            <option value="mentor">Mentor</option>
            <option value="voice">Voice</option>
          </select>

          {mode === 'conversation' && (
            <select
              value={convType}
              onChange={e => setConvType(e.target.value)}
              className="p-2 border rounded"
            >
              <option value="convCasual">Casual</option>
              <option value="convFormal">Formal</option>
            </select>
          )}

          {/* NEW: Language selector */}
          <select
            value={language}
            onChange={e => handleLanguageChange(e.target.value)}
            className="p-2 border rounded bg-green-50"
          >
            <option value="japanese">🇯🇵 Japanese</option>
            <option value="korean">🇰🇷 Korean</option>
          </select>
        </div>

        {/* input area */}
        {mode === 'voice' ? (
          <>
            <div className="flex items-center space-x-2 mt-2">
              <button
                onClick={() => {
                  if (!isRecording) {
                    setRecordedWav(null);
                    setIsRecording(true);
                  } else {
                    setIsRecording(false);
                  }
                }}
                className={`px-4 py-2 rounded ${
                  isRecording ? 'bg-red-500' : 'bg-green-600'
                } text-white`}
              >
                {isRecording ? 'Stop' : 'Record'}
              </button>
              <ReactMic
                record={isRecording}
                className="w-32 h-8"
                onStop={rec => setRecordedWav(rec.blob)}
                strokeColor="#000"
                backgroundColor="#e0e7ef"
                mimeType="audio/wav"
                echoCancellation
                channelCount={1}
                sampleRate={16000}
              />
              {recordedWav && (
                <audio controls src={URL.createObjectURL(recordedWav)} className="ml-2" />
              )}
              <span className="text-gray-400 text-sm ml-2">or upload:</span>
              <input
                type="file"
                accept="audio/*"
                onChange={e => {
                  setFile(e.target.files[0]);
                  setRecordedWav(null);
                }}
                className="w-auto"
              />
              {file && <span className="text-xs text-gray-600 ml-2">{file.name}</span>}
            </div>
          </>
        ) : (
          <textarea
            rows={3}
            value={input}
            onChange={e => setInput(e.target.value)}
            className="w-full p-2 border rounded mt-2"
            placeholder={
              language === 'korean' 
                ? "메시지를 입력하세요..." 
                : "Type a message…"
            }
          />
        )}

        <button
          onClick={handleSubmit}
          disabled={loading}
          className="w-full bg-blue-600 text-white py-2 rounded mt-2 hover:bg-blue-700 disabled:opacity-50"
        >
          {loading ? (language === 'korean' ? '처리 중...' : 'Processing…') : (language === 'korean' ? '전송' : 'Send')}
        </button>
      </footer>
    </div>
  );
}