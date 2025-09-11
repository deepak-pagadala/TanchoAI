import React, { useState, useRef } from 'react';
import axios from 'axios';
import { ReactMic } from 'react-mic';

const API_BASE = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8080';
const TEST_UID = 'demo';                          

/* === Google-Sheets webhook === */
const LOG_URL = "https://script.google.com/macros/s/AKfycbxBZ1hcCm2q5G7AdsrwErQe93ugrQoi4KMRx53jOe4jeAPHljAj11BVojzZEQHeYkc/exec";

/** Enhanced logging - compatible with existing Google Sheets format */
async function logTurn(mode, subMode, uid, payload) {
  const logData = {
    sheet: mode,               // conversation | mentor | voice | dictionary | sentence_analysis | conjugation
    uid,
    sub: subMode || '',        // casual / formal or ''
    payload: payload           // Keep original payload structure
  };

  console.log('📊 Logging to Excel:', logData);

  try {
    const response = await fetch(LOG_URL, {
      method: 'POST',
      mode: 'no-cors', // Important for Google Apps Script
      headers: { 
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(logData)
    });
    
    console.log('✅ Log sent successfully');
    return true;
  } catch (error) {
    console.error('❌ Logging failed:', error);
    
    // Fallback: Store in localStorage for manual export
    const fallbackKey = `tancho_logs_${Date.now()}`;
    const enhancedLogData = {
      timestamp: new Date().toISOString(),
      mode,
      language: payload.language || 'japanese',
      subMode: subMode || '',
      userInput: payload.userInput || '',
      aiResponse: payload.reply || payload.answer || payload.jp || '',
      uid,
      metadata: JSON.stringify(payload)
    };
    localStorage.setItem(fallbackKey, JSON.stringify(enhancedLogData));
    console.log('💾 Stored in localStorage as fallback:', fallbackKey);
    
    return false;
  }
}

/* ================= CSV Export Fallback ================= */
function exportLogsToCSV() {
  const logs = [];
  
  // Get all tancho logs from localStorage
  for (let i = 0; i < localStorage.length; i++) {
    const key = localStorage.key(i);
    if (key && key.startsWith('tancho_logs_')) {
      try {
        const logData = JSON.parse(localStorage.getItem(key));
        logs.push(logData);
      } catch (e) {
        console.error('Failed to parse log:', key, e);
      }
    }
  }
  
  if (logs.length === 0) {
    alert('No logs found to export');
    return;
  }
  
  // Sort by timestamp
  logs.sort((a, b) => new Date(a.timestamp) - new Date(b.timestamp));
  
  // Create CSV content matching your Google Sheets format
  const headers = ['Timestamp', 'UID', 'Mode', 'SubMode', 'User Input', 'AI Response', 'Language', 'Metadata'];
  const csvContent = [
    headers.join(','),
    ...logs.map(log => {
      return [
        log.timestamp,
        log.uid,
        log.mode,
        log.subMode,
        `"${(log.userInput || '').replace(/"/g, '""')}"`,
        `"${(log.aiResponse || '').replace(/"/g, '""')}"`,
        log.language,
        `"${(log.metadata || '').replace(/"/g, '""')}"`,
      ].join(',');
    })
  ].join('\n');
  
  // Download CSV
  const blob = new Blob([csvContent], { type: 'text/csv' });
  const url = window.URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `tancho_logs_${new Date().toISOString().slice(0, 10)}.csv`;
  a.click();
  window.URL.revokeObjectURL(url);
  
  // Clear localStorage after export
  const keysToRemove = [];
  for (let i = 0; i < localStorage.length; i++) {
    const key = localStorage.key(i);
    if (key && key.startsWith('tancho_logs_')) {
      keysToRemove.push(key);
    }
  }
  keysToRemove.forEach(key => localStorage.removeItem(key));
  
  alert(`Exported ${logs.length} log entries to CSV`);
}

/* ================================================================= */

function formatCorrection(wrong, fix, explanation) {
  const strip = txt => txt?.replace(/<[^>]+>/g, '') ?? '';
  const explain = strip(explanation);
  if (!wrong && !fix && (!explain || explain.toLowerCase() === 'en: no errors found.' || explain.includes('오류가 발견되지 않았습니다')))
    return null;

  return (
    <div className="bg-gray-100 border-l-4 border-yellow-400 p-3 mt-2 mb-1 text-sm">
      <div className="mb-1 font-semibold">🔍 Correction</div>
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

/* === Dictionary Result Component === */
function DictionaryResult({ result }) {
  if (!result.found) {
    return (
      <div className="bg-red-50 border border-red-200 p-3 rounded mt-2">
        <div className="text-red-800 font-medium">Not Found</div>
        <div className="text-red-600 text-sm">{result.error}</div>
      </div>
    );
  }

  return (
    <div className="bg-blue-50 border border-blue-200 p-4 rounded mt-2">
      <div className="mb-2">
        <span className="text-xl font-bold">{result.word}</span>
        {result.reading && (
          <span className="ml-2 text-gray-600">({result.reading})</span>
        )}
        {result.level && (
          <span className="ml-2 px-2 py-1 bg-blue-100 text-blue-800 text-xs rounded">
            {result.level}
          </span>
        )}
      </div>
      
      {result.part_of_speech && (
        <div className="text-sm text-gray-600 mb-2">
          <strong>Part of Speech:</strong> {result.part_of_speech}
        </div>
      )}
      
      <div className="mb-3">
        <strong className="text-sm">Meanings:</strong>
        <ul className="list-disc list-inside text-sm mt-1">
          {result.meanings.map((meaning, i) => (
            <li key={i}>{meaning}</li>
          ))}
        </ul>
      </div>

      {/* Kanji Breakdown (Japanese) */}
      {result.kanji_breakdown && Object.keys(result.kanji_breakdown).length > 0 && (
        <div className="mb-3">
          <strong className="text-sm">Kanji Breakdown:</strong>
          <div className="mt-1 grid grid-cols-1 gap-2">
            {Object.entries(result.kanji_breakdown).map(([kanji, info], i) => (
              <div key={i} className="text-sm bg-white p-2 rounded border-l-4 border-orange-400">
                <div className="flex items-center justify-between">
                  <span className="text-lg font-bold text-orange-700">{kanji}</span>
                  <span className="text-xs text-gray-500">
                    {info.strokes ? `${info.strokes} strokes` : ''}
                  </span>
                </div>
                <div className="text-gray-700">
                  <strong>Reading:</strong> {info.reading}
                </div>
                <div className="text-gray-700">
                  <strong>Meaning:</strong> {info.meaning}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Hangul Breakdown (Korean) */}
      {result.hangul_breakdown && Object.keys(result.hangul_breakdown).length > 0 && (
        <div className="mb-3">
          <strong className="text-sm">Character Breakdown:</strong>
          <div className="mt-1 grid grid-cols-1 gap-2">
            {Object.entries(result.hangul_breakdown).map(([char, info], i) => (
              <div key={i} className="text-sm bg-white p-2 rounded border-l-4 border-green-400">
                <div className="flex items-center justify-between">
                  <span className="text-lg font-bold text-green-700">{char}</span>
                  {info.hanja && (
                    <span className="text-sm text-gray-600">Hanja: {info.hanja}</span>
                  )}
                </div>
                <div className="text-gray-700">
                  <strong>Pronunciation:</strong> {info.pronunciation}
                </div>
                <div className="text-gray-700">
                  <strong>Meaning:</strong> {info.meaning}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {result.example_sentences && result.example_sentences.length > 0 && (
        <div className="mb-3">
          <strong className="text-sm">Examples:</strong>
          <div className="mt-1 space-y-2">
            {result.example_sentences.map((ex, i) => (
              <div key={i} className="text-sm bg-white p-2 rounded">
                <div className="font-medium">{ex.japanese || ex.korean}</div>
                <div className="text-gray-600">{ex.english}</div>
                {(ex.reading || ex.romanization) && (
                  <div className="text-xs text-gray-500">{ex.reading || ex.romanization}</div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {result.conjugations && result.conjugations.length > 0 && (
        <div>
          <strong className="text-sm">Conjugations:</strong>
          <div className="mt-1 grid grid-cols-2 gap-2">
            {result.conjugations.map((conj, i) => (
              <div key={i} className="text-sm bg-white p-2 rounded">
                <div className="font-medium text-xs text-gray-500">{conj.form}</div>
                <div>{conj.japanese || conj.korean}</div>
                {(conj.reading || conj.romanization) && (
                  <div className="text-xs text-gray-400">{conj.reading || conj.romanization}</div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

/* === Enhanced Sentence Analysis Result Component === */
function SentenceAnalysisResult({ result }) {
  if (!result.found) {
    return (
      <div className="bg-red-50 border border-red-200 p-3 rounded mt-2">
        <div className="text-red-800 font-medium">Analysis Failed</div>
        <div className="text-red-600 text-sm">{result.error}</div>
      </div>
    );
  }

  const ScoreBar = ({ label, score, color = 'blue' }) => (
    <div className="mb-2">
      <div className="flex justify-between text-sm">
        <span>{label}</span>
        <span>{score}%</span>
      </div>
      <div className="w-full bg-gray-200 rounded-full h-2">
        <div 
          className={`bg-${color}-500 h-2 rounded-full`} 
          style={{ width: `${score}%` }}
        />
      </div>
    </div>
  );

  return (
    <div className="bg-green-50 border border-green-200 p-4 rounded mt-2">
      <div className="mb-4">
        <div className="text-lg font-bold mb-2">
          Overall Score: {result.correctnessScore}%
        </div>
        
        <div className="grid grid-cols-2 gap-4">
          <div>
            <ScoreBar label="Grammar" score={result.grammarBreakdown?.grammar || 0} />
            <ScoreBar label="Particles" score={result.grammarBreakdown?.particles || 0} />
            <ScoreBar label="Word Usage" score={result.grammarBreakdown?.wordUsage || 0} />
          </div>
          <div>
            <ScoreBar label="Spelling" score={result.grammarBreakdown?.spelling || 0} />
            {result.grammarBreakdown?.kanjiUsage && (
              <ScoreBar label="Kanji Usage" score={result.grammarBreakdown.kanjiUsage} />
            )}
            {result.grammarBreakdown?.honorifics && (
              <ScoreBar label="Honorifics" score={result.grammarBreakdown.honorifics} />
            )}
          </div>
        </div>
      </div>

      <div className="space-y-3">
        <div>
          <strong className="text-sm">Your sentence:</strong>
          <div className="bg-white p-2 rounded text-sm">{result.originalSentence}</div>
        </div>

        <div>
          <strong className="text-sm">What you said:</strong>
          <div className="bg-white p-2 rounded text-sm">{result.userTranslation}</div>
        </div>

        {result.correctedSentence && result.correctedSentence !== result.originalSentence && (
          <>
            <div>
              <strong className="text-sm">Corrected version:</strong>
              <div className="bg-green-100 p-2 rounded text-sm">{result.correctedSentence}</div>
            </div>

            <div>
              <strong className="text-sm">Corrected meaning:</strong>
              <div className="bg-green-100 p-2 rounded text-sm">{result.correctedTranslation}</div>
            </div>
          </>
        )}

        {result.improvements && result.improvements.length > 0 && (
          <div>
            <strong className="text-sm">Improvements:</strong>
            <div className="mt-1 space-y-2">
              {result.improvements.map((imp, i) => (
                <div key={i} className="bg-yellow-50 border-l-4 border-yellow-400 p-2 text-sm">
                  <div className="font-medium">{imp.type}</div>
                  <div className="text-gray-700">{imp.explanation}</div>
                  {imp.original !== imp.corrected && (
                    <div className="mt-1">
                      <span className="text-red-600">"{imp.original}"</span>
                      <span className="mx-2">→</span>
                      <span className="text-green-600">"{imp.corrected}"</span>
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Word Analysis */}
        {result.wordAnalysis && result.wordAnalysis.length > 0 && (
          <div>
            <strong className="text-sm">Word-by-word Analysis:</strong>
            <div className="mt-1 space-y-1">
              {result.wordAnalysis.map((word, i) => (
                <div key={i} className={`text-xs p-2 rounded ${word.isCorrect ? 'bg-green-50' : 'bg-red-50'}`}>
                  <div className="flex justify-between items-center">
                    <span className="font-medium">{word.word}</span>
                    <span className={word.isCorrect ? 'text-green-600' : 'text-red-600'}>
                      {word.isCorrect ? '✓' : '✗'}
                    </span>
                  </div>
                  {word.reading && (
                    <div className="text-gray-500">{word.reading}</div>
                  )}
                  <div className="text-gray-600">{word.partOfSpeech} - {word.meaning}</div>
                  {word.usage && (
                    <div className="text-gray-700 italic">{word.usage}</div>
                  )}
                  {word.correction && (
                    <div className="text-green-700">→ {word.correction}</div>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

/* === NEW: Conjugation Result Component === */
function ConjugationResult({ result }) {
  if (!result.found) {
    return (
      <div className="bg-red-50 border border-red-200 p-3 rounded mt-2">
        <div className="text-red-800 font-medium">Conjugation Analysis Failed</div>
        <div className="text-red-600 text-sm">{result.error}</div>
      </div>
    );
  }

  const { verbInfo, conjugations } = result;

  return (
    <div className="bg-purple-50 border border-purple-200 p-4 rounded mt-2">
      {/* Verb Info Header */}
      <div className="mb-4 bg-white p-3 rounded">
        <div className="text-lg font-bold">{verbInfo.baseForm}</div>
        {verbInfo.reading && (
          <div className="text-gray-600">({verbInfo.reading})</div>
        )}
        {verbInfo.romanization && (
          <div className="text-gray-600">{verbInfo.romanization}</div>
        )}
        <div className="text-sm text-gray-700 mt-1">{verbInfo.meaning}</div>
        
        <div className="flex gap-2 mt-2">
          <span className="px-2 py-1 bg-purple-100 text-purple-800 text-xs rounded">
            {verbInfo.verbType}
          </span>
          {verbInfo.conjugationGroup && (
            <span className="px-2 py-1 bg-gray-100 text-gray-700 text-xs rounded">
              {verbInfo.conjugationGroup}
            </span>
          )}
          {verbInfo.isConjugated && (
            <span className="px-2 py-1 bg-orange-100 text-orange-800 text-xs rounded">
              Conjugated Form
            </span>
          )}
        </div>

        {verbInfo.isConjugated && (
          <div className="mt-2 text-sm">
            <div><strong>Your input:</strong> {verbInfo.originalInput}</div>
            <div><strong>Input meaning:</strong> {verbInfo.originalInput === verbInfo.baseForm ? verbInfo.meaning : verbInfo.originalMeaning || verbInfo.meaning}</div>
          </div>
        )}
      </div>

      {/* Conjugation Categories */}
      <div className="space-y-4">
        {Object.entries(conjugations).map(([category, forms]) => (
          <div key={category} className="bg-white p-3 rounded">
            <h3 className="font-semibold text-purple-800 mb-2">{category}</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
              {forms.map((form, i) => (
                <div key={i} className="border border-gray-200 p-2 rounded text-sm">
                  <div className="font-medium text-purple-700">{form.formName}</div>
                  <div className="text-lg">{form.conjugated}</div>
                  {(form.reading || form.romanization) && (
                    <div className="text-gray-500 text-xs">
                      {form.reading || form.romanization}
                    </div>
                  )}
                  {form.explanation && (
                    <div className="text-gray-600 text-xs mt-1">{form.explanation}</div>
                  )}
                  {form.usage && (
                    <div className="text-gray-500 text-xs italic">{form.usage}</div>
                  )}
                  {form.politenessLevel && (
                    <span className="inline-block mt-1 px-1 py-0.5 bg-blue-100 text-blue-700 text-xs rounded">
                      {form.politenessLevel}
                    </span>
                  )}
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

/* ================================================================= */

export default function App() {
  const [mode, setMode] = useState('conversation');
  const [convType, setConvType] = useState('convCasual'); // convCasual / convFormal
  const [language, setLanguage] = useState('japanese');   // japanese / korean
  const [input, setInput] = useState('');
  const [intendedMeaning, setIntendedMeaning] = useState(''); // NEW: for enhanced sentence analysis
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
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
        : val === 'dictionary'
        ? 'Dictionary Lookup'
        : val === 'sentence_analysis'
        ? 'Sentence Analysis'
        : val === 'conjugation'
        ? 'Conjugation Analysis'
        : val.charAt(0).toUpperCase() + val.slice(1);
    pushMsg('system', `——— Switched to ${label} Mode (${language}) ———`);
    setFile(null); setRecordedWav(null); setIsRecording(false); setIntendedMeaning('');
  };

  const handleLanguageChange = val => {
    setLanguage(val);
    pushMsg('system', `——— Language changed to ${val.charAt(0).toUpperCase() + val.slice(1)} ———`);
  };

  /* ========= Submit ========= */
  async function handleSubmit() {
    if (mode !== 'voice' && !input) return;
    if (mode === 'voice' && !file && !recordedWav)
      return alert('Please record or select a file!');

    const userInput = mode !== 'voice' ? input : 'Voice input';
    if (mode !== 'voice') pushMsg('user', input);
    setLoading(true);

    try {
      /* === Conversation === */
      if (mode === 'conversation') {
        const { data } = await axios.post(`${API_BASE}/chat`, {
          uid: TEST_UID,
          mode: convType,
          userMessage: input,
          language: language
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
          `${convType}_${language}`,
          TEST_UID,
          { wrong, fix, reply, explanation, chatTitle, language, userInput: input }
        );
      }

      /* === Dictionary === */
      else if (mode === 'dictionary') {
        const { data } = await axios.post(`${API_BASE}/dictionary`, {
          uid: TEST_UID,
          word: input,
          language: language
        });

        pushMsg(
          'assistant',
          <DictionaryResult result={data} />
        );

        await logTurn(
          'dictionary',
          language,
          TEST_UID,
          { word: input, result: data, language, userInput: input }
        );
      }

      /* === NEW: Conjugation Analysis === */
      else if (mode === 'conjugation') {
        const { data } = await axios.post(`${API_BASE}/conjugation`, {
          uid: TEST_UID,
          word: input,
          language: language
        });

        pushMsg(
          'assistant',
          <ConjugationResult result={data} />
        );

        await logTurn(
          'conjugation',
          language,
          TEST_UID,
          { word: input, result: data, language, userInput: input }
        );
      }

      /* === Enhanced Sentence Analysis === */
      else if (mode === 'sentence_analysis') {
        const endpoint = intendedMeaning.trim() 
          ? '/sentence_analysis_enhanced'  // Use enhanced version if intended meaning provided
          : '/sentence_analysis';

        const requestBody = {
          uid: TEST_UID,
          sentence: input,
          language: language
        };

        // Add intended meaning for enhanced analysis
        if (intendedMeaning.trim()) {
          requestBody.intended_english_meaning = intendedMeaning.trim();
          requestBody.analysis_type = 'enhanced_with_context';
        }

        const { data } = await axios.post(`${API_BASE}${endpoint}`, requestBody);

        pushMsg(
          'assistant',
          <SentenceAnalysisResult result={data} />
        );

        await logTurn(
          'sentence_analysis',
          `${language}${intendedMeaning.trim() ? '_enhanced' : ''}`,
          TEST_UID,
          { 
            sentence: input, 
            intended_meaning: intendedMeaning.trim() || undefined,
            result: data, 
            language, 
            userInput: input 
          }
        );
      }

      /* === Mentor (stream) === */
      else if (mode === 'mentor') {
        pushMsg('assistant', ''); // placeholder
        const mentorResponse = await doMentorStream(input);
        setLoading(false);
        setInput('');
        return;
      }

      /* === Voice === */
      else {
        const audioBlob = recordedWav || file;
        const form = new FormData();
        form.append('uid', TEST_UID);
        form.append('voiceMode', 'conversation');
        form.append('convSubMode', convType);
        form.append('language', language);
        form.append('audio', audioBlob,
          audioBlob instanceof File ? audioBlob.name : 'audio.wav');

        const { data } = await axios.post(`${API_BASE}/voice_chat`, form);
        const { jp, en, correction, transcript, ttsUrl, chatTitle } = data;

        /* user bubble */
        pushMsg('user', transcript || en || jp);

        /* assistant bubble */
        const aiResponse = jp + (en ? `\n${en}` : '');
        pushMsg(
          'assistant',
          <>
            <div>{aiResponse}</div>
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

        await logTurn(
          'voice',
          `${language}`,
          TEST_UID,
          { jp, en, correction, chatTitle, ttsUrl, language, userInput: transcript || 'Voice input', transcript }
        );
        
        setFile(null); setRecordedWav(null);
      }
    } catch (err) {
      console.error(err);
      pushMsg('assistant', `❌ Error: ${err.response?.data?.detail || err.message || err}`);
    } finally {
      setLoading(false);
      setInput('');
      setIntendedMeaning('');
    }
  }

  /* ========= Mentor stream helper ========= */
  async function doMentorStream(question) {
    let finalAnswer = '';
    try {
      const res = await fetch(`${API_BASE}/mentor`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          uid: TEST_UID, 
          question,
          language: language
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
                finalAnswer = answer || streamedContent;
                if (chatTitle) {
                  finalAnswer += `\n\n💡 Chat title: ${chatTitle}`;
                }
                
                updateLastAssistantMsg(() => finalAnswer);
                
                await logTurn(
                  'mentor',
                  `${language}`,
                  TEST_UID,
                  { answer, recommendation, chatTitle, language, userInput: question }
                );
              } catch (parseErr) {
                console.error('Failed to parse final JSON:', parseErr);
                finalAnswer = streamedContent;
              }
            }
            break;
          } else if (block.startsWith('data:')) {
            const token = block.replace(/^data:\s*/, '');
            
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
    
    return finalAnswer;
  }

  /* ========= UI ========= */
  return (
    <div className="flex flex-col h-screen bg-gray-100">
      <header className="p-4 bg-blue-600 text-white">
        <div className="flex justify-between items-center">
          <div className="text-xl font-semibold">
            Tancho AI Tester - {language === 'korean' ? '한국어' : '日本語'}
          </div>
          <button
            onClick={exportLogsToCSV}
            className="px-3 py-1 bg-green-500 text-white text-sm rounded hover:bg-green-600"
            title="Export conversation logs to CSV"
          >
            📊 Export CSV
          </button>
        </div>
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
            <option value="conversation">💬 Conversation</option>
            <option value="mentor">🎓 Mentor</option>
            <option value="voice">🎤 Voice</option>
            <option value="dictionary">📚 Dictionary</option>
            <option value="sentence_analysis">🔍 Sentence Analysis</option>
            <option value="conjugation">🔄 Conjugation</option>
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

          <select
            value={language}
            onChange={e => handleLanguageChange(e.target.value)}
            className="p-2 border rounded bg-green-50"
          >
            <option value="japanese">🇯🇵 Japanese</option>
            <option value="korean">🇰🇷 Korean</option>
          </select>
        </div>

        {/* Enhanced input area for sentence analysis */}
        {mode === 'sentence_analysis' && (
          <div className="bg-blue-50 p-3 rounded border">
            <div className="text-sm font-medium text-blue-800 mb-2">
              Enhanced Analysis (Optional)
            </div>
            <textarea
              rows={2}
              value={intendedMeaning}
              onChange={e => setIntendedMeaning(e.target.value)}
              className="w-full p-2 border rounded text-sm"
              placeholder="What did you intend to say in English? (This helps provide better corrections)"
            />
            <div className="text-xs text-blue-600 mt-1">
              💡 Providing your intended meaning enables enhanced contextual analysis
            </div>
          </div>
        )}

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
              mode === 'dictionary' 
                ? (language === 'korean' ? "단어를 입력하세요..." : "単語を入力してください...")
                : mode === 'sentence_analysis'
                ? (language === 'korean' ? "분석할 문장을 입력하세요..." : "分析する文章を入力してください...")
                : mode === 'conjugation'
                ? (language === 'korean' ? "활용할 동사/형용사를 입력하세요..." : "活用する動詞・形容詞を入力してください...")
                : language === 'korean' 
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
          {loading ? (language === 'korean' ? '처리 중...' : 'Processing…') : 
           mode === 'dictionary' ? (language === 'korean' ? '검색' : 'Look up') :
           mode === 'sentence_analysis' ? (language === 'korean' ? '분석' : 'Analyze') :
           mode === 'conjugation' ? (language === 'korean' ? '활용 분석' : 'Conjugate') :
           (language === 'korean' ? '전송' : 'Send')}
        </button>
      </footer>
    </div>
  );
}