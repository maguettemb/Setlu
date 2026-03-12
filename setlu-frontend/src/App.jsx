import { useState, useRef, useEffect, useCallback } from "react";

const API_URL = "https://setlu-agent.up.railway.app" 
const PROFILES = ["Recruiter", "Technical Hiring Manager", "General"];

const PROFILE_DESCRIPTIONS = {
  Recruiter: "Professional overview & experience",
  "Technical Hiring Manager": "Deep technical skills & architecture",
  General: "Background, interests & contact",
};

const WelcomeMessages = {
  Recruiter: "Hello! I can tell you about Maguette's professional experience, skills, and achievements. What would you like to know?",
  "Technical Hiring Manager": "Hi! I can dive deep into Maguette's technical expertise, architecture decisions, and engineering approach. What are you curious about?",
  General: "Hello! I'm here to share information about Maguette MBAYE — her background, interests, and how to reach her. Ask me anything!",
};

const SendIcon = () => (
  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <line x1="22" y1="2" x2="11" y2="13" />
    <polygon points="22 2 15 22 11 13 2 9 22 2" />
  </svg>
);

const ThinkingDots = () => (
  <div style={{ display: "flex", gap: "4px", alignItems: "center", padding: "4px 0" }}>
    {[0, 1, 2].map((i) => (
      <div key={i} style={{
        width: "6px", height: "6px", borderRadius: "50%",
        background: "#94a3b8",
        animation: "bounce 1.2s ease-in-out infinite",
        animationDelay: `${i * 0.2}s`,
      }} />
    ))}
  </div>
);

export default function App() {
  const [profile, setProfile] = useState("Recruiter");
  const [messages, setMessages] = useState([
    { role: "assistant", content: WelcomeMessages["Recruiter"], id: 0 },
  ]);
  const [input, setInput] = useState("");
  const [isStreaming, setIsStreaming] = useState(false);
  const [sessionId] = useState(() => `session_${Date.now()}`);
  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);
  const abortRef = useRef(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const handleProfileChange = (newProfile) => {
    setProfile(newProfile);
    setMessages([{ role: "assistant", content: WelcomeMessages[newProfile], id: Date.now() }]);
    setInput("");
    inputRef.current?.focus();
  };

  const sendMessage = useCallback(async () => {
    const text = input.trim();
    if (!text || isStreaming) return;

    const userMsg = { role: "user", content: text, id: Date.now() };
    const assistantId = Date.now() + 1;
    const assistantMsg = { role: "assistant", content: "", id: assistantId, streaming: true };

    setMessages((prev) => [...prev, userMsg, assistantMsg]);
    setInput("");
    setIsStreaming(true);

    try {
      const res = await fetch(`${API_URL}/chat/stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ 
    message: text,        // ← "message" pas "query" ou "input"
    profile,              // ← "profile" 
    session_id: sessionId // ← "session_id"
}),
      });

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n\n");
        buffer = lines.pop();

        for (const line of lines) {
          if (!line.startsWith("data: ")) continue;
          try {
            const data = JSON.parse(line.slice(6));
            if (data.error) throw new Error(data.error);
            if (data.done) break;
            setMessages((prev) =>
              prev.map((m) =>
                m.id === assistantId
                  ? { ...m, content: m.content + data.token }
                  : m
              )
            );
          } catch {}
        }
      }
    } catch (err) {
      setMessages((prev) =>
        prev.map((m) =>
          m.id === assistantId
            ? { ...m, content: "Sorry, something went wrong. Please try again." }
            : m
        )
      );
    } finally {
      setMessages((prev) =>
        prev.map((m) => (m.id === assistantId ? { ...m, streaming: false } : m))
      );
      setIsStreaming(false);
      inputRef.current?.focus();
    }
  }, [input, isStreaming, profile, sessionId]);

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  return (
    <>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500&display=swap');

        *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

        :root {
          --bg: #f8f7f4;
          --surface: #ffffff;
          --border: #e8e5df;
          --text-primary: #1a1917;
          --text-secondary: #6b6860;
          --text-muted: #a09d96;
          --accent: #1a1917;
          --accent-light: #f0ede8;
          --user-bg: #1a1917;
          --user-text: #f8f7f4;
          --assistant-bg: #ffffff;
          --shadow-sm: 0 1px 3px rgba(0,0,0,0.06);
          --shadow-md: 0 4px 16px rgba(0,0,0,0.08);
          --radius: 16px;
          --radius-sm: 8px;
        }

        body {
          font-family: 'DM Sans', sans-serif;
          background: var(--bg);
          color: var(--text-primary);
          height: 100vh;
          overflow: hidden;
        }

        @keyframes bounce {
          0%, 80%, 100% { transform: translateY(0); opacity: 0.4; }
          40% { transform: translateY(-6px); opacity: 1; }
        }

        @keyframes fadeSlideUp {
          from { opacity: 0; transform: translateY(10px); }
          to   { opacity: 1; transform: translateY(0); }
        }

        @keyframes fadeIn {
          from { opacity: 0; }
          to   { opacity: 1; }
        }

        .app {
          display: grid;
          grid-template-columns: 280px 1fr;
          height: 100vh;
        }

        /* ── Sidebar ── */
        .sidebar {
          background: var(--surface);
          border-right: 1px solid var(--border);
          display: flex;
          flex-direction: column;
          padding: 32px 24px;
          gap: 32px;
        }

        .sidebar-header {}

        .logo {
          font-family: 'DM Serif Display', serif;
          font-size: 22px;
          color: var(--text-primary);
          letter-spacing: -0.3px;
          line-height: 1.2;
          margin-bottom: 4px;
        }

        .logo span {
          font-style: italic;
          color: var(--text-secondary);
        }

        .logo-sub {
          font-size: 12px;
          color: var(--text-muted);
          font-weight: 300;
          letter-spacing: 0.5px;
          text-transform: uppercase;
        }

        .divider {
          height: 1px;
          background: var(--border);
        }

        .profile-section {}

        .section-label {
          font-size: 10px;
          font-weight: 500;
          color: var(--text-muted);
          text-transform: uppercase;
          letter-spacing: 1px;
          margin-bottom: 12px;
        }

        .profile-list {
          display: flex;
          flex-direction: column;
          gap: 4px;
        }

        .profile-btn {
          display: flex;
          flex-direction: column;
          align-items: flex-start;
          padding: 12px 14px;
          border-radius: var(--radius-sm);
          border: 1px solid transparent;
          background: transparent;
          cursor: pointer;
          transition: all 0.15s ease;
          text-align: left;
          width: 100%;
        }

        .profile-btn:hover {
          background: var(--accent-light);
          border-color: var(--border);
        }

        .profile-btn.active {
          background: var(--accent);
          border-color: var(--accent);
        }

        .profile-btn-name {
          font-size: 13px;
          font-weight: 500;
          color: var(--text-primary);
          transition: color 0.15s;
        }

        .profile-btn.active .profile-btn-name { color: var(--user-text); }

        .profile-btn-desc {
          font-size: 11px;
          color: var(--text-muted);
          margin-top: 2px;
          transition: color 0.15s;
        }

        .profile-btn.active .profile-btn-desc { color: rgba(248,247,244,0.6); }

        .sidebar-footer {
          margin-top: auto;
          font-size: 11px;
          color: var(--text-muted);
          line-height: 1.6;
        }

        .sidebar-footer strong {
          color: var(--text-secondary);
          font-weight: 500;
          display: block;
          margin-bottom: 2px;
        }

        /* ── Chat area ── */
        .chat-area {
          display: flex;
          flex-direction: column;
          height: 100vh;
          overflow: hidden;
        }

        .chat-header {
          padding: 20px 32px;
          border-bottom: 1px solid var(--border);
          display: flex;
          align-items: center;
          justify-content: space-between;
          background: var(--surface);
          flex-shrink: 0;
        }

        .chat-header-title {
          font-family: 'DM Serif Display', serif;
          font-size: 18px;
          color: var(--text-primary);
        }

        .chat-header-badge {
          font-size: 11px;
          font-weight: 500;
          padding: 4px 10px;
          border-radius: 20px;
          background: var(--accent-light);
          color: var(--text-secondary);
          border: 1px solid var(--border);
        }

        .messages-container {
          flex: 1;
          overflow-y: auto;
          padding: 32px;
          display: flex;
          flex-direction: column;
          gap: 24px;
          scroll-behavior: smooth;
        }

        .messages-container::-webkit-scrollbar { width: 4px; }
        .messages-container::-webkit-scrollbar-track { background: transparent; }
        .messages-container::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }

        .message {
          display: flex;
          gap: 12px;
          max-width: 75%;
          animation: fadeSlideUp 0.25s ease forwards;
        }

        .message.user {
          align-self: flex-end;
          flex-direction: row-reverse;
        }

        .message.assistant {
          align-self: flex-start;
        }

        .avatar {
          width: 32px;
          height: 32px;
          border-radius: 50%;
          flex-shrink: 0;
          display: flex;
          align-items: center;
          justify-content: center;
          font-size: 12px;
          font-weight: 500;
          margin-top: 2px;
        }

        .avatar.user-avatar {
          background: var(--accent);
          color: var(--user-text);
        }

        .avatar.assistant-avatar {
          background: var(--accent-light);
          color: var(--text-secondary);
          border: 1px solid var(--border);
          font-family: 'DM Serif Display', serif;
          font-style: italic;
          font-size: 14px;
        }

        .bubble {
          padding: 14px 18px;
          border-radius: var(--radius);
          line-height: 1.65;
          font-size: 14px;
          box-shadow: var(--shadow-sm);
        }

        .message.user .bubble {
          background: var(--user-bg);
          color: var(--user-text);
          border-bottom-right-radius: 4px;
        }

        .message.assistant .bubble {
          background: var(--assistant-bg);
          color: var(--text-primary);
          border: 1px solid var(--border);
          border-bottom-left-radius: 4px;
        }

        .cursor {
          display: inline-block;
          width: 2px;
          height: 14px;
          background: var(--text-secondary);
          margin-left: 2px;
          vertical-align: middle;
          animation: blink 0.8s step-end infinite;
        }

        @keyframes blink {
          0%, 100% { opacity: 1; }
          50% { opacity: 0; }
        }

        /* ── Input area ── */
        .input-area {
          padding: 20px 32px 28px;
          background: var(--surface);
          border-top: 1px solid var(--border);
          flex-shrink: 0;
        }

        .input-wrapper {
          display: flex;
          align-items: flex-end;
          gap: 10px;
          background: var(--bg);
          border: 1px solid var(--border);
          border-radius: var(--radius);
          padding: 12px 14px;
          transition: border-color 0.15s, box-shadow 0.15s;
        }

        .input-wrapper:focus-within {
          border-color: #c0bdb6;
          box-shadow: 0 0 0 3px rgba(26,25,23,0.04);
        }

        .input-field {
          flex: 1;
          border: none;
          background: transparent;
          font-family: 'DM Sans', sans-serif;
          font-size: 14px;
          color: var(--text-primary);
          resize: none;
          outline: none;
          line-height: 1.5;
          max-height: 120px;
          min-height: 22px;
        }

        .input-field::placeholder { color: var(--text-muted); }

        .send-btn {
          width: 34px;
          height: 34px;
          border-radius: 8px;
          border: none;
          background: var(--accent);
          color: var(--user-text);
          cursor: pointer;
          display: flex;
          align-items: center;
          justify-content: center;
          flex-shrink: 0;
          transition: opacity 0.15s, transform 0.1s;
        }

        .send-btn:hover:not(:disabled) { opacity: 0.85; transform: scale(1.03); }
        .send-btn:active:not(:disabled) { transform: scale(0.97); }
        .send-btn:disabled { opacity: 0.35; cursor: not-allowed; }

        .input-hint {
          font-size: 11px;
          color: var(--text-muted);
          margin-top: 8px;
          text-align: center;
        }

        @media (max-width: 768px) {
          .app { grid-template-columns: 1fr; grid-template-rows: auto 1fr; }
          .sidebar {
            flex-direction: row;
            padding: 16px;
            gap: 16px;
            overflow-x: auto;
            border-right: none;
            border-bottom: 1px solid var(--border);
          }
          .sidebar-footer { display: none; }
          .profile-list { flex-direction: row; }
          .messages-container { padding: 20px 16px; }
          .input-area { padding: 16px; }
          .message { max-width: 90%; }
        }
      `}</style>

      <div className="app">
        {/* Sidebar */}
        <aside className="sidebar">
          <div className="sidebar-header">
            <div className="logo">
              Maguette<br /><span>MBAYE</span>
            </div>
            <div className="logo-sub">AI Portfolio Agent</div>
          </div>

          <div className="divider" />

          <div className="profile-section">
            <div className="section-label">You are a</div>
            <div className="profile-list">
              {PROFILES.map((p) => (
                <button
                  key={p}
                  className={`profile-btn ${profile === p ? "active" : ""}`}
                  onClick={() => handleProfileChange(p)}
                >
                  <span className="profile-btn-name">{p}</span>
                  <span className="profile-btn-desc">{PROFILE_DESCRIPTIONS[p]}</span>
                </button>
              ))}
            </div>
          </div>

          <div className="divider" />

          <div className="sidebar-footer">
            <strong>Dr. Maguette MBAYE</strong>
            Doctor in AI · Engineer · Mom<br />
            Powered by RAG + LLM-as-a-Judge
          </div>
        </aside>

        {/* Chat */}
        <div className="chat-area">
          <div className="chat-header">
            <span className="chat-header-title">Ask me anything</span>
            <span className="chat-header-badge">{profile}</span>
          </div>

          <div className="messages-container">
            {messages.map((msg) => (
              <div key={msg.id} className={`message ${msg.role}`}>
                <div className={`avatar ${msg.role === "user" ? "user-avatar" : "assistant-avatar"}`}>
                  {msg.role === "user" ? "You" : "M"}
                </div>
                <div className="bubble">
                  {msg.content === "" && msg.streaming ? (
                    <ThinkingDots />
                  ) : (
                    <>
                      {msg.content}
                      {msg.streaming && msg.content !== "" && (
                        <span className="cursor" />
                      )}
                    </>
                  )}
                </div>
              </div>
            ))}
            <div ref={messagesEndRef} />
          </div>

          <div className="input-area">
            <div className="input-wrapper">
              <textarea
                ref={inputRef}
                className="input-field"
                placeholder="Ask about Maguette's experience, skills, or projects…"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={handleKeyDown}
                rows={1}
                disabled={isStreaming}
                onInput={(e) => {
                  e.target.style.height = "auto";
                  e.target.style.height = Math.min(e.target.scrollHeight, 120) + "px";
                }}
              />
              <button
                className="send-btn"
                onClick={sendMessage}
                disabled={!input.trim() || isStreaming}
                aria-label="Send message"
              >
                <SendIcon />
              </button>
            </div>
            <div className="input-hint">Press Enter to send · Shift+Enter for new line</div>
          </div>
        </div>
      </div>
    </>
  );
}
