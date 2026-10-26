import { useEffect, useRef, useState, type FormEvent, type KeyboardEvent } from "react";
import {
  ArrowUp, Check, CircleHelp, Clock3, Copy, CreditCard, Headphones,
  LifeBuoy, MessageCircle, PackageSearch, RotateCcw, ShieldCheck, Sparkles, Truck,
} from "lucide-react";

type RetrievalMode = "hybrid" | "semantic" | "bm25";
type ChatMessage = {
  role: "user" | "assistant";
  content: string;
  destination?: string;
  sources?: string[];
  tool_used?: string | null;
  retrieval_mode?: RetrievalMode | null;
  retrieval_fallback?: boolean;
  escalated?: boolean;
  error?: boolean;
};
type ChatResponse = {
  request_id: string;
  reply: string;
  destination: string;
  sources?: string[];
  tool_used?: string | null;
  retrieval_mode?: RetrievalMode | null;
  retrieval_fallback?: boolean;
  escalated?: boolean;
};

const suggestions = [
  { icon: PackageSearch, label: "Track an order", text: "Where is my order ORD-1001?" },
  { icon: Truck, label: "Shipping times", text: "How long does shipping take?" },
  { icon: RotateCcw, label: "Returns & refunds", text: "What is your return policy?" },
  { icon: CreditCard, label: "Payment help", text: "What is the status of payment pi_MOCK1001?" },
];
const modeLabels: Record<RetrievalMode, string> = {
  hybrid: "Hybrid", semantic: "Semantic", bm25: "BM25 keywords",
};

function App() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [draft, setDraft] = useState("");
  const [mode, setMode] = useState<RetrievalMode>("hybrid");
  const [sessionId] = useState(() => crypto.randomUUID());
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [copied, setCopied] = useState(false);
  const endRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages, busy]);

  async function sendMessage(value: string) {
    const content = value.trim();
    if (!content || busy) return;
    setError("");
    setDraft("");
    setBusy(true);
    const previous = messages;
    setMessages(previous.concat({ role: "user", content }));
    try {
      const response = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: content,
          session_id: sessionId,
          retrieval_mode: mode,
          history: previous.map(({ role, content: text }) => ({ role, content: text })),
        }),
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || "The support service could not process that message.");
      const result = data as ChatResponse;
      setMessages((current) => current.concat({
        role: "assistant",
        content: result.reply,
        destination: result.destination,
        sources: result.sources || [],
        tool_used: result.tool_used,
        retrieval_mode: result.retrieval_mode,
        retrieval_fallback: result.retrieval_fallback,
        escalated: result.escalated,
      }));
    } catch (err) {
      const message = err instanceof Error ? err.message : "Something went wrong. Please try again.";
      setError(message);
      setMessages((current) => current.concat({ role: "assistant", content: message, error: true }));
    } finally {
      setBusy(false);
      inputRef.current?.focus();
    }
  }

  function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    void sendMessage(draft);
  }

  function composerKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      void sendMessage(draft);
    }
  }

  function startNewConversation() {
    if (busy) return;
    setMessages([]);
    setError("");
    setDraft("");
    inputRef.current?.focus();
  }

  async function copyTranscript() {
    const transcript = messages
      .map((item) => (item.role === "user" ? "You" : "SupportDesk") + ": " + item.content)
      .join("\n\n");
    if (!transcript) return;
    try {
      await navigator.clipboard.writeText(transcript);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1600);
    } catch {
      setError("Could not copy this conversation from your browser.");
    }
  }

  const hasMessages = messages.length > 0;
  const promptButton = (item: typeof suggestions[number], compact = false) => {
    const Icon = item.icon;
    return (
      <button className={compact ? "nav-item topic-link" : "help-link"} key={item.label}
        onClick={() => void sendMessage(item.text)} disabled={busy}>
        {compact ? <Icon size={17} /> : <span className="help-link-icon"><Icon size={17} /></span>}
        <span>{item.label}</span>{!compact && <ArrowUp className="rotate-arrow" size={14} />}
      </button>
    );
  };

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <a className="brand" href="#top" aria-label="SupportDesk home">
          <span className="brand-mark"><Headphones size={19} strokeWidth={2.2} /></span>
          <span className="brand-copy"><strong>SupportDesk</strong><small>Customer care</small></span>
        </a>
        <div className="sidebar-section">
          <div className="section-label">YOUR SPACE</div>
          <button className="nav-item active" onClick={startNewConversation}>
            <MessageCircle size={17} /><span>New conversation</span><span className="nav-plus">+</span>
          </button>
        </div>
        <div className="sidebar-section topic-section">
          <div className="section-label">POPULAR TOPICS</div>
          {suggestions.slice(0, 3).map((item) => promptButton(item, true))}
        </div>
        <div className="sidebar-bottom">
          <div className="privacy-card">
            <span className="privacy-icon"><ShieldCheck size={17} /></span>
            <div><strong>Your privacy matters</strong><p>Please don’t share passwords or full card details.</p></div>
          </div>
          <div className="service-status"><span className="status-dot" /> Support is ready</div>
        </div>
      </aside>

      <main className="main-panel" id="top">
        <header className="topbar">
          <div className="topbar-title"><span className="mobile-brand"><Headphones size={17} /></span>
            Support <span className="crumb-slash">/</span> <span className="crumb-current">Conversation</span>
          </div>
          <div className="topbar-actions">
            {hasMessages && <button className="quiet-button" onClick={() => void copyTranscript()}>
              {copied ? <Check size={15} /> : <Copy size={14} />}{copied ? "Copied" : "Copy chat"}
            </button>}
            <button className="new-chat-button" onClick={startNewConversation}><span>+</span> New chat</button>
          </div>
        </header>

        <div className="conversation-wrap">
          {!hasMessages ? (
            <section className="welcome">
              <div className="welcome-mark"><Sparkles size={21} strokeWidth={1.8} /></div>
              <div className="eyebrow"><span className="status-dot" /> HERE TO HELP, EVERY STEP</div>
              <h1>How can we help<br />you today?</h1>
              <p className="welcome-copy">Get a quick answer about your order, delivery, returns, or payment. Tell us what you need.</p>
              <div className="suggestion-grid">
                {suggestions.map((item) => {
                  const Icon = item.icon;
                  return (
                    <button className="suggestion-card" key={item.label} onClick={() => void sendMessage(item.text)} disabled={busy}>
                      <span className="suggestion-icon"><Icon size={18} strokeWidth={1.8} /></span>
                      <span><strong>{item.label}</strong><small>{item.text}</small></span>
                      <ArrowUp className="suggestion-arrow" size={15} />
                    </button>
                  );
                })}
              </div>
              <div className="trust-note"><ShieldCheck size={15} /> Your conversation stays private and secure.</div>
            </section>
          ) : (
            <section className="conversation" aria-live="polite">
              <div className="conversation-date"><span /> TODAY <span /></div>
              {messages.map((message, index) => (
                <article className={"message-row " + message.role} key={index + "-" + message.role}>
                  {message.role === "assistant" && <div className="assistant-avatar"><Headphones size={16} /></div>}
                  <div className="message-content">
                    <div className="message-author">{message.role === "user" ? "You" : "SupportDesk"}
                      {message.role === "assistant" && <time>Just now</time>}
                    </div>
                    <div className={"message-bubble " + message.role + (message.error ? " error" : "")}>{message.content}</div>
                    {message.role === "assistant" && !message.error && (
                      <div className="message-meta">
                        {message.destination && <span className="meta-pill"><Sparkles size={12} /> {message.destination.replace("_", " ")}</span>}
                        {message.tool_used && <span className="meta-pill">{message.tool_used}</span>}
                        {message.retrieval_mode && <span className="meta-pill">{modeLabels[message.retrieval_mode]}{message.retrieval_fallback ? " · BM25 fallback" : ""}</span>}
                        {message.sources?.map((source) => <span className="source-tag" key={source}>{source.replace(/_/g, " ").replace(/\.md$/, "")}</span>)}
                      </div>
                    )}
                  </div>
                </article>
              ))}
              {busy && <div className="typing-row"><div className="assistant-avatar"><Headphones size={16} /></div>
                <div className="typing-bubble"><i /><i /><i /></div><span>Finding the right answer…</span></div>}
              <div ref={endRef} />
            </section>
          )}

          <div className="composer-area">
            {error && <div className="inline-error"><CircleHelp size={15} />{error}</div>}
            <form className="composer" onSubmit={submit}>
              <textarea ref={inputRef} value={draft} onChange={(event) => setDraft(event.target.value)}
                onKeyDown={composerKeyDown} placeholder="Write your message…" rows={1}
                aria-label="Your message" disabled={busy} />
              <div className="composer-footer">
                <div className="composer-hints"><span><kbd>↵</kbd> to send</span><span><kbd>⇧ ↵</kbd> new line</span></div>
                <button className="send-button" type="submit" disabled={!draft.trim() || busy} aria-label="Send message">
                  {busy ? <span className="send-spinner" /> : <ArrowUp size={17} strokeWidth={2.3} />}
                </button>
              </div>
            </form>
            <p className="disclaimer">AI can make mistakes. For urgent concerns, ask for a human support agent.</p>
          </div>
        </div>
      </main>

      <aside className="help-panel">
        <div className="help-heading"><div><span className="section-label">HELP CENTER</span><h2>Quick assistance</h2></div><LifeBuoy size={18} /></div>
        <p className="help-intro">Choose a topic to get started or ask us anything below.</p>
        <div className="help-links">{suggestions.map((item) => promptButton(item))}</div>
        <div className="search-card">
          <div className="search-card-icon"><CircleHelp size={18} /></div>
          <strong>Can’t find what you need?</strong>
          <p>Describe the issue in your own words and we’ll point you in the right direction.</p>
          <button onClick={() => inputRef.current?.focus()}>Ask a question <ArrowUp size={14} /></button>
        </div>
        <div className="mode-card">
          <label htmlFor="retrieval-mode"><span className="section-label">POLICY SEARCH</span>
            <span className="mode-info">Compare how the assistant finds policy answers.</span></label>
          <select id="retrieval-mode" value={mode} onChange={(event) => setMode(event.target.value as RetrievalMode)}>
            <option value="hybrid">Hybrid</option><option value="semantic">Semantic</option><option value="bm25">BM25 keywords</option>
          </select>
          <div className="mode-caption"><Sparkles size={13} /> Applied to your next message</div>
        </div>
        <div className="hours-note"><Clock3 size={15} /><span><strong>Always here to help</strong><small>AI support is available any time.</small></span></div>
      </aside>
    </div>
  );
}

export default App;
