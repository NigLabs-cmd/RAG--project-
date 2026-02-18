import { useState, useRef, useEffect } from 'react';
import './index.css';

const API_BASE_URL = 'http://localhost:8000';

function App() {
  const [messages, setMessages] = useState([]);
  const [query, setQuery] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [theme, setTheme] = useState('dark');
  const [uploadedDocs, setUploadedDocs] = useState([]);
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState(null);
  const fileInputRef = useRef(null);
  const messagesEndRef = useRef(null);

  // Load existing documents on mount
  useEffect(() => {
    fetchDocuments();
  }, []);

  // Auto-scroll to latest message
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, loading]);

  const fetchDocuments = async () => {
    try {
      const res = await fetch(`${API_BASE_URL}/documents`);
      if (res.ok) {
        const data = await res.json();
        setUploadedDocs(data.documents || []);
      }
    } catch (err) {
      // Backend may not be ready yet — silently ignore
    }
  };

  const handleDeleteDoc = async (docName) => {
    if (!window.confirm(`Remove "${docName}" from the knowledge base?`)) return;
    try {
      const res = await fetch(`${API_BASE_URL}/documents/${encodeURIComponent(docName)}`, {
        method: 'DELETE',
      });
      if (!res.ok) {
        const errData = await res.json();
        throw new Error(errData.detail || `Delete failed (${res.status})`);
      }
      await fetchDocuments();
      setMessages(prev => [...prev, {
        type: 'system',
        content: `🗑️ "${docName}" removed from knowledge base.`,
        timestamp: new Date(),
      }]);
    } catch (err) {
      setUploadError(`Delete failed: ${err.message}`);
    }
  };

  const toggleTheme = () => {
    setTheme(prev => prev === 'dark' ? 'light' : 'dark');
  };

  const getConfidenceLevel = (confidence) => {
    if (confidence >= 0.7) return 'high';
    if (confidence >= 0.4) return 'medium';
    return 'low';
  };

  const handleUploadClick = () => {
    setUploadError(null);
    fileInputRef.current?.click();
  };

  const handleFileChange = async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;

    // Reset input so same file can be re-uploaded if needed
    e.target.value = '';

    if (!file.name.toLowerCase().endsWith('.pdf')) {
      setUploadError('Only PDF files are supported.');
      return;
    }

    setUploading(true);
    setUploadError(null);

    try {
      const formData = new FormData();
      formData.append('file', file);

      const res = await fetch(`${API_BASE_URL}/upload`, {
        method: 'POST',
        body: formData,
      });

      if (!res.ok) {
        const errData = await res.json();
        throw new Error(errData.detail || `Upload failed (${res.status})`);
      }

      const data = await res.json();
      // Refresh document list from server
      await fetchDocuments();

      // Show a system message in chat
      setMessages(prev => [...prev, {
        type: 'system',
        content: `✅ "${data.filename}" uploaded — ${data.chunks_added} chunks added to knowledge base.`,
        timestamp: new Date(),
      }]);

    } catch (err) {
      setUploadError(err.message);
    } finally {
      setUploading(false);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!query.trim()) return;

    const userMessage = {
      type: 'user',
      content: query,
      timestamp: new Date(),
    };

    setMessages(prev => [...prev, userMessage]);
    setQuery('');
    setLoading(true);
    setError(null);

    try {
      const response = await fetch(`${API_BASE_URL}/query`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: query }),
      });

      if (!response.ok) {
        throw new Error(`Server error: ${response.status}`);
      }

      const data = await response.json();

      const assistantMessage = {
        type: 'assistant',
        content: data.answer,
        confidence: data.confidence,
        hasAnswer: data.has_answer,
        citations: data.citations,
        sources: data.sources,
        queryTime: data.query_time,
        timestamp: new Date(),
      };

      setMessages(prev => [...prev, assistantMessage]);

    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className={`app ${theme}`}>
      {/* Hidden file input */}
      <input
        ref={fileInputRef}
        type="file"
        accept=".pdf"
        style={{ display: 'none' }}
        onChange={handleFileChange}
      />

      {/* Sidebar */}
      <aside className="sidebar">
        <div className="sidebar-header">
          <h2>RAG Assistant</h2>
          <p className="sidebar-subtitle">Document Q&amp;A</p>
        </div>

        <button
          className={`upload-button ${uploading ? 'uploading' : ''}`}
          onClick={handleUploadClick}
          disabled={uploading}
        >
          {uploading ? (
            <>
              <span className="upload-spinner" />
              Processing…
            </>
          ) : (
            <>
              <span className="upload-icon">⬆</span>
              Upload PDF
            </>
          )}
        </button>

        {uploadError && (
          <div className="upload-error">❌ {uploadError}</div>
        )}

        <div className="documents-section">
          <h3 className="documents-title">
            DOCUMENTS ({uploadedDocs.length})
          </h3>

          {uploadedDocs.length === 0 ? (
            <div className="empty-documents">
              <div className="empty-doc-icon">📄</div>
              <p className="empty-doc-text">No documents uploaded yet</p>
            </div>
          ) : (
            <ul className="doc-list">
              {uploadedDocs.map((doc, i) => (
                <li key={i} className="doc-item">
                  <span className="doc-icon">📄</span>
                  <span className="doc-name" title={doc.name}>
                    {doc.name.length > 22 ? doc.name.slice(0, 20) + '…' : doc.name}
                  </span>
                  <span className="doc-badge">{doc.chunks}c</span>
                  <button
                    className="doc-delete-btn"
                    title={`Remove ${doc.name}`}
                    onClick={() => handleDeleteDoc(doc.name)}
                  >×</button>
                </li>
              ))}
            </ul>
          )}
        </div>
      </aside>

      {/* Main Chat Area */}
      <main className="main-content">
        <header className="chat-header">
          <div>
            <h1>Chat</h1>
            <p className="chat-subtitle">Ask questions about your documents</p>
          </div>
          <button className="theme-toggle" title="Toggle theme" onClick={toggleTheme}>
            <span>{theme === 'dark' ? '☀️' : '🌙'}</span>
          </button>
        </header>

        <div className="chat-container">
          <div className="messages">
            {messages.length === 0 ? (
              <div className="empty-state">
                <div className="empty-state-icon">💬</div>
                <h3>Start a conversation</h3>
                <p>Upload a PDF using the sidebar and ask questions about its content.</p>
              </div>
            ) : (
              messages.map((message, index) => (
                <div key={index} className="message">
                  {message.type === 'user' ? (
                    <div className="user-query">{message.content}</div>
                  ) : message.type === 'system' ? (
                    <div className="system-message">{message.content}</div>
                  ) : (
                    <div className="assistant-response">
                      <div className="response-header">
                        <span className={`confidence-badge confidence-${getConfidenceLevel(message.confidence)}`}>
                          Confidence: {(message.confidence * 100).toFixed(1)}%
                        </span>
                        {!message.hasAnswer && (
                          <span className="confidence-badge confidence-low">
                            ⚠️ No Answer Found
                          </span>
                        )}
                        <span style={{ marginLeft: 'auto', fontSize: '0.85rem', color: 'var(--text-muted)' }}>
                          {message.queryTime.toFixed(2)}s
                        </span>
                      </div>

                      <div className="answer-text">{message.content}</div>

                      {message.citations && message.citations.length > 0 && (
                        <div className="citations">
                          <div className="citations-title">📚 Citations</div>
                          <div className="citation-tags">
                            {message.citations.map((citation, idx) => (
                              <span key={idx} className="citation-tag">[{citation}]</span>
                            ))}
                          </div>
                        </div>
                      )}

                      {message.sources && message.sources.length > 0 && (
                        <div className="sources">
                          <div className="sources-title">
                            📄 Sources ({message.sources.length})
                          </div>
                          {message.sources.map((source, idx) => (
                            <div key={idx} className="source-item">
                              {source.metadata?.source && (
                                <div className="source-filename">
                                  📎 {source.metadata.source}
                                  {source.metadata.page && ` · page ${source.metadata.page}`}
                                </div>
                              )}
                              <div className="source-text">
                                {source.text.substring(0, 200)}…
                              </div>
                              <div className="source-score">
                                Similarity: {(source.score * 100).toFixed(1)}%
                              </div>
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  )}
                </div>
              ))
            )}

            {loading && (
              <div className="loading">
                <div className="spinner" />
                <span>Searching knowledge base…</span>
              </div>
            )}

            <div ref={messagesEndRef} />
          </div>

          {error && (
            <div className="error-message">
              ❌ {error}. Make sure the backend is running on port 8000.
            </div>
          )}

          <form onSubmit={handleSubmit} className="input-area">
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Ask a question about your documents…"
              className="input-field"
              disabled={loading}
            />
            <button
              type="submit"
              className="submit-button"
              disabled={loading || !query.trim()}
            >
              {loading ? 'Searching…' : 'Ask'}
            </button>
          </form>
        </div>
      </main>
    </div>
  );
}

export default App;
