import { useState } from 'react';
import './index.css';

const API_BASE_URL = 'http://localhost:8000';

function App() {
  const [messages, setMessages] = useState([]);
  const [query, setQuery] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [theme, setTheme] = useState('dark');

  const toggleTheme = () => {
    setTheme(prevTheme => prevTheme === 'dark' ? 'light' : 'dark');
  };

  const getConfidenceLevel = (confidence) => {
    if (confidence >= 0.7) return 'high';
    if (confidence >= 0.4) return 'medium';
    return 'low';
  };

  const handleSubmit = async (e) => {
    e.preventDefault();

    if (!query.trim()) return;

    // Add user message
    const userMessage = {
      type: 'user',
      content: query,
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    setQuery('');
    setLoading(true);
    setError(null);

    try {
      const response = await fetch(`${API_BASE_URL}/query`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ question: query }),
      });

      if (!response.ok) {
        throw new Error(`Server error: ${response.status}`);
      }

      const data = await response.json();

      // Add assistant message
      const assistantMessage = {
        type: 'assistant',
        content: data.answer,
        confidence: data.confidence,
        hasAnswer: data.has_answer,
        citations: data.citations,
        sources: data.sources,
        queryTime: data.query_time,
        timestamp: new Date()
      };

      setMessages(prev => [...prev, assistantMessage]);

    } catch (err) {
      setError(err.message);
      console.error('Query error:', err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className={`app ${theme}`}>
      {/* Sidebar */}
      <aside className="sidebar">
        <div className="sidebar-header">
          <h2>RAG Assistant</h2>
          <p className="sidebar-subtitle">Document Q&A</p>
        </div>

        <button className="upload-button">
          <span className="upload-icon">⬆</span>
          Upload Document
        </button>

        <div className="documents-section">
          <h3 className="documents-title">DOCUMENTS (0)</h3>
          <div className="empty-documents">
            <div className="empty-doc-icon">📄</div>
            <p className="empty-doc-text">No documents uploaded yet</p>
          </div>
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
                <p>Upload documents using the sidebar and ask questions about their content. The AI will provide answers based on your documents.</p>
              </div>
            ) : (
              messages.map((message, index) => (
                <div key={index} className="message">
                  {message.type === 'user' ? (
                    <div className="user-query">
                      {message.content}
                    </div>
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

                      <div className="answer-text">
                        {message.content}
                      </div>

                      {message.citations && message.citations.length > 0 && (
                        <div className="citations">
                          <div className="citations-title">📚 Citations</div>
                          <div className="citation-tags">
                            {message.citations.map((citation, idx) => (
                              <span key={idx} className="citation-tag">
                                [{citation}]
                              </span>
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
                              <div className="source-text">
                                {source.text.substring(0, 150)}...
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
                <div className="spinner"></div>
                <span>Searching knowledge base...</span>
              </div>
            )}
          </div>

          {error && (
            <div className="error-message">
              ❌ Error: {error}. Make sure the backend server is running on port 8000.
            </div>
          )}

          <form onSubmit={handleSubmit} className="input-area">
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Ask a question..."
              className="input-field"
              disabled={loading}
            />
            <button
              type="submit"
              className="submit-button"
              disabled={loading || !query.trim()}
            >
              {loading ? 'Searching...' : 'Ask'}
            </button>
          </form>
        </div>
      </main>
    </div>
  );
}

export default App;
