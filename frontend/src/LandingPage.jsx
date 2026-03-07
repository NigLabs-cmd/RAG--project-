import { useNavigate } from 'react-router-dom';
import { useEffect, useRef } from 'react';
import './landing.css';

function LandingPage() {
    const navigate = useNavigate();
    const heroMockupRef = useRef(null);

    // 3D tilt effect on the hero mockup card
    useEffect(() => {
        const card = heroMockupRef.current;
        if (!card) return;

        const handleMouseMove = (e) => {
            const rect = card.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;
            const centerX = rect.width / 2;
            const centerY = rect.height / 2;
            const rotateX = ((y - centerY) / centerY) * -8;
            const rotateY = ((x - centerX) / centerX) * 8;
            card.style.transform = `perspective(800px) rotateX(${rotateX}deg) rotateY(${rotateY}deg) scale(1.02)`;
        };

        const handleMouseLeave = () => {
            card.style.transform = 'perspective(800px) rotateX(0) rotateY(0) scale(1)';
        };

        card.addEventListener('mousemove', handleMouseMove);
        card.addEventListener('mouseleave', handleMouseLeave);
        return () => {
            card.removeEventListener('mousemove', handleMouseMove);
            card.removeEventListener('mouseleave', handleMouseLeave);
        };
    }, []);

    // Scroll-triggered fade-in
    useEffect(() => {
        const observer = new IntersectionObserver(
            (entries) => {
                entries.forEach((entry) => {
                    if (entry.isIntersecting) {
                        entry.target.classList.add('lp-visible');
                    }
                });
            },
            { threshold: 0.15 }
        );

        document.querySelectorAll('.lp-animate').forEach((el) => observer.observe(el));
        return () => observer.disconnect();
    }, []);

    return (
        <div className="lp">
            {/* Animated background */}
            <div className="lp-bg-grid" />

            {/* ─── Navbar ─── */}
            <nav className="lp-nav">
                <div className="lp-nav-inner">
                    <div className="lp-logo">
                        <div className="lp-logo-icon">
                            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                                <path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20" />
                                <path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z" />
                            </svg>
                        </div>
                        <span className="lp-logo-text">RAG ASSISTANT</span>
                    </div>

                    <div className="lp-nav-links">
                        <a href="#features">Features</a>
                        <a href="#how-it-works">How It Works</a>
                        <a href="#privacy">Privacy</a>
                    </div>

                    <button className="lp-nav-cta" onClick={() => navigate('/app')}>
                        Get Started
                    </button>
                </div>
            </nav>

            {/* ─── Hero ─── */}
            <section className="lp-hero">
                <div className="lp-hero-inner">
                    <div className="lp-hero-text">
                        <div className="lp-badge lp-animate">
                            <span className="lp-badge-dot">✦</span>
                            NEXT GENERATION LOCAL AI
                        </div>

                        <h1 className="lp-hero-title lp-animate">
                            Your Documents,<br />
                            Answered<br />
                            by <span className="lp-gradient-text">AI</span>
                        </h1>

                        <p className="lp-hero-desc lp-animate">
                            Experience the future of local intelligence.
                            Secure, private, and lightning-fast document
                            processing that never leaves your machine.
                        </p>

                        <button className="lp-cta-button lp-animate" onClick={() => navigate('/app')}>
                            Get Started Free
                        </button>
                    </div>

                    {/* 3D Mockup Card */}
                    <div className="lp-hero-mockup lp-animate" ref={heroMockupRef}>
                        <div className="lp-mockup-card">
                            <div className="lp-mockup-dots">
                                <span className="lp-dot lp-dot-red" />
                                <span className="lp-dot lp-dot-yellow" />
                                <span className="lp-dot lp-dot-green" />
                                <span className="lp-mockup-label">LOCAL LLM INTERFACE</span>
                            </div>

                            <div className="lp-mockup-messages">
                                <div className="lp-mock-msg lp-mock-user">
                                    "Explain the quarterly growth in the PDF."
                                </div>
                                <div className="lp-mock-msg lp-mock-ai">
                                    Based on Page 12, growth was 14.2% due to expansion in local markets.
                                </div>
                                <div className="lp-mock-msg lp-mock-user">
                                    "Is the data stored in the cloud?"
                                </div>
                                <div className="lp-mock-msg lp-mock-ai">
                                    <span className="lp-mock-status">●</span> No, processing is 100% local.
                                </div>
                            </div>

                            <div className="lp-mockup-input">
                                <span>Type your question...</span>
                                <div className="lp-mockup-send">▶</div>
                            </div>

                            <div className="lp-mockup-footer">VECTOR DATABASE ACTIVE</div>
                        </div>
                    </div>
                </div>
            </section>

            {/* ─── Features ─── */}
            <section className="lp-features" id="features">
                <div className="lp-section-inner">
                    <h2 className="lp-section-title lp-animate">Futuristic Intelligence</h2>
                    <p className="lp-section-subtitle lp-animate">
                        Our RAG-powered engine processes your data locally with zero compromise on privacy.
                    </p>

                    <div className="lp-features-grid">
                        {[
                            {
                                icon: (
                                    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                                        <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
                                        <polyline points="14 2 14 8 20 8" />
                                        <line x1="12" y1="18" x2="12" y2="12" />
                                        <line x1="9" y1="15" x2="15" y2="15" />
                                    </svg>
                                ),
                                title: 'Upload PDFs',
                                desc: 'Securely ingest your documents into our high-speed local vector database for instant retrieval.',
                            },
                            {
                                icon: (
                                    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                                        <circle cx="12" cy="12" r="3" />
                                        <path d="M12 1v2M12 21v2M4.22 4.22l1.42 1.42M18.36 18.36l1.42 1.42M1 12h2M21 12h2M4.22 19.78l1.42-1.42M18.36 5.64l1.42-1.42" />
                                    </svg>
                                ),
                                title: 'AI-Powered Answers',
                                desc: 'Ask complex questions and get instant, cited answers based solely on your specific document corpus.',
                            },
                            {
                                icon: (
                                    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                                        <rect x="3" y="11" width="18" height="11" rx="2" ry="2" />
                                        <path d="M7 11V7a5 5 0 0 1 10 0v4" />
                                    </svg>
                                ),
                                title: '100% Private & Local',
                                desc: 'Your data never leaves your machine. Local LLM processing at its best, ensuring total privacy.',
                            },
                        ].map((feature, i) => (
                            <div className="lp-feature-card lp-animate" key={i}>
                                <div className="lp-feature-icon">{feature.icon}</div>
                                <h3 className="lp-feature-title">{feature.title}</h3>
                                <p className="lp-feature-desc">{feature.desc}</p>
                                <button className="lp-feature-link" onClick={() => navigate('/app')}>
                                    LEARN MORE <span>›</span>
                                </button>
                            </div>
                        ))}
                    </div>
                </div>
            </section>

            {/* ─── How It Works ─── */}
            <section className="lp-how" id="how-it-works">
                <div className="lp-section-inner">
                    <h2 className="lp-section-title lp-animate">How It Works</h2>

                    <div className="lp-steps">
                        {[
                            {
                                num: '1',
                                icon: (
                                    <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                                        <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                                        <polyline points="17 8 12 3 7 8" />
                                        <line x1="12" y1="3" x2="12" y2="15" />
                                    </svg>
                                ),
                                title: 'Upload',
                                desc: 'Drag and drop your PDFs',
                            },
                            {
                                num: '2',
                                icon: (
                                    <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                                        <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
                                    </svg>
                                ),
                                title: 'Ask',
                                desc: 'Natural language queries',
                            },
                            {
                                num: '3',
                                icon: (
                                    <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                                        <polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2" />
                                    </svg>
                                ),
                                title: 'Get Answers',
                                desc: 'Instant AI insights',
                            },
                        ].map((step, i) => (
                            <div className="lp-step lp-animate" key={i}>
                                <div className="lp-step-badge">{step.num}</div>
                                <div className="lp-step-icon-wrap">
                                    <div className="lp-step-icon">{step.icon}</div>
                                </div>
                                <h3 className="lp-step-title">{step.title}</h3>
                                <p className="lp-step-desc">{step.desc}</p>
                            </div>
                        ))}
                    </div>
                </div>
            </section>

            {/* ─── Footer ─── */}
            <footer className="lp-footer" id="privacy">
                <div className="lp-footer-inner">
                    <div className="lp-footer-logo">
                        <div className="lp-logo-icon lp-logo-icon-sm">
                            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                                <path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20" />
                                <path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z" />
                            </svg>
                        </div>
                        <span>RAG ASSISTANT</span>
                    </div>

                    <p className="lp-footer-text">
                        Built with <span className="lp-heart">❤</span> using RAG technology. © 2026
                    </p>

                    <div className="lp-footer-share">
                        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                            <circle cx="18" cy="5" r="3" />
                            <circle cx="6" cy="12" r="3" />
                            <circle cx="18" cy="19" r="3" />
                            <line x1="8.59" y1="13.51" x2="15.42" y2="17.49" />
                            <line x1="15.41" y1="6.51" x2="8.59" y2="10.49" />
                        </svg>
                    </div>
                </div>
            </footer>
        </div>
    );
}

export default LandingPage;
