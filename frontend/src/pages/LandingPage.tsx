import { useState, useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAppStore } from '../store/appStore';
import './LandingPage.css';

export default function LandingPage() {
  const navigate = useNavigate();
  const animatedRefs = useRef<(HTMLElement | HTMLDivElement | null)[]>([]);

  // Login / Signup dropdown state
  const [showLogin, setShowLogin] = useState(false);
  const [authTab, setAuthTab] = useState<'signin' | 'signup'>('signin');
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [authError, setAuthError] = useState('');
  const [createdApiKey, setCreatedApiKey] = useState<string | null>(null);
  const dropdownRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            entry.target.classList.add('is-visible');
          }
        });
      },
      { threshold: 0.15 }
    );

    animatedRefs.current.forEach((ref) => {
      if (ref) observer.observe(ref);
    });

    return () => observer.disconnect();
  }, []);

  // Close dropdown when clicking outside
  useEffect(() => {
    function handleClickOutside(e: MouseEvent) {
      if (dropdownRef.current && !dropdownRef.current.contains(e.target as Node)) {
        setShowLogin(false);
        setAuthError('');
        setCreatedApiKey(null);
      }
    }
    if (showLogin) {
      document.addEventListener('mousedown', handleClickOutside);
    }
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, [showLogin]);

  // Helper to load accounts from localStorage
  const getStoredUsers = () => {
    try {
      const stored = localStorage.getItem('hex_users');
      return stored ? JSON.parse(stored) : {};
    } catch {
      return {};
    }
  };

  const handleAuthSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const cleanUser = username.trim();

    if (!cleanUser || !password) {
      setAuthError('Please enter username and password');
      return;
    }

    if (authTab === 'signin') {
      // Sign in logic
      const users = getStoredUsers();
      
      // Default hardcoded check or registered user check
      if ((cleanUser.toLowerCase() === 'paybros' && password === '1234') || (users[cleanUser.toLowerCase()] && users[cleanUser.toLowerCase()].password === password)) {
        const apiKey = cleanUser.toLowerCase() === 'paybros' ? 'cad_998124a3b81f' : users[cleanUser.toLowerCase()].apiKey;
        localStorage.setItem('hex_currentUser', JSON.stringify({ username: cleanUser, apiKey }));
        useAppStore.setState({ overview: null, customers: [], clusters: [], graphNodes: [], graphLinks: [], activityEvents: [] });
        setAuthError('');
        navigate('/overview');
      } else {
        setAuthError('Invalid username or password');
      }
    } else {
      // Sign up logic
      if (password !== confirmPassword) {
        setAuthError('Passwords do not match');
        return;
      }
      if (password.length < 3) {
        setAuthError('Password must be at least 3 characters');
        return;
      }

      const users = getStoredUsers();
      if (cleanUser.toLowerCase() === 'paybros' || users[cleanUser.toLowerCase()]) {
        setAuthError('Username already exists');
        return;
      }

      // Generate API key formatted as cad_<12_hex_chars>
      const randomHex = Array.from({ length: 12 }, () => Math.floor(Math.random() * 16).toString(16)).join('');
      const generatedApiKey = `cad_${randomHex}`;

      // Save user
      users[cleanUser.toLowerCase()] = {
        username: cleanUser,
        password,
        apiKey: generatedApiKey,
        createdAt: new Date().toISOString()
      };
      localStorage.setItem('hex_users', JSON.stringify(users));

      // Save active session
      localStorage.setItem('hex_currentUser', JSON.stringify({ username: cleanUser, apiKey: generatedApiKey }));

      setCreatedApiKey(generatedApiKey);
      setAuthError('');
    }
  };

  return (
    <div className="landing-container">
      {/* SECTION 1: HERO VIEWPORT */}
      <section className="landing-hero-section">
        {/* Top Header */}
        <header className="landing-header">
          <div className="landing-logo">HEX</div>

          {/* Enter Dashboard Button & Dropdown Container */}
          <div className="landing-login-wrapper" ref={dropdownRef}>
            <button
              className={`landing-action ${showLogin ? 'active' : ''}`}
              onClick={() => {
                setShowLogin(!showLogin);
                setAuthError('');
                setCreatedApiKey(null);
              }}
            >
              ENTER DASHBOARD <span className="arrow">{showLogin ? '▲' : '→'}</span>
            </button>

            {/* Animated Login Popover Box */}
            <div className={`login-dropdown-popover ${showLogin ? 'open' : ''}`}>
              <div className="login-dropdown-header">
                <span className="login-dropdown-title">PAY BROS AUTHENTICATION</span>
              </div>

              {/* Tab Switcher */}
              <div className="login-tabs">
                <button
                  type="button"
                  className={`login-tab-btn ${authTab === 'signin' ? 'active' : ''}`}
                  onClick={() => {
                    setAuthTab('signin');
                    setAuthError('');
                    setCreatedApiKey(null);
                  }}
                >
                  SIGN IN
                </button>
                <button
                  type="button"
                  className={`login-tab-btn ${authTab === 'signup' ? 'active' : ''}`}
                  onClick={() => {
                    setAuthTab('signup');
                    setAuthError('');
                    setCreatedApiKey(null);
                  }}
                >
                  SIGN UP
                </button>
              </div>

              {createdApiKey ? (
                <div className="login-success-box">
                  <div className="success-badge">ACCOUNT CREATED ✓</div>
                  <p className="success-sub">Your generated API key:</p>
                  <code className="generated-key-display">{createdApiKey}</code>
                  <button
                    type="button"
                    className="login-submit-btn"
                    style={{ marginTop: '0.8rem' }}
                    onClick={() => navigate('/overview')}
                  >
                    CONTINUE TO DASHBOARD <span className="arrow">→</span>
                  </button>
                </div>
              ) : (
                <form onSubmit={handleAuthSubmit} className="login-dropdown-form">
                  <div className="login-field-group">
                    <label className="login-field-label">USERNAME</label>
                    <input
                      type="text"
                      className="login-field-input"
                      placeholder={authTab === 'signin' ? "payBros" : "new_merchant"}
                      value={username}
                      onChange={(e) => {
                        setUsername(e.target.value);
                        setAuthError('');
                      }}
                      autoFocus={showLogin}
                    />
                  </div>

                  <div className="login-field-group">
                    <label className="login-field-label">PASSWORD</label>
                    <input
                      type="password"
                      className="login-field-input"
                      placeholder="••••"
                      value={password}
                      onChange={(e) => {
                        setPassword(e.target.value);
                        setAuthError('');
                      }}
                    />
                  </div>

                  {authTab === 'signup' && (
                    <div className="login-field-group">
                      <label className="login-field-label">CONFIRM PASSWORD</label>
                      <input
                        type="password"
                        className="login-field-input"
                        placeholder="••••"
                        value={confirmPassword}
                        onChange={(e) => {
                          setConfirmPassword(e.target.value);
                          setAuthError('');
                        }}
                      />
                    </div>
                  )}

                  {authError && (
                    <div className="login-error-msg">
                      ✕ {authError}
                    </div>
                  )}

                  <button type="submit" className="login-submit-btn">
                    {authTab === 'signin' ? 'SIGN IN TO CONSOLE' : 'CREATE ACCOUNT & GENERATE KEY'}{' '}
                    <span className="arrow">→</span>
                  </button>

                  {authTab === 'signin' && (
                    <div className="login-hint-box">
                      Default demo: <code>payBros</code> / <code>1234</code>
                    </div>
                  )}
                </form>
              )}
            </div>
          </div>
        </header>

        {/* Hero Headline & Subtext */}
        <main className="landing-hero">
          <h1 className="landing-title">
            <span className="landing-title-line">ONE ABUSE RING.</span>
            <span className="landing-title-line">MANY IDENTITIES.</span>
          </h1>
          <p className="landing-subtext">Real-time offer abuse intelligence</p>
        </main>

        {/* Bottom Editorial Metadata */}
        <footer className="landing-footer">
          <div className="landing-meta-left">
            GRAPH INTELLIGENCE · MACHINE LEARNING · REAL-TIME DETECTION
          </div>
          <div className="landing-meta-right">
            MERCHANT SECURITY
          </div>
        </footer>
      </section>

      {/* SECTION 2: VERTICAL 3-STEP EDITORIAL NARRATIVE (Completely untouched) */}
      <section className="vertical-editorial-section">
        <div className="vertical-editorial-container">
          
          {/* STEP 01 — MERCHANT */}
          <article className="vertical-step-block scroll-reveal" ref={(el) => { animatedRefs.current[0] = el; }}>
            <div className="step-content">
              <div className="step-tag">01 — MERCHANT</div>
              <h2 className="step-headline">YOU ISSUE A SINGLE-USE OFFER.</h2>
              <p className="step-subtext">
                "Designed for one redemption. One customer."
              </p>
            </div>

            <div className="step-visual-container">
              <svg className="step-svg" viewBox="0 0 450 450" style={{ filter: 'drop-shadow(0 18px 36px rgba(0,0,0,0.85))' }}>
                {/* Sticky Ticket / Promo Code Paper */}
                <g transform="translate(45, 30) rotate(-2 180 180)">
                  {/* Paper Background */}
                  <rect x="0" y="0" width="310" height="310" rx="4" fill="#EBE9E1" stroke="#D3CFCE" strokeWidth="1.5" />
                  
                  {/* Inner Dashed Ticket Border */}
                  <rect x="18" y="18" width="274" height="274" fill="none" stroke="#7A7775" strokeWidth="1" strokeDasharray="4 3" />

                  {/* Header */}
                  <text x="155" y="62" textAnchor="middle" fill="#4A4745" fontSize="13" fontFamily="JetBrains Mono, monospace" fontWeight="600" letterSpacing="0.2em">
                    PROMO CODE
                  </text>

                  {/* Promo Code HEX50 */}
                  <text x="155" y="145" textAnchor="middle" fill="#E5341C" fontSize="84" fontFamily="League Gothic, Bebas Neue, sans-serif" fontWeight="400" letterSpacing="0.04em">
                    HEX50
                  </text>

                  {/* Single Use Label */}
                  <text x="155" y="186" textAnchor="middle" fill="#2E2C2B" fontSize="13" fontFamily="JetBrains Mono, monospace" fontWeight="700" letterSpacing="0.15em">
                    SINGLE USE ONLY
                  </text>

                  {/* Barcode Graphic */}
                  <g transform="translate(65, 220)">
                    {[3,2,5,2,4,3,2,6,3,2,5,3,2,4,2,5,3,2,4,3,2,5,3,2,4].map((width, i, arr) => {
                      const x = arr.slice(0, i).reduce((sum, val) => sum + val + 2, 0);
                      return <rect key={i} x={x} y={0} width={width} height={38} fill="#2E2C2B" />;
                    })}
                  </g>
                </g>

                {/* Red Tape Strip across Bottom */}
                <g transform="translate(100, 280) rotate(-5)">
                  <path d="M 0,0 L 265,-6 L 261,54 L -4,60 Z" fill="#E5341C" />
                  <text x="130" y="37" textAnchor="middle" fill="#F4F3EE" fontSize="24" fontFamily="League Gothic, Bebas Neue, sans-serif" fontWeight="400" letterSpacing="0.1em">
                    1 REDEMPTION
                  </text>
                  <line x1="42" y1="43" x2="218" y2="39" stroke="#F4F3EE" strokeWidth="2.5" />
                </g>
              </svg>
            </div>
          </article>

          {/* STEP 02 — ABUSE */}
          <article className="vertical-step-block scroll-reveal" ref={(el) => { animatedRefs.current[1] = el; }}>
            <div className="step-content">
              <div className="step-tag">02 — ABUSE</div>
              <h2 className="step-headline">CUSTOMERS REDEEM IT ACROSS MULTIPLE IDENTITIES.</h2>
              <p className="step-subtext">
                "New accounts. Repeated access to the same promotion."
              </p>
            </div>

            <div className="step-visual-container">
              <svg className="step-svg" viewBox="0 0 520 440" style={{ filter: 'drop-shadow(0 14px 28px rgba(0,0,0,0.85))' }}>
                {/* Scattered Customer Identity Tickets */}
                <g transform="translate(35, 45) rotate(-5)">
                  <rect x="0" y="0" width="145" height="64" rx="3" fill="#EBE9E1" stroke="#D3CFCE" strokeWidth="1" />
                  <text x="18" y="30" fill="#1C1B1A" fontSize="17" fontFamily="JetBrains Mono, monospace" fontWeight="800" letterSpacing="0.05em">CUST_01</text>
                  <text x="18" y="48" fill="#5E5C5A" fontSize="10" fontFamily="JetBrains Mono, monospace" fontWeight="700" letterSpacing="0.1em">NEW ACCOUNT</text>
                </g>

                <g transform="translate(275, 25) rotate(4)">
                  <rect x="0" y="0" width="145" height="64" rx="3" fill="#EBE9E1" stroke="#D3CFCE" strokeWidth="1" />
                  <text x="18" y="30" fill="#1C1B1A" fontSize="17" fontFamily="JetBrains Mono, monospace" fontWeight="800" letterSpacing="0.05em">CUST_07</text>
                  <text x="18" y="48" fill="#5E5C5A" fontSize="10" fontFamily="JetBrains Mono, monospace" fontWeight="700" letterSpacing="0.1em">NEW ACCOUNT</text>
                </g>

                <g transform="translate(170, 130) rotate(-2)">
                  <rect x="0" y="0" width="145" height="64" rx="3" fill="#EBE9E1" stroke="#D3CFCE" strokeWidth="1" />
                  <text x="18" y="30" fill="#1C1B1A" fontSize="17" fontFamily="JetBrains Mono, monospace" fontWeight="800" letterSpacing="0.05em">CUST_18</text>
                  <text x="18" y="48" fill="#5E5C5A" fontSize="10" fontFamily="JetBrains Mono, monospace" fontWeight="700" letterSpacing="0.1em">NEW ACCOUNT</text>
                </g>

                <g transform="translate(360, 115) rotate(3)">
                  <rect x="0" y="0" width="145" height="64" rx="3" fill="#EBE9E1" stroke="#D3CFCE" strokeWidth="1" />
                  <text x="18" y="30" fill="#1C1B1A" fontSize="17" fontFamily="JetBrains Mono, monospace" fontWeight="800" letterSpacing="0.05em">CUST_13</text>
                  <text x="18" y="48" fill="#5E5C5A" fontSize="10" fontFamily="JetBrains Mono, monospace" fontWeight="700" letterSpacing="0.1em">NEW ACCOUNT</text>
                </g>

                <g transform="translate(10, 185) rotate(-4)">
                  <rect x="0" y="0" width="145" height="64" rx="3" fill="#EBE9E1" stroke="#D3CFCE" strokeWidth="1" />
                  <text x="18" y="30" fill="#1C1B1A" fontSize="17" fontFamily="JetBrains Mono, monospace" fontWeight="800" letterSpacing="0.05em">CUST_12</text>
                  <text x="18" y="48" fill="#5E5C5A" fontSize="10" fontFamily="JetBrains Mono, monospace" fontWeight="700" letterSpacing="0.1em">NEW ACCOUNT</text>
                </g>

                <g transform="translate(265, 235) rotate(1)">
                  <rect x="0" y="0" width="145" height="64" rx="3" fill="#EBE9E1" stroke="#D3CFCE" strokeWidth="1" />
                  <text x="18" y="30" fill="#1C1B1A" fontSize="17" fontFamily="JetBrains Mono, monospace" fontWeight="800" letterSpacing="0.05em">CUST_31</text>
                  <text x="18" y="48" fill="#5E5C5A" fontSize="10" fontFamily="JetBrains Mono, monospace" fontWeight="700" letterSpacing="0.1em">NEW ACCOUNT</text>
                </g>

                <g transform="translate(45, 315) rotate(5)">
                  <rect x="0" y="0" width="145" height="64" rx="3" fill="#EBE9E1" stroke="#D3CFCE" strokeWidth="1" />
                  <text x="18" y="30" fill="#1C1B1A" fontSize="17" fontFamily="JetBrains Mono, monospace" fontWeight="800" letterSpacing="0.05em">CUST_42</text>
                  <text x="18" y="48" fill="#5E5C5A" fontSize="10" fontFamily="JetBrains Mono, monospace" fontWeight="700" letterSpacing="0.1em">NEW ACCOUNT</text>
                </g>

                {/* Bottom Right Red Stamped Text */}
                <g transform="translate(275, 360)">
                  <text x="0" y="20" fill="#E5341C" fontSize="15" fontFamily="JetBrains Mono, monospace" fontWeight="800" letterSpacing="0.12em">
                    SAME OFFER.
                  </text>
                  <text x="0" y="40" fill="#E5341C" fontSize="15" fontFamily="JetBrains Mono, monospace" fontWeight="800" letterSpacing="0.12em">
                    DIFFERENT IDENTITIES.
                  </text>
                </g>
              </svg>
            </div>
          </article>

          {/* STEP 03 — HEX */}
          <article className="vertical-step-block scroll-reveal" ref={(el) => { animatedRefs.current[2] = el; }}>
            <div className="step-content">
              <div className="step-tag">03 — HEX</div>
              <h2 className="step-headline">HEX CONNECTS THE IDENTITIES AND REVEALS THE NETWORK.</h2>
              <p className="step-subtext">
                "Devices, payments, addresses and IPs reveal the network."
              </p>
            </div>

            <div className="step-visual-container">
              <svg className="step-svg" viewBox="0 0 540 450" style={{ filter: 'drop-shadow(0 16px 32px rgba(0,0,0,0.85))' }}>
                {/* Red Key Network Edges (Radiating from Central Core) */}
                <line x1="270" y1="190" x2="160" y2="80" stroke="#E5341C" strokeWidth="2.5" opacity="0.9" />
                <line x1="270" y1="190" x2="400" y2="100" stroke="#E5341C" strokeWidth="2.5" opacity="0.9" />
                <line x1="270" y1="190" x2="100" y2="290" stroke="#E5341C" strokeWidth="2.5" opacity="0.9" />
                <line x1="270" y1="190" x2="410" y2="340" stroke="#E5341C" strokeWidth="2.5" opacity="0.9" />

                {/* Network Web Edges */}
                <line x1="160" y1="80" x2="70" y2="100" stroke="rgba(244,243,238,0.22)" strokeWidth="1.2" />
                <line x1="160" y1="80" x2="110" y2="140" stroke="rgba(244,243,238,0.22)" strokeWidth="1.2" />
                <line x1="160" y1="80" x2="220" y2="45" stroke="rgba(244,243,238,0.22)" strokeWidth="1.2" />
                <line x1="400" y1="100" x2="330" y2="40" stroke="rgba(244,243,238,0.22)" strokeWidth="1.2" />
                <line x1="400" y1="100" x2="470" y2="180" stroke="rgba(244,243,238,0.22)" strokeWidth="1.2" />
                <line x1="400" y1="100" x2="295" y2="145" stroke="rgba(244,243,238,0.22)" strokeWidth="1.2" />
                <line x1="100" y1="290" x2="40" y2="200" stroke="rgba(244,243,238,0.22)" strokeWidth="1.2" />
                <line x1="100" y1="290" x2="180" y2="240" stroke="rgba(244,243,238,0.22)" strokeWidth="1.2" />
                <line x1="100" y1="290" x2="180" y2="340" stroke="rgba(244,243,238,0.22)" strokeWidth="1.2" />
                <line x1="410" y1="340" x2="330" y2="330" stroke="rgba(244,243,238,0.22)" strokeWidth="1.2" />
                <line x1="410" y1="340" x2="470" y2="300" stroke="rgba(244,243,238,0.22)" strokeWidth="1.2" />

                {/* Sub-mesh Connections */}
                <line x1="110" y1="140" x2="180" y2="240" stroke="rgba(244,243,238,0.18)" strokeWidth="1" />
                <line x1="295" y1="145" x2="330" y2="240" stroke="rgba(244,243,238,0.18)" strokeWidth="1" />
                <line x1="330" y1="240" x2="410" y2="340" stroke="rgba(244,243,238,0.18)" strokeWidth="1" />
                <line x1="180" y1="340" x2="270" y2="190" stroke="rgba(244,243,238,0.18)" strokeWidth="1" />

                {/* Off-White Customer Account Nodes */}
                <circle cx="70" cy="100" r="7" fill="#EBE9E1" />
                <circle cx="110" cy="140" r="8" fill="#EBE9E1" />
                <circle cx="220" cy="45" r="7" fill="#EBE9E1" />
                <circle cx="330" cy="40" r="6" fill="#EBE9E1" />
                <circle cx="250" cy="120" r="7" fill="#EBE9E1" />
                <circle cx="295" cy="145" r="8" fill="#EBE9E1" />
                <circle cx="180" cy="240" r="8" fill="#EBE9E1" />
                <circle cx="180" cy="340" r="7" fill="#EBE9E1" />
                <circle cx="330" cy="240" r="8" fill="#EBE9E1" />
                <circle cx="330" cy="330" r="7" fill="#EBE9E1" />
                <circle cx="470" cy="180" r="7" fill="#EBE9E1" />
                <circle cx="470" cy="300" r="8" fill="#EBE9E1" />
                <circle cx="40" cy="200" r="7" fill="#EBE9E1" />

                {/* Central Cluster Core Node */}
                <circle cx="270" cy="190" r="16" fill="#E5341C" />
                <circle cx="270" cy="190" r="24" fill="none" stroke="#E5341C" strokeWidth="1.5" opacity="0.6" />

                {/* 4 Red Key Shared Entity Nodes with Halo Outer Rings */}
                <g>
                  <circle cx="160" cy="80" r="11" fill="#E5341C" />
                  <circle cx="160" cy="80" r="16" fill="none" stroke="#E5341C" strokeWidth="1.5" opacity="0.6" />
                  <text x="160" y="54" textAnchor="middle" fill="#E5341C" fontSize="11" fontFamily="JetBrains Mono, monospace" fontWeight="800" letterSpacing="0.08em">SHARED DEVICE</text>
                </g>

                <g>
                  <circle cx="400" cy="100" r="11" fill="#E5341C" />
                  <circle cx="400" cy="100" r="16" fill="none" stroke="#E5341C" strokeWidth="1.5" opacity="0.6" />
                  <text x="400" y="74" textAnchor="middle" fill="#E5341C" fontSize="11" fontFamily="JetBrains Mono, monospace" fontWeight="800" letterSpacing="0.08em">SHARED PAYMENT</text>
                </g>

                <g>
                  <circle cx="100" cy="290" r="11" fill="#E5341C" />
                  <circle cx="100" cy="290" r="16" fill="none" stroke="#E5341C" strokeWidth="1.5" opacity="0.6" />
                  <text x="100" y="322" textAnchor="middle" fill="#E5341C" fontSize="11" fontFamily="JetBrains Mono, monospace" fontWeight="800" letterSpacing="0.08em">SHARED ADDRESS</text>
                </g>

                <g>
                  <circle cx="410" cy="340" r="11" fill="#E5341C" />
                  <circle cx="410" cy="340" r="16" fill="none" stroke="#E5341C" strokeWidth="1.5" opacity="0.6" />
                  <text x="420" y="372" textAnchor="middle" fill="#E5341C" fontSize="11" fontFamily="JetBrains Mono, monospace" fontWeight="800" letterSpacing="0.08em">SHARED IP</text>
                </g>

                {/* Bottom Red Tape Strip */}
                <g transform="translate(100, 360) rotate(-3)">
                  <path d="M 0,0 L 335,-6 L 331,52 L -4,58 Z" fill="#E5341C" />
                  <text x="165" y="35" textAnchor="middle" fill="#101114" fontSize="20" fontFamily="JetBrains Mono, monospace" fontWeight="800" letterSpacing="0.08em">
                    RELATIONSHIPS DON'T LIE.
                  </text>
                </g>
              </svg>
            </div>
          </article>

        </div>
      </section>

      {/* SECTION 3: 2×2 EDITORIAL ANALYTICS GRID (Cream #F4F3EE Colorway matching Section 1) */}
      <section className="editorial-grid-section">
        <div className="editorial-grid-container">
          <div className="editorial-grid-title">EVIDENCE & MODEL PERFORMANCE</div>

          <div className="editorial-grid">
            
            {/* TOP LEFT: THE SIGNAL */}
            <div
              className="grid-quadrant quad-top-left"
              ref={(el) => { animatedRefs.current[3] = el; }}
            >
              <div>
                <div className="quadrant-tag">01 — THE SIGNAL</div>
                <div className="quadrant-giant-number">21</div>
                <h3 className="quadrant-headline">21 FEATURES ANALYZED</h3>
              </div>
              <p className="quadrant-subtext">
                Behavior, timing, redemption, velocity and relationship signals combine into a single risk assessment.
              </p>
            </div>

            {/* TOP RIGHT: THE NETWORK */}
            <div
              className="grid-quadrant quad-top-right"
              ref={(el) => { animatedRefs.current[4] = el; }}
            >
              <div>
                <div className="quadrant-tag">02 — THE NETWORK</div>
                <div className="quadrant-giant-number">4</div>
                <h3 className="quadrant-headline">RELATIONSHIP TYPES</h3>
              </div>
              <p className="quadrant-subtext" style={{ fontWeight: 700, letterSpacing: '0.08em' }}>
                DEVICE · ADDRESS · PAYMENT · IP
              </p>

              {/* Sticker-Style Evidence Network Visual for Cream Background */}
              <div className="network-evidence-art">
                <svg viewBox="0 0 450 160" style={{ width: '100%', height: '100%', filter: 'drop-shadow(0 10px 20px rgba(0,0,0,0.15))' }}>
                  <line x1="80" y1="70" x2="200" y2="40" stroke="rgba(229,52,28,0.3)" strokeWidth="1.5" />
                  <line x1="200" y1="40" x2="320" y2="80" stroke="rgba(229,52,28,0.3)" strokeWidth="1.5" />
                  <line x1="200" y1="40" x2="160" y2="120" stroke="rgba(229,52,28,0.3)" strokeWidth="1.5" />
                  <line x1="320" y1="80" x2="390" y2="40" stroke="rgba(229,52,28,0.3)" strokeWidth="1.5" />
                  <line x1="320" y1="80" x2="280" y2="130" stroke="rgba(229,52,28,0.3)" strokeWidth="1.5" />

                  <line x1="200" y1="40" x2="200" y2="100" stroke="#E5341C" strokeWidth="2.5" />
                  <line x1="320" y1="80" x2="200" y2="100" stroke="#E5341C" strokeWidth="2.5" />

                  <circle cx="80" cy="70" r="7" fill="#E5341C" opacity="0.7" />
                  <circle cx="160" cy="120" r="7" fill="#E5341C" opacity="0.7" />
                  <circle cx="390" cy="40" r="7" fill="#E5341C" opacity="0.7" />
                  <circle cx="280" cy="130" r="7" fill="#E5341C" opacity="0.7" />

                  <circle cx="200" cy="40" r="10" fill="#E5341C" />
                  <circle cx="320" cy="80" r="10" fill="#E5341C" />
                  <circle cx="200" cy="100" r="13" fill="#E5341C" />

                  <g transform="translate(180, 95) rotate(-3)">
                    <path d="M 0,0 L 220,-4 L 217,38 L -3,42 Z" fill="#E5341C" />
                    <text x="110" y="26" textAnchor="middle" fill="#F4F3EE" fontSize="15" fontFamily="League Gothic, Bebas Neue, sans-serif" letterSpacing="0.1em">
                      4 RELATIONSHIPS
                    </text>
                  </g>
                </svg>
              </div>
            </div>

            {/* BOTTOM LEFT: THE MODEL */}
            <div
              className="grid-quadrant quad-bottom-left"
              ref={(el) => { animatedRefs.current[5] = el; }}
            >
              <div>
                <div className="quadrant-tag">03 — THE MODEL</div>
                <h3 className="quadrant-headline">XGBOOST</h3>
                <h3 className="quadrant-headline">GROUP-AWARE</h3>
              </div>
              <p className="quadrant-subtext">
                Evaluated across held-out customer groups to test whether the model generalizes beyond individual accounts.
              </p>
            </div>

            {/* BOTTOM RIGHT: THE RESULT */}
            <div
              className="grid-quadrant quad-bottom-right"
              ref={(el) => { animatedRefs.current[6] = el; }}
            >
              <div>
                <div className="quadrant-tag">04 — THE RESULT</div>
                <div className="quadrant-giant-number">0</div>
                <h3 className="quadrant-headline">FALSE POSITIVES</h3>
                <p className="quadrant-subtext" style={{ fontWeight: 700, letterSpacing: '0.08em', marginTop: '0.5rem' }}>
                  686 LEGITIMATE CUSTOMERS TESTED
                </p>
              </div>

              <p className="quadrant-subtext" style={{ marginTop: '1.5rem' }}>
                88.4% recall on unseen abuse rings (LOGOO cross-validation)
              </p>
            </div>

          </div>
        </div>
      </section>

      {/* SECTION 4: SYSTEM ARCHITECTURE BLUEPRINT */}
      <section className="arch-blueprint-section scroll-reveal" ref={(el) => { animatedRefs.current[7] = el; }}>
        <div className="arch-blueprint-container">
          <div className="arch-section-tag">05 — SYSTEM ARCHITECTURE</div>
          <h2 className="arch-section-headline">FULL-STACK SYSTEM ARCHITECTURE</h2>

          <div className="arch-blueprint-canvas">
            {/* TIER 1 */}
            <div className="arch-tier-box">
              <div className="arch-tier-header">
                TIER 1 · FRONTEND SPA (REACT + VITE + D3.JS)
              </div>
              <div className="arch-tier-grid tier-1-grid">
                <div className="arch-card">
                  <div className="arch-card-number">01</div>
                  <div className="arch-card-title">OVERVIEW DASHBOARD</div>
                  <ul className="arch-card-list">
                    <li>Real-Time KPI Cards</li>
                    <li>Risk Distribution Pie</li>
                  </ul>
                </div>
                <div className="arch-card">
                  <div className="arch-card-number">02</div>
                  <div className="arch-card-title">CUSTOMER CONSOLE</div>
                  <ul className="arch-card-list">
                    <li>ML Abuse Probabilities</li>
                    <li>Risk Level Badge Filters</li>
                  </ul>
                </div>
                <div className="arch-card">
                  <div className="arch-card-number">03</div>
                  <div className="arch-card-title">ABUSE CLUSTERS (D3.JS)</div>
                  <ul className="arch-card-list">
                    <li>Interactive Multi-Graph</li>
                    <li>Force Layout &amp; Ring Zoom</li>
                  </ul>
                </div>
                <div className="arch-card">
                  <div className="arch-card-number">04</div>
                  <div className="arch-card-title">INTEGRATION SPEC</div>
                  <ul className="arch-card-list">
                    <li>Live Webhook Payload Doc</li>
                    <li>Generated X-API-Key</li>
                  </ul>
                </div>
              </div>
            </div>

            {/* CONNECTING ARROW 1 */}
            <div className="arch-connector">
              <div className="arch-connector-line"></div>
              <div className="arch-connector-badge">HTTP REST / SSE STREAMING</div>
              <div className="arch-connector-line"></div>
              <div className="arch-connector-arrow">↓</div>
            </div>

            {/* TIER 2 */}
            <div className="arch-tier-box">
              <div className="arch-tier-header">
                TIER 2 · FASTAPI BACKEND &amp; MULTI-TENANT ORCHESTRATION LAYER
              </div>
              
              <div className="arch-banner-bar">
                API GATEWAY &amp; TENANT ISOLATION · X-API-KEY HEADER AUTHENTICATION · ISOLATED MEMORY WORKSPACES (_TENANT_DATASETS)
              </div>

              <div className="arch-tier-grid tier-2-grid">
                <div className="arch-card">
                  <div className="arch-card-number">01</div>
                  <div className="arch-card-title">INGESTION &amp; IDEMPOTENCY</div>
                  <div className="arch-card-subhead">POST /v1/events</div>
                  <ul className="arch-card-list">
                    <li>customer_created · order · redemption</li>
                    <li>Entity links (device, ip, payment, address)</li>
                  </ul>
                </div>
                <div className="arch-card">
                  <div className="arch-card-number">02</div>
                  <div className="arch-card-title">GRAPH RESOLUTION ENGINE</div>
                  <div className="arch-card-subhead">NETWORKX MULTI-GRAPH</div>
                  <ul className="arch-card-list">
                    <li>4 Edge Types: Hardware, Payment, IP, Addr</li>
                    <li>Connected Component Abuse Rings</li>
                  </ul>
                </div>
                <div className="arch-card">
                  <div className="arch-card-number">03</div>
                  <div className="arch-card-title">REAL-TIME EVENT BUS</div>
                  <div className="arch-card-subhead">SSE STREAM &amp; SCORE DIFFING</div>
                  <ul className="arch-card-list">
                    <li>GET /v1/events/stream (Asyncio Queue)</li>
                    <li>Live Risk Transitions (Clear → High)</li>
                  </ul>
                </div>
              </div>
            </div>

            {/* CONNECTING ARROW 2 */}
            <div className="arch-connector simple-arrow">
              <div className="arch-connector-line"></div>
              <div className="arch-connector-arrow">↓</div>
            </div>

            {/* TIER 3 */}
            <div className="arch-tier-box">
              <div className="arch-tier-header">
                TIER 3 · DATA PERSISTENCE &amp; ML INFERENCE PIPELINE
              </div>
              <div className="arch-tier-grid tier-3-grid">
                <div className="arch-card">
                  <div className="arch-card-title">STORAGE &amp; PERSISTENCE</div>
                  <div className="arch-card-subhead">SQLITE DB &amp; CSV STORE</div>
                  <ul className="arch-card-list">
                    <li>Entity mapping tables &amp; activity_logs</li>
                    <li>Single-source overview stats cache</li>
                  </ul>
                </div>
                <div className="arch-card">
                  <div className="arch-card-title">VECTOR EXTRACTION</div>
                  <div className="arch-card-subhead">21 NUMERICAL FEATURES</div>
                  <ul className="arch-card-list">
                    <li>Velocity, Spans, Discount Ratios</li>
                    <li>Vectorized NumPy &amp; StandardScaler</li>
                  </ul>
                </div>
                <div className="arch-card highlight-red-card">
                  <div className="arch-card-title red-title">ML INFERENCE ENGINE</div>
                  <div className="arch-card-subhead red-subhead">GROUP-AWARE XGBOOST</div>
                  <ul className="arch-card-list">
                    <li>LOGOO Evaluated (0 False Positives)</li>
                    <li>88.4% Recall on Unseen Abuse Rings</li>
                  </ul>
                </div>
              </div>
            </div>

            {/* BOTTOM SPECS STRIP */}
            <div className="arch-specs-strip">
              <div className="arch-spec-col">
                <div className="arch-spec-label">API LATENCY</div>
                <div className="arch-spec-value">&lt; 50 MS END-TO-END</div>
              </div>
              <div className="arch-spec-col">
                <div className="arch-spec-label">CROSS-VALIDATION</div>
                <div className="arch-spec-value">LOGOO GROUP-AWARE</div>
              </div>
              <div className="arch-spec-col">
                <div className="arch-spec-label">FALSE POSITIVE RATE</div>
                <div className="arch-spec-value">0.00% (0 / 686 TESTED)</div>
              </div>
              <div className="arch-spec-col">
                <div className="arch-spec-label">WORKSPACE ISOLATION</div>
                <div className="arch-spec-value">X-API-KEY TENANT SILOS</div>
              </div>
            </div>
          </div>
        </div>
      </section>
    </div>
  );
}
