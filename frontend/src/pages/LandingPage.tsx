import { Link } from 'react-router-dom';
import './LandingPage.css';

export default function LandingPage() {
  return (
    <div className="landing-viewport">
      {/* Top Header */}
      <header className="landing-header">
        <div className="landing-logo">HEX</div>
        <Link to="/overview" className="landing-action">
          ENTER DASHBOARD <span className="arrow">→</span>
        </Link>
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
    </div>
  );
}
