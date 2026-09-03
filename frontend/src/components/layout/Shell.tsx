import { Outlet, useLocation, useNavigate } from 'react-router-dom';
import { useAppStore } from '../../store/appStore';

const NAV_ITEMS = [
  { label: 'Overview', path: '/overview', icon: (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8">
      <rect x="3" y="3" width="7" height="7" rx="1"/><rect x="14" y="3" width="7" height="7" rx="1"/>
      <rect x="3" y="14" width="7" height="7" rx="1"/><rect x="14" y="14" width="7" height="7" rx="1"/>
    </svg>
  )},
  { label: 'Customers', path: '/customers', icon: (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8">
      <circle cx="8" cy="7" r="3"/><path d="M2 21v-1a5 5 0 0 1 10 0v1"/><circle cx="18" cy="8" r="2.5"/>
      <path d="M16 21v-0.5a4 4 0 0 1 4-4h1"/>
    </svg>
  )},
  { label: 'Abuse Clusters', path: '/clusters', icon: (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8">
      <circle cx="6" cy="6" r="2"/><circle cx="18" cy="6" r="2"/><circle cx="12" cy="18" r="2"/>
      <path d="M7 7l4 4M17 7l-4 4M13 13l-1 3M11 13l1 3"/>
    </svg>
  )},
  { label: 'Activity', path: '/activity', icon: (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8">
      <path d="M13 2L3 14h9l-1 8 10-12h-9l1-8z"/>
    </svg>
  )},
  { label: 'Analytics', path: '/analytics', icon: (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8">
      <path d="M3 3v18h18"/><path d="M7 14l3-4 3 3 4-6"/>
    </svg>
  )},
];

export default function Shell() {
  const location = useLocation();
  const navigate = useNavigate();
  const activeView = useAppStore((s) => s.activeView);
  const setActiveView = useAppStore((s) => s.setActiveView);

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="sidebar-logo">
          <div className="sidebar-logo-name">ABUSE DETECTION</div>
          <div className="sidebar-logo-sub">Offer Risk Console</div>
        </div>
        <nav className="sidebar-nav">
          {NAV_ITEMS.map((item) => {
            const active = location.pathname === item.path || activeView === item.label.toLowerCase();
            return (
              <button
                key={item.path}
                className={`nav-item ${active ? 'active' : ''}`}
                onClick={() => {
                  setActiveView(item.label.toLowerCase());
                  navigate(item.path);
                }}
              >
                <span className="nav-icon">{item.icon}</span>
                <span>{item.label}</span>
              </button>
            );
          })}
        </nav>
        <div className="sidebar-logo" style={{ padding: '16px 20px', fontSize: '10px', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
          MODEL: xgboost_groupaware<br/>v2026-08-31
        </div>
      </aside>

      <div className="main-content">
        <TopBar />
        <div className="page-body">
          <Outlet />
        </div>
      </div>
    </div>
  );
}

function TopBar() {
  return (
    <header className="topbar">
      <div className="topbar-title">
        <strong>Acme Retail</strong> · Merchant Operations
      </div>
    </header>
  );
}