import { useState } from 'react';
import { Outlet, useLocation, useNavigate } from 'react-router-dom';
import { useAppStore } from '../../store/appStore';
import { resetDatabase } from '../../api/client';

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
  { label: 'Settings', path: '/settings', icon: (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8">
      <circle cx="12" cy="12" r="3"/>
      <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"/>
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
  const [resetting, setResetting] = useState(false);

  const handleReset = async () => {
    if (!window.confirm('Reset database state and re-seed clean baseline dataset (1,000 customers across 21 abuse rings)?')) {
      return;
    }
    setResetting(true);
    try {
      await resetDatabase();
      window.location.reload();
    } catch (err) {
      alert('Failed to reset database: ' + err);
      setResetting(false);
    }
  };

  return (
    <header className="topbar" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
      <div className="topbar-title">
        <strong>Acme Retail</strong> · Merchant Operations
      </div>
      <button
        className="btn btn-secondary"
        style={{ fontSize: 11, padding: '4px 10px', display: 'flex', alignItems: 'center', gap: 6, cursor: resetting ? 'wait' : 'pointer' }}
        onClick={handleReset}
        disabled={resetting}
        title="Reset database to clean seed baseline dataset"
      >
        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8"/>
          <path d="M3 3v5h5"/>
        </svg>
        <span>{resetting ? 'Resetting Baseline...' : 'Reset Baseline Data'}</span>
      </button>
    </header>
  );
}