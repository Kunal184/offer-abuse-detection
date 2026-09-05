import { useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAppStore } from '../store/appStore';
import { loadOverview, loadGraph, loadClusters } from '../api/client';
import MiniPieChart from '../components/shared/MiniPieChart';
import './Overview.css';

export default function OverviewPage() {
  const navigate = useNavigate();
  const overview = useAppStore((s) => s.overview);
  const clusters = useAppStore((s) => s.clusters);
  const loading = useAppStore((s) => s.loading);
  const error = useAppStore((s) => s.error);

  useEffect(() => {
    const { setOverview, setGraph, setClusters, setLoading, setError } = useAppStore.getState();

    setLoading('overview', true);
    loadOverview()
      .then(setOverview)
      .catch((e: unknown) =>
        setError('overview', e instanceof Error ? e.message : String(e))
      )
      .finally(() => setLoading('overview', false));

    setLoading('graph', true);
    loadGraph()
      .then((g) => setGraph(g.nodes, g.links))
      .catch((e: unknown) => {
        setError('graph', e instanceof Error ? e.message : String(e));
      })
      .finally(() => setLoading('graph', false));

    setLoading('clusters', true);
    loadClusters()
      .then((d) => setClusters(d.clusters || []))
      .catch((e: unknown) => {
        setError('clusters', e instanceof Error ? e.message : String(e));
      })
      .finally(() => setLoading('clusters', false));
  }, []);

  const riskDist = overview?.riskDistribution;
  const pieData = riskDist
    ? [
      { name: 'High Risk', value: riskDist.high, color: '#E5341C' },
      { name: 'Medium Watch', value: riskDist.medium, color: '#EF9F27' },
      { name: 'Clear', value: riskDist.clear, color: '#1D9E75' },
    ]
    : [];

  const topCluster = clusters[0];

  const renderLoading = () => (
    <div style={{ height: 100, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
      <span style={{ color: 'var(--text-muted)', fontSize: 13, fontFamily: 'JetBrains Mono, monospace' }}>SCANNING SYSTEM...</span>
    </div>
  );

  const renderError = (err: string) => (
    <div style={{ height: 100, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
      <span style={{ color: '#E5341C', fontSize: 13, fontFamily: 'JetBrains Mono, monospace' }}>SYSTEM ERROR: {err}</span>
    </div>
  );

  return (
    <div className="overview-container">
      {/* Header */}
      <div className="overview-header">
        <div>
          <h1 className="overview-title">SYSTEM OVERVIEW</h1>
          <p className="overview-subtitle">Real-time graph intelligence & offer abuse detection</p>
        </div>
      </div>

      {/* Hero System Verdict Banner */}
      <div className="overview-verdict-banner" onClick={() => navigate('/clusters')}>
        <h2 className="verdict-headline">
          {loading.overview ? 'ANALYZING...' : (overview ? `${overview.abuseClusters} ABUSE RINGS DETECTED` : '—')}
        </h2>
        <div className="verdict-exposure">
          {loading.overview ? '...' : (overview
            ? `₹${Math.round(overview.totalExposure || 0).toLocaleString()} TOTAL EXPOSURE`
            : '—')}
        </div>
        <div className="verdict-meta">
          {loading.overview
            ? 'Running graph extraction...'
            : `${overview?.customersFlagged || 0} FLAGGED CUSTOMERS · XGBOOST GROUP-AWARE MODEL · GRAPH INTELLIGENCE ACTIVE`}
        </div>
      </div>

      {/* 4 KPI Evidence Quads */}
      <div className="overview-kpi-grid">
        <div className="kpi-quad">
          <div className="kpi-quad-label">01 — ANALYZED</div>
          <div className="kpi-quad-val">{loading.overview ? '—' : (overview?.customersAnalyzed ?? 0)}</div>
          <div className="kpi-quad-sub">Total Customer Entities</div>
        </div>

        <div className="kpi-quad">
          <div className="kpi-quad-label">02 — FLAGGED</div>
          <div className="kpi-quad-val highlight-red">
            {loading.overview ? '—' : (overview?.customersFlagged ?? 0)}
          </div>
          <div className="kpi-quad-sub">High Risk Abuse Accounts</div>
        </div>

        <div className="kpi-quad">
          <div className="kpi-quad-label">03 — CLUSTERS</div>
          <div className="kpi-quad-val highlight-red">
            {loading.clusters ? '—' : (clusters.length || (overview?.abuseClusters ?? 0))}
          </div>
          <div className="kpi-quad-sub">Connected Abuse Networks</div>
        </div>

        <div className="kpi-quad">
          <div className="kpi-quad-label">04 — CLEAR / WATCH</div>
          <div className="kpi-quad-val" style={{ color: '#1D9E75' }}>
            {loading.overview ? '—' : (overview?.riskDistribution?.clear ?? 0)}
          </div>
          <div className="kpi-quad-sub">
            {loading.overview ? '' : `${overview?.riskDistribution?.medium ?? 0} Accounts on Watch`}
          </div>
        </div>
      </div>

      {/* Bottom 2 Editorial Cards */}
      <div className="overview-bottom-grid">
        {/* Risk Distribution */}
        <div className="editorial-card">
          <div className="editorial-card-header">
            <h3 className="editorial-card-title">RISK DISTRIBUTION</h3>
            <span style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: 11, color: 'rgba(250,250,248,0.5)' }}>MODEL OUTPUT</span>
          </div>

          {loading.overview
            ? renderLoading()
            : error.overview
              ? renderError(error.overview)
              : riskDist
                ? (
                  <div style={{ display: 'flex', gap: 28, alignItems: 'center', padding: '0.5rem 0' }}>
                    <MiniPieChart data={pieData} size={130} innerRadius={38} outerRadius={60} />
                    <div style={{ flex: 1 }}>
                      {pieData.map((d) => (
                        <div key={d.name} style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 10, borderBottom: '1px solid rgba(244,243,238,0.08)', paddingBottom: 6 }}>
                          <span style={{ fontSize: 12, color: d.color, fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.06em' }}>{d.name}</span>
                          <span style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: 13, color: '#FAFAF8', fontWeight: 600 }}>{d.value}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                )
                : (
                  <div style={{ color: 'var(--text-muted)', fontSize: 12 }}>No data available.</div>
                )}
        </div>

        {/* Highest-Risk Cluster */}
        <div className="editorial-card">
          <div className="editorial-card-header">
            <h3 className="editorial-card-title">HIGHEST-RISK ABUSE RING</h3>
            <button className="editorial-btn-action" onClick={() => navigate('/clusters')}>
              INSPECT ALL RINGS →
            </button>
          </div>

          {loading.clusters
            ? renderLoading()
            : error.clusters && !topCluster
              ? renderError(error.clusters)
              : topCluster
                ? (
                  <div>
                    <div className="cluster-stat-box">
                      <div>
                        <div className="cluster-stat-num">{topCluster.customerCount}</div>
                        <div className="cluster-stat-lbl">ACCOUNTS</div>
                      </div>
                      <div>
                        <div className="cluster-stat-num">{topCluster.flaggedCustomerCount}</div>
                        <div className="cluster-stat-lbl">FLAGGED HIGH RISK</div>
                      </div>
                      <div>
                        <div className="cluster-stat-num" style={{ color: '#FAFAF8' }}>
                          {topCluster.sharedEntities.reduce((a, e) => a + e.count, 0)}
                        </div>
                        <div className="cluster-stat-lbl">SHARED ENTITIES</div>
                      </div>
                    </div>

                    <div style={{ marginTop: '1.25rem' }}>
                      <div style={{ fontSize: 10, fontFamily: 'JetBrains Mono, monospace', color: 'rgba(250,250,248,0.5)', textTransform: 'uppercase', marginBottom: 8, letterSpacing: '0.08em' }}>
                        CONNECTED RELATIONSHIPS:
                      </div>
                      <div>
                        {topCluster.sharedEntities.map((e) => (
                          <span key={e.type} className="entity-pill">
                            {e.type.toUpperCase()}: {e.count}
                          </span>
                        ))}
                      </div>
                    </div>
                  </div>
                )
                : (
                  <div style={{ color: 'var(--text-muted)', fontSize: 12 }}>No cluster evidence extracted yet.</div>
                )}
        </div>
      </div>
    </div>
  );
}
