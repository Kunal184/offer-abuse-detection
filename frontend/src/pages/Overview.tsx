import { useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAppStore } from '../store/appStore';
import { loadOverview, loadGraph, loadClusters } from '../api/client';
import { PieChart, Pie, Cell, ResponsiveContainer } from 'recharts';

export default function OverviewPage() {
  const navigate = useNavigate();
  const overview = useAppStore((s) => s.overview);
  const clusters = useAppStore((s) => s.clusters);
  const loadingOverview = useAppStore((s) => s.loading.overview);
  const loadingClusters = useAppStore((s) => s.loading.clusters);

  const initRef = useRef(false);

  useEffect(() => {
    if (initRef.current) return;
    initRef.current = true;

    const store = useAppStore.getState();
    const setOverview = store.setOverview;
    const setGraph = store.setGraph;
    const setClusters = store.setClusters;
    const setLoading = store.setLoading;
    const setError = store.setError;

    if (!store.overview) {
      setLoading('overview', true);
      loadOverview()
        .then(setOverview)
        .catch((e: string) => setError('overview', e))
        .finally(() => setLoading('overview', false));
    }
    if (store.graphNodes.length === 0) {
      setLoading('graph', true);
      loadGraph()
        .then((g) => setGraph(g.nodes, g.links))
        .catch(() => setGraph([], []))
        .finally(() => setLoading('graph', false));
    }
    if (store.clusters.length === 0) {
      setLoading('clusters', true);
      loadClusters()
        .then((d) => setClusters(d.clusters || []))
        .catch(() => setClusters([]))
        .finally(() => setLoading('clusters', false));
    }
  }, []);

  const riskDist = overview?.riskDistribution;
  const pieData = riskDist ? [
    { name: 'High', value: riskDist.high, color: '#D9391F' },
    { name: 'Medium', value: riskDist.medium, color: '#EF9F27' },
    { name: 'Clear', value: riskDist.clear, color: '#1D9E75' },
  ] : [];

  const topCluster = clusters[0];

  return (
    <div>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 28 }}>
        <div>
          <h1 style={{ fontSize: 24, fontWeight: 700, color: 'var(--text-primary)', marginBottom: 4 }}>Overview</h1>
          <p style={{ fontSize: 13, color: 'var(--text-secondary)' }}>Real-time abuse detection dashboard</p>
        </div>
        <div style={{ fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--text-muted)' }}>
          SCANNING · {overview?.customersAnalyzed ?? '—'} CUSTOMERS
        </div>
      </div>

      {/* Hero verdict */}
      <div style={{ marginBottom: 28 }}>
        <div className="verdict-card" style={{ cursor: 'pointer' }} onClick={() => navigate('/clusters')}>
          <div className="verdict-card-label">SYSTEM FINDING</div>
          <div className="verdict-card-number">
            {overview ? `${overview.abuseClusters} RINGS` : '—'}
          </div>
          <div className="verdict-card-statement">
            {overview
              ? `₹${Math.round(overview.totalExposure || 0).toLocaleString()} EXPOSED`
              : '—'}
          </div>
          <div className="verdict-card-meta">
            {overview?.customersFlagged || 0} flagged customers · group-aware model · xgboost_groupaware
          </div>
        </div>
      </div>

      {/* KPI grid */}
      <div className="kpi-grid" style={{ marginBottom: 28 }}>
        <div className="kpi-card">
          <div className="kpi-label">CUSTOMERS ANALYZED</div>
          <div className="kpi-value">{loadingOverview ? '—' : (overview?.customersAnalyzed ?? 0)}</div>
          <div className="kpi-sub">
            {overview?.flaggedRatio ? `${(overview.flaggedRatio * 100).toFixed(1)}%` : '0%'} flagged
          </div>
        </div>
        <div className="kpi-card">
          <div className="kpi-label">CUSTOMERS FLAGGED</div>
          <div className="kpi-value" style={{ color: 'var(--risk-high)' }}>
            {loadingOverview ? '—' : (overview?.customersFlagged ?? 0)}
          </div>
          <div className="kpi-sub">High abuse probability</div>
        </div>
        <div className="kpi-card">
          <div className="kpi-label">ABUSE CLUSTERS</div>
          <div className="kpi-value" style={{ color: 'var(--risk-high)' }}>
            {loadingClusters ? '—' : (clusters.length || (overview?.abuseClusters ?? 0))}
          </div>
          <div className="kpi-sub">Coordinated rings detected</div>
        </div>
        <div className="kpi-card">
          <div className="kpi-label">CLEAR / WATCH</div>
          <div className="kpi-value" style={{ color: 'var(--risk-low)' }}>
            {loadingOverview ? '—' : (overview?.riskDistribution?.clear ?? 0)}
          </div>
          <div className="kpi-sub">
            {loadingOverview ? '' : `${overview?.riskDistribution?.medium ?? 0} on watch`}
          </div>
        </div>
      </div>

      {/* Bottom grid */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 24 }}>
        {/* Risk distribution */}
        <div className="card">
          <div className="card-body">
            <h3 className="section-label">RISK DISTRIBUTION</h3>
            {riskDist && (
              <div style={{ display: 'flex', gap: 24, alignItems: 'center' }}>
                <ResponsiveContainer width={140} height={120}>
                  <PieChart>
                    <Pie
                      data={pieData}
                      innerRadius={35}
                      outerRadius={55}
                      dataKey="value"
                      stroke="none"
                    >
                      {pieData.map((entry, i) => (
                        <Cell key={i} fill={entry.color} />
                      ))}
                    </Pie>
                  </PieChart>
                </ResponsiveContainer>
                <div style={{ flex: 1 }}>
                  {pieData.map((d) => (
                    <div key={d.name} style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 6 }}>
                      <span style={{ fontSize: 12, color: d.color, fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.04em' }}>{d.name}</span>
                      <span style={{ fontFamily: 'var(--font-mono)', fontSize: 12, color: 'var(--text-secondary)' }}>{d.value}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}
            {!riskDist && (
              <div style={{ height: 80, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                <span style={{ color: 'var(--text-muted)', fontSize: 12 }}>Loading...</span>
              </div>
            )}
          </div>
        </div>

        {/* Top cluster */}
        <div className="card">
          <div className="card-body">
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
              <h3 className="section-label">HIGHEST-RISK CLUSTER</h3>
              <button className="btn btn-ghost" style={{ fontSize: 11, padding: '4px 10px' }} onClick={() => navigate('/clusters')}>
                View All →
              </button>
            </div>
            {topCluster ? (
              <div>
                <div style={{ display: 'flex', gap: 16, marginBottom: 10 }}>
                  <div>
                    <div style={{ fontFamily: 'var(--font-mono)', fontSize: 22, fontWeight: 600, color: 'var(--risk-high)' }}>
                      {topCluster.customerCount}
                    </div>
                    <div style={{ fontSize: 10, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.06em' }}>Accounts</div>
                  </div>
                  <div>
                    <div style={{ fontFamily: 'var(--font-mono)', fontSize: 22, fontWeight: 600, color: 'var(--risk-high)' }}>
                      {topCluster.flaggedCustomerCount}
                    </div>
                    <div style={{ fontSize: 10, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.06em' }}>Flagged</div>
                  </div>
                  <div>
                    <div style={{ fontFamily: 'var(--font-mono)', fontSize: 22, fontWeight: 600, color: 'var(--text-primary)' }}>
                      {topCluster.sharedEntities.reduce((a, e) => a + e.count, 0)}
                    </div>
                    <div style={{ fontSize: 10, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.06em' }}>Shared Entities</div>
                  </div>
                </div>
                <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
                  {topCluster.sharedEntities.map((e) => (
                    <span key={e.type} className="badge badge-neutral" style={{ fontSize: 10 }}>
                      {e.type}: {e.count}
                    </span>
                  ))}
                </div>
              </div>
            ) : (
              <div style={{ height: 80, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                <span style={{ color: 'var(--text-muted)', fontSize: 12 }}>Loading...</span>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
