import { useEffect, useRef, useState } from 'react';
import { loadClusters, loadGraph } from '../api/client';
import { useAppStore } from '../store/appStore';

export default function AbuseClustersPage() {
  const clusters = useAppStore((s) => s.clusters);
  const setClusters = useAppStore((s) => s.setClusters);
  const graphNodes = useAppStore((s) => s.graphNodes);
  const setGraph = useAppStore((s) => s.setGraph);
  const selectedCluster = useAppStore((s) => s.selectedCluster);
  const setSelectedCluster = useAppStore((s) => s.setSelectedCluster);
  const loadingClusters = useAppStore((s) => s.loading.clusters);
  const loadingGraph = useAppStore((s) => s.loading.graph);
  const setLoading = useAppStore((s) => s.setLoading);
  const setError = useAppStore((s) => s.setError);

  // Track local errors independently so a clusters failure doesn't erase graph data
  const [clusterError, setClusterError] = useState<string | null>(null);
  const [graphError, setGraphError] = useState<string | null>(null);

  const initRef = useRef(false);

  useEffect(() => {
    if (initRef.current) return;
    initRef.current = true;

    const store = useAppStore.getState();

    // Load clusters independently — a failure here must NOT clear graph data
    if (store.clusters.length === 0) {
      setLoading('clusters', true);
      setClusterError(null);
      loadClusters()
        .then((c) => {
          setClusters(c.clusters || []);
        })
        .catch((e: unknown) => {
          const msg = e instanceof Error ? e.message : String(e);
          setClusterError(msg);
          setError('clusters', msg);
          // Do NOT call setClusters([]) — that would erase data loaded by Overview
        })
        .finally(() => setLoading('clusters', false));
    }

    // Load graph independently — a failure here must NOT clear cluster data
    if (store.graphNodes.length === 0) {
      setLoading('graph', true);
      setGraphError(null);
      loadGraph()
        .then((g) => {
          setGraph(g.nodes, g.links);
        })
        .catch((e: unknown) => {
          const msg = e instanceof Error ? e.message : String(e);
          setGraphError(msg);
          setError('graph', msg);
          // Do NOT call setGraph([], []) — that clears what was successfully loaded
        })
        .finally(() => setLoading('graph', false));
    }
  }, []);

  const totalFlagged = clusters.reduce((a, c) => a + c.flaggedCustomerCount, 0);
  const totalCustomers = clusters.reduce((a, c) => a + c.customerCount, 0);
  const totalShared = clusters.reduce((a, c) => a + c.sharedEntities.reduce((s, e) => s + e.count, 0), 0);

  const isLoadingAnything = loadingClusters || loadingGraph;

  return (
    <div>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 28 }}>
        <div>
          <h1 style={{ fontSize: 24, fontWeight: 700, color: 'var(--text-primary)', marginBottom: 4 }}>Abuse Clusters</h1>
          <p style={{ fontSize: 13, color: 'var(--text-secondary)' }}>Coordinated abuse networks</p>
        </div>
        <div style={{ display: 'flex', gap: 8 }}>
          <span className="badge badge-high">{clusters.length} clusters</span>
          <span className="badge badge-high">{totalFlagged} flagged</span>
          <span className="badge badge-neutral">{totalShared} shared entities</span>
        </div>
      </div>

      {/* Verdict finding strip */}
      <div className="verdict-card" style={{ marginBottom: 28 }}>
        <div className="verdict-card-label">COORDINATED ABUSE DETECTED</div>
        <div className="verdict-card-statement">
          {isLoadingAnything ? '—' : `${clusters.length} CLUSTERS · ${totalFlagged} FLAGGED ACCOUNTS`}
        </div>
        <div className="verdict-card-meta">
          {totalCustomers} customers · {totalShared} shared entities · graph-based detection
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 24 }}>
        {/* Graph visualization */}
        <div className="card">
          <div className="card-body">
            <h3 className="section-label" style={{ marginBottom: 16 }}>RELATIONSHIP GRAPH</h3>
            <div style={{ background: 'var(--bg-page)', borderRadius: 8, overflow: 'hidden', height: 480 }}>
              {loadingGraph ? (
                <div style={{ height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                  <span style={{ color: 'var(--text-muted)', fontSize: 12 }}>Loading graph...</span>
                </div>
              ) : graphError && graphNodes.length === 0 ? (
                <div style={{ height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', flexDirection: 'column', gap: 8 }}>
                  <span style={{ color: 'var(--risk-high)', fontSize: 12 }}>Graph unavailable</span>
                  <span style={{ color: 'var(--text-muted)', fontSize: 11 }}>{graphError}</span>
                </div>
              ) : graphNodes.length === 0 ? (
                <div style={{ height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                  <span style={{ color: 'var(--text-muted)', fontSize: 12 }}>No graph data</span>
                </div>
              ) : (
                // Key the graph on nodes length so it never remounts due to empty → loaded transitions
                <RelationshipGraph
                  key={`graph-${graphNodes.length}`}
                  nodes={graphNodes.slice(0, 40)}
                  clusters={clusters}
                />
              )}
            </div>
            <p style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 12 }}>
              Steel-gray nodes: customer / entity · Red nodes: flagged cluster members
            </p>
          </div>
        </div>

        {/* Cluster list */}
        <div className="card">
          <div className="card-body">
            <h3 className="section-label" style={{ marginBottom: 16 }}>CLUSTERS</h3>
            {loadingClusters ? (
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: 100 }}>
                <span style={{ color: 'var(--text-muted)', fontSize: 12 }}>Loading clusters...</span>
              </div>
            ) : clusterError && clusters.length === 0 ? (
              <div className="empty-state">
                <div className="empty-state-title" style={{ color: 'var(--risk-high)' }}>Error loading clusters</div>
                <div className="empty-state-desc">{clusterError}</div>
              </div>
            ) : clusters.length === 0 ? (
              <div className="empty-state">
                <div className="empty-state-title">No clusters</div>
                <div className="empty-state-desc">No coordinated abuse detected</div>
              </div>
            ) : (
              <div style={{ display: 'flex', flexDirection: 'column', gap: 12, maxHeight: 480, overflowY: 'auto' }}>
                {clusters.map((cluster) => (
                  <div
                    key={cluster.id}
                    className="card card-sm"
                    style={{
                      cursor: 'pointer',
                      borderColor: selectedCluster?.id === cluster.id ? '#D9391F' : 'var(--border)',
                      background: selectedCluster?.id === cluster.id ? 'var(--bg-elevated)' : 'var(--bg-card)',
                    }}
                    onClick={() => setSelectedCluster(selectedCluster?.id === cluster.id ? null : cluster)}
                  >
                    <div style={{ padding: 14, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                      <div>
                        <div style={{ fontFamily: 'var(--font-mono)', fontSize: 13, fontWeight: 600, color: 'var(--text-primary)' }}>
                          {cluster.id}
                        </div>
                        <div style={{ fontSize: 11, color: 'var(--text-secondary)', marginTop: 2 }}>
                          {cluster.customerCount} accounts · {cluster.flaggedCustomerCount} flagged
                        </div>
                      </div>
                      <div style={{ display: 'flex', gap: 6 }}>
                        <span className={`badge ${cluster.overallRisk === 'high' ? 'badge-high' : cluster.overallRisk === 'medium' ? 'badge-medium' : 'badge-clear'}`}>
                          {cluster.overallRisk}
                        </span>
                      </div>
                    </div>
                    {selectedCluster?.id === cluster.id && (
                      <div className="detail-panel" style={{ borderTop: '1px solid var(--border)' }}>
                        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
                          <div>
                            <div style={{ fontSize: 10, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.06em' }}>Customers</div>
                            <div className="mono" style={{ fontSize: 14, fontWeight: 600 }}>{cluster.customerCount}</div>
                          </div>
                          <div>
                            <div style={{ fontSize: 10, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.06em' }}>Flagged</div>
                            <div className="mono" style={{ fontSize: 14, fontWeight: 600, color: 'var(--risk-high)' }}>{cluster.flaggedCustomerCount}</div>
                          </div>
                          <div>
                            <div style={{ fontSize: 10, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.06em' }}>Shared</div>
                            <div className="mono" style={{ fontSize: 14, fontWeight: 600 }}>
                              {cluster.sharedEntities.reduce((a, e) => a + e.count, 0)}
                            </div>
                          </div>
                          <div>
                            <div style={{ fontSize: 10, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.06em' }}>Risk</div>
                            <div className="mono" style={{ fontSize: 14, fontWeight: 600 }}>{cluster.overallRisk}</div>
                          </div>
                        </div>
                        <div style={{ marginTop: 12 }}>
                          <div style={{ fontSize: 10, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: 6 }}>Shared Entities</div>
                          <div style={{ display: 'flex', gap: 6 }}>
                            {cluster.sharedEntities.map((e) => (
                              <span key={e.type} className="badge badge-neutral" style={{ fontSize: 10 }}>
                                {e.type}: {e.count}
                              </span>
                            ))}
                          </div>
                        </div>
                        <div style={{ marginTop: 12 }}>
                          <div style={{ fontSize: 10, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: 6 }}>Customers</div>
                          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4 }}>
                            {cluster.customers.slice(0, 8).map((c: string) => (
                              <span key={c} className="mono" style={{ fontSize: 10, color: 'var(--text-secondary)', background: 'var(--bg-card)', padding: '2px 6px', borderRadius: 3 }}>
                                {c.slice(0, 6)}…
                              </span>
                            ))}
                          </div>
                        </div>
                      </div>
                    )}
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

/* ─── Relationship graph component ─────────────────────────── */

function RelationshipGraph({ nodes, clusters }: { nodes: any[]; clusters: any[] }) {
  const [hovered, setHovered] = useState<string | null>(null);

  // Build a simple layout: customers in center ring, entities around
  const customers = nodes.filter((n) => n.type === 'customer');
  const entities = nodes.filter((n) => n.type !== 'customer');
  const flaggedIds = new Set<string>();
  clusters.forEach((c) => {
    if (c.overallRisk === 'high') {
      c.customers.forEach((id: string) => flaggedIds.add(`c_${id}`));
    }
  });

  // Position customers in inner circle
  const positions: Record<string, { x: number; y: number }> = {};
  const cx = 200, cy = 200;
  const innerR = 60, outerR = 150;

  customers.forEach((n, i) => {
    const angle = (i / Math.max(customers.length, 1)) * 2 * Math.PI - Math.PI / 2;
    const r = innerR + (i % 3) * 20;
    positions[n.id] = { x: cx + Math.cos(angle) * r, y: cy + Math.sin(angle) * r };
  });

  entities.forEach((n, i) => {
    const angle = (i / Math.max(entities.length, 1)) * 2 * Math.PI - Math.PI / 2;
    positions[n.id] = { x: cx + Math.cos(angle) * outerR, y: cy + Math.sin(angle) * outerR };
  });

  return (
    <svg viewBox="0 0 400 400" width="100%" height="100%" style={{ background: '#0A0A0A' }}>
      {/* Nodes */}
      {nodes.slice(0, 40).map((n) => {
        const pos = positions[n.id] || { x: 200, y: 200 };
        const isCustomer = n.type === 'customer';
        const isFlagged = flaggedIds.has(n.id);
        const isHovered = hovered === n.id;
        const isConnected = hovered && (nodes.find((x) => x.id === hovered)?.type === n.type);
        const opacity = hovered && !isConnected && n.id !== hovered ? 0.25 : 1;

        const nodeColor = isFlagged ? '#D9391F' : isCustomer ? '#FAFAF8' : '#9BA3AB';
        const nodeRadius = isCustomer ? (isHovered ? 10 : 8) : (isHovered ? 7 : 5);
        const pulse = isFlagged ? 'pulse-cluster' : undefined;

        return (
          <g
            key={n.id}
            opacity={opacity}
            onMouseEnter={() => setHovered(n.id)}
            onMouseLeave={() => setHovered(null)}
            style={{ cursor: 'pointer', transition: 'opacity 200ms ease' }}
            className={pulse}
          >
            {isCustomer ? (
              <circle cx={pos.x} cy={pos.y} r={nodeRadius} fill={nodeColor} />
            ) : (
              <rect x={pos.x - nodeRadius} y={pos.y - nodeRadius} width={nodeRadius * 2} height={nodeRadius * 2} rx={2} fill={nodeColor} />
            )}
            {isFlagged && (
              <circle cx={pos.x} cy={pos.y} r={nodeRadius + 4} fill="none" stroke="#D9391F" strokeWidth={1} opacity={0.5} className="pulse-cluster" />
            )}
          </g>
        );
      })}

      {/* Labels */}
      <text x={200} y={20} textAnchor="middle" fontSize="9" fill="#5E5E5C" fontFamily="JetBrains Mono">
        CUSTOMERS → SHARED ENTITIES → OTHER CUSTOMERS
      </text>
    </svg>
  );
}
