import { useEffect, useMemo, useState } from 'react';
import { loadClusters, loadGraph } from '../api/client';
import { useAppStore } from '../store/appStore';

export default function AbuseClustersPage() {
  const clusters = useAppStore((s) => s.clusters);
  const setClusters = useAppStore((s) => s.setClusters);
  const graphNodes = useAppStore((s) => s.graphNodes);
  const graphLinks = useAppStore((s) => s.graphLinks);
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

  useEffect(() => {
    const store = useAppStore.getState();

    if (store.clusters.length === 0 && !store.loading.clusters) {
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
    if (store.graphNodes.length === 0 && !store.loading.graph) {
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
                  nodes={graphNodes}
                  links={graphLinks}
                  clusters={clusters}
                  selectedCluster={selectedCluster}
                  onSelectCluster={setSelectedCluster}
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

function RelationshipGraph({
  nodes,
  links,
  clusters,
  selectedCluster,
  onSelectCluster,
}: {
  nodes: any[];
  links: any[];
  clusters: any[];
  selectedCluster: any;
  onSelectCluster: (c: any) => void;
}) {
  const [hovered, setHovered] = useState<string | null>(null);

  // If a cluster is selected, highlight its nodes & edges
  const activeClusterNodeIds = useMemo(() => {
    if (!selectedCluster) return null;
    const ids = new Set<string>();
    selectedCluster.customers.forEach((c: string) => ids.add(`c_${c}`));
    selectedCluster.entities.forEach((e: string) => ids.add(e));
    return ids;
  }, [selectedCluster]);

  const flaggedIds = useMemo(() => {
    const set = new Set<string>();
    clusters.forEach((c) => {
      if (c.overallRisk === 'high') {
        c.customers.forEach((id: string) => set.add(`c_${id}`));
      }
    });
    return set;
  }, [clusters]);

  // Layout: customers in inner circle, entities in outer circle
  const displayNodes = useMemo(() => nodes.slice(0, 45), [nodes]);
  const displayNodeIds = useMemo(() => new Set(displayNodes.map((n) => n.id)), [displayNodes]);

  const customers = useMemo(() => displayNodes.filter((n) => n.type === 'customer'), [displayNodes]);
  const entities = useMemo(() => displayNodes.filter((n) => n.type !== 'customer'), [displayNodes]);

  const positions = useMemo(() => {
    const pos: Record<string, { x: number; y: number }> = {};
    const cx = 200, cy = 200;
    const innerR = 75, outerR = 145;

    customers.forEach((n, i) => {
      const angle = (i / Math.max(customers.length, 1)) * 2 * Math.PI - Math.PI / 2;
      pos[n.id] = { x: cx + Math.cos(angle) * innerR, y: cy + Math.sin(angle) * innerR };
    });

    entities.forEach((n, i) => {
      const angle = (i / Math.max(entities.length, 1)) * 2 * Math.PI - Math.PI / 2;
      pos[n.id] = { x: cx + Math.cos(angle) * outerR, y: cy + Math.sin(angle) * outerR };
    });

    return pos;
  }, [customers, entities]);

  // Filter links connecting displayed nodes
  const displayLinks = useMemo(() => {
    return links.filter(
      (l) => displayNodeIds.has(l.source) && displayNodeIds.has(l.target)
    ).slice(0, 60);
  }, [links, displayNodeIds]);

  const hoveredNode = displayNodes.find((n) => n.id === hovered);

  return (
    <div style={{ position: 'relative', width: '100%', height: '100%' }}>
      <svg viewBox="0 0 400 400" width="100%" height="100%" style={{ background: '#0A0A0A' }}>
        {/* Render Connecting Edges */}
        {displayLinks.map((l, i) => {
          const p1 = positions[l.source];
          const p2 = positions[l.target];
          if (!p1 || !p2) return null;

          const isHoveredEdge = hovered && (l.source === hovered || l.target === hovered);
          const isClusterEdge =
            activeClusterNodeIds && activeClusterNodeIds.has(l.source) && activeClusterNodeIds.has(l.target);

          const stroke = isHoveredEdge || isClusterEdge ? '#D9391F' : '#282826';
          const strokeWidth = isHoveredEdge || isClusterEdge ? 1.8 : 0.8;
          const opacity = hovered || activeClusterNodeIds ? (isHoveredEdge || isClusterEdge ? 1 : 0.15) : 0.4;

          return (
            <line
              key={`link-${i}`}
              x1={p1.x}
              y1={p1.y}
              x2={p2.x}
              y2={p2.y}
              stroke={stroke}
              strokeWidth={strokeWidth}
              strokeOpacity={opacity}
              style={{ transition: 'all 200ms ease' }}
            />
          );
        })}

        {/* Render Nodes */}
        {displayNodes.map((n) => {
          const pos = positions[n.id] || { x: 200, y: 200 };
          const isCustomer = n.type === 'customer';
          const isFlagged = flaggedIds.has(n.id);
          const isHovered = hovered === n.id;
          const isSelected = activeClusterNodeIds?.has(n.id);

          let opacity = 1;
          if (hovered) {
            const isConnected = displayLinks.some(
              (l) => (l.source === hovered && l.target === n.id) || (l.target === hovered && l.source === n.id)
            );
            opacity = isHovered || isConnected ? 1 : 0.2;
          } else if (activeClusterNodeIds) {
            opacity = isSelected ? 1 : 0.25;
          }

          const nodeColor = isFlagged
            ? '#D9391F'
            : isCustomer
            ? '#FAFAF8'
            : n.type === 'device'
            ? '#EF9F27'
            : n.type === 'payment'
            ? '#1D9E75'
            : '#9BA3AB';

          const nodeRadius = isCustomer ? (isHovered ? 9 : 7) : (isHovered ? 7 : 5);

          return (
            <g
              key={n.id}
              opacity={opacity}
              onMouseEnter={() => setHovered(n.id)}
              onMouseLeave={() => setHovered(null)}
              onClick={() => {
                const foundCluster = clusters.find((c) =>
                  c.customers.some((cid: string) => `c_${cid}` === n.id) || c.entities.includes(n.id)
                );
                if (foundCluster) onSelectCluster(foundCluster);
              }}
              style={{ cursor: 'pointer', transition: 'opacity 200ms ease' }}
            >
              {isCustomer ? (
                <circle cx={pos.x} cy={pos.y} r={nodeRadius} fill={nodeColor} />
              ) : (
                <rect
                  x={pos.x - nodeRadius}
                  y={pos.y - nodeRadius}
                  width={nodeRadius * 2}
                  height={nodeRadius * 2}
                  rx={2}
                  fill={nodeColor}
                />
              )}
              {(isFlagged || isSelected) && (
                <circle
                  cx={pos.x}
                  cy={pos.y}
                  r={nodeRadius + 4}
                  fill="none"
                  stroke="#D9391F"
                  strokeWidth={1.2}
                  opacity={0.7}
                  className={isFlagged ? 'pulse-cluster' : undefined}
                />
              )}
            </g>
          );
        })}

        {/* Legend */}
        <g transform="translate(12, 375)">
          <circle cx={4} cy={0} r={4} fill="#FAFAF8" />
          <text x={12} y={3} fontSize="8" fill="#8A8A88" fontFamily="JetBrains Mono">
            CUSTOMER
          </text>

          <circle cx={70} cy={0} r={4} fill="#D9391F" />
          <text x={78} y={3} fontSize="8" fill="#8A8A88" fontFamily="JetBrains Mono">
            FLAGGED
          </text>

          <rect x={126} y={-4} width={8} height={8} rx={1} fill="#EF9F27" />
          <text x={138} y={3} fontSize="8" fill="#8A8A88" fontFamily="JetBrains Mono">
            DEVICE
          </text>

          <rect x={184} y={-4} width={8} height={8} rx={1} fill="#1D9E75" />
          <text x={196} y={3} fontSize="8" fill="#8A8A88" fontFamily="JetBrains Mono">
            PAYMENT
          </text>
        </g>
      </svg>

      {/* Hover Tooltip Overlay */}
      {hoveredNode && (
        <div
          style={{
            position: 'absolute',
            bottom: 35,
            left: '50%',
            transform: 'translateX(-50%)',
            background: '#1A1A1A',
            border: '1px solid #2E2E2C',
            borderRadius: 6,
            padding: '6px 12px',
            pointerEvents: 'none',
            fontSize: 11,
            color: '#FAFAF8',
            fontFamily: 'JetBrains Mono, monospace',
            boxShadow: '0 4px 12px rgba(0,0,0,0.5)',
            whiteSpace: 'nowrap',
            zIndex: 10,
          }}
        >
          <span style={{ color: hoveredNode.type === 'customer' ? '#FAFAF8' : '#9BA3AB', textTransform: 'uppercase', marginRight: 6 }}>
            [{hoveredNode.type}]
          </span>
          {hoveredNode.label}
        </div>
      )}
    </div>
  );
}
