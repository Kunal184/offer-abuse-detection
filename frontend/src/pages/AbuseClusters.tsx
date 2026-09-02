import { useEffect, useMemo, useRef, useState } from 'react';
import * as d3 from 'd3';
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
        })
        .finally(() => setLoading('clusters', false));
    }

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

      <div style={{ display: 'grid', gridTemplateColumns: '1.2fr 0.8fr', gap: 24 }}>
        {/* Graph visualization */}
        <div className="card">
          <div className="card-body" style={{ padding: 16 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
              <h3 className="section-label" style={{ margin: 0 }}>RELATIONSHIP GRAPH</h3>
              {selectedCluster && (
                <button
                  className="btn btn-ghost"
                  style={{ fontSize: 10, padding: '2px 8px' }}
                  onClick={() => setSelectedCluster(null)}
                >
                  Clear Selection ({selectedCluster.id.toUpperCase()}) ✕
                </button>
              )}
            </div>
            <div style={{ background: '#0A0A0A', borderRadius: 8, overflow: 'hidden', height: 480, position: 'relative' }}>
              {loadingGraph ? (
                <div style={{ height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                  <span style={{ color: 'var(--text-muted)', fontSize: 12 }}>Loading graph data...</span>
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
                <ForceRelationshipGraph
                  nodes={graphNodes}
                  links={graphLinks}
                  clusters={clusters}
                  selectedCluster={selectedCluster}
                  onSelectCluster={setSelectedCluster}
                />
              )}
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: 10 }}>
              <p style={{ fontSize: 11, color: 'var(--text-muted)', margin: 0 }}>
                Circles: Customers · Squares: Shared Entities · Drag to move · Scroll to zoom
              </p>
            </div>
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
              <div style={{ display: 'flex', flexDirection: 'column', gap: 10, maxHeight: 440, overflowY: 'auto', paddingRight: 4 }}>
                {clusters.map((cluster) => {
                  const isSelected = selectedCluster?.id === cluster.id;
                  const totalSharedCount = cluster.sharedEntities.reduce((a, e) => a + e.count, 0);

                  return (
                    <div
                      key={cluster.id}
                      style={{
                        background: isSelected ? '#1E1E1C' : '#141414',
                        border: `1px solid ${isSelected ? '#D9391F' : '#282826'}`,
                        borderRadius: 8,
                        padding: '12px 16px',
                        cursor: 'pointer',
                        transition: 'all 150ms ease',
                      }}
                      onClick={() => setSelectedCluster(isSelected ? null : cluster)}
                    >
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                        <div>
                          <div style={{ fontFamily: 'var(--font-mono)', fontSize: 13, fontWeight: 700, color: 'var(--text-primary)' }}>
                            {cluster.id.toUpperCase()}
                          </div>
                          <div style={{ fontSize: 11, color: 'var(--text-secondary)', marginTop: 3 }}>
                            <span style={{ color: cluster.flaggedCustomerCount > 0 ? '#D9391F' : 'inherit', fontWeight: 600 }}>
                              {cluster.flaggedCustomerCount} flagged
                            </span>
                            {' · '}{cluster.customerCount} accounts · {totalSharedCount} shared
                          </div>
                        </div>
                        <span className={`badge ${cluster.overallRisk === 'high' ? 'badge-high' : cluster.overallRisk === 'medium' ? 'badge-medium' : 'badge-clear'}`}>
                          {cluster.overallRisk}
                        </span>
                      </div>

                      {isSelected && (
                        <div style={{ marginTop: 14, paddingTop: 12, borderTop: '1px solid #2E2E2C' }}>
                          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 8, marginBottom: 12 }}>
                            <div style={{ background: '#0A0A0A', padding: '8px 10px', borderRadius: 4 }}>
                              <div style={{ fontSize: 9, color: 'var(--text-muted)', textTransform: 'uppercase' }}>Accounts</div>
                              <div className="mono" style={{ fontSize: 14, fontWeight: 700, color: 'var(--text-primary)' }}>{cluster.customerCount}</div>
                            </div>
                            <div style={{ background: '#0A0A0A', padding: '8px 10px', borderRadius: 4 }}>
                              <div style={{ fontSize: 9, color: 'var(--text-muted)', textTransform: 'uppercase' }}>Flagged</div>
                              <div className="mono" style={{ fontSize: 14, fontWeight: 700, color: '#D9391F' }}>{cluster.flaggedCustomerCount}</div>
                            </div>
                            <div style={{ background: '#0A0A0A', padding: '8px 10px', borderRadius: 4 }}>
                              <div style={{ fontSize: 9, color: 'var(--text-muted)', textTransform: 'uppercase' }}>Shared</div>
                              <div className="mono" style={{ fontSize: 14, fontWeight: 700, color: 'var(--text-primary)' }}>{totalSharedCount}</div>
                            </div>
                          </div>

                          <div style={{ marginBottom: 10 }}>
                            <div style={{ fontSize: 9, color: 'var(--text-muted)', textTransform: 'uppercase', marginBottom: 4 }}>Shared Entities</div>
                            <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
                              {cluster.sharedEntities.filter((e) => e.count > 0).map((e) => (
                                <span key={e.type} className="badge badge-neutral" style={{ fontSize: 10 }}>
                                  {e.type}: {e.count}
                                </span>
                              ))}
                            </div>
                          </div>

                          <div>
                            <div style={{ fontSize: 9, color: 'var(--text-muted)', textTransform: 'uppercase', marginBottom: 4 }}>Member Accounts</div>
                            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4 }}>
                              {cluster.customers.slice(0, 8).map((c: string) => (
                                <span key={c} className="mono" style={{ fontSize: 10, color: 'var(--text-secondary)', background: '#0A0A0A', padding: '2px 6px', borderRadius: 3, border: '1px solid #242422' }}>
                                  {c.slice(0, 8)}…
                                </span>
                              ))}
                            </div>
                          </div>
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

/* ─── D3 Force-Directed Relationship Graph Component ─────────────────────────── */

interface GraphNodeItem {
  id: string;
  type: string;
  label: string;
  x?: number;
  y?: number;
  vx?: number;
  vy?: number;
  fx?: number | null;
  fy?: number | null;
}

interface GraphLinkItem {
  source: any;
  target: any;
  sourceType: string;
  targetType: string;
}

function ForceRelationshipGraph({
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
  const svgRef = useRef<SVGSVGElement | null>(null);
  const containerRef = useRef<SVGGElement | null>(null);

  const [activeTooltip, setActiveTooltip] = useState<{ x: number; y: number; node: GraphNodeItem } | null>(null);

  // Set of flagged customer node IDs
  const flaggedIds = useMemo(() => {
    const set = new Set<string>();
    clusters.forEach((c) => {
      if (c.overallRisk === 'high') {
        c.customers.forEach((id: string) => set.add(`c_${id}`));
      }
    });
    return set;
  }, [clusters]);

  // Extract relevant subgraph for simulation
  const { subNodes, subLinks } = useMemo(() => {
    let targetNodeIds: Set<string>;

    if (selectedCluster) {
      targetNodeIds = new Set<string>();
      selectedCluster.customers.forEach((cid: string) => targetNodeIds.add(`c_${cid}`));
      selectedCluster.entities.forEach((eid: string) => targetNodeIds.add(eid));
    } else {
      // Show nodes from top clusters
      targetNodeIds = new Set<string>();
      const topClusters = clusters.slice(0, 8);
      topClusters.forEach((c) => {
        c.customers.forEach((cid: string) => targetNodeIds.add(`c_${cid}`));
        c.entities.forEach((eid: string) => targetNodeIds.add(eid));
      });
      // Fallback if no clusters
      if (targetNodeIds.size === 0) {
        nodes.slice(0, 100).forEach((n) => targetNodeIds.add(n.id));
      }
    }

    const filteredNodes: GraphNodeItem[] = nodes
      .filter((n) => targetNodeIds.has(n.id))
      .map((n) => ({ ...n }));

    const filteredNodeIds = new Set(filteredNodes.map((n) => n.id));

    const filteredLinks: GraphLinkItem[] = links
      .filter((l) => filteredNodeIds.has(l.source) && filteredNodeIds.has(l.target))
      .map((l) => ({ ...l }));

    return { subNodes: filteredNodes, subLinks: filteredLinks };
  }, [nodes, links, clusters, selectedCluster]);

  useEffect(() => {
    if (!svgRef.current || !containerRef.current || subNodes.length === 0) return;

    const width = 500;
    const height = 480;

    const svg = d3.select(svgRef.current);
    const container = d3.select(containerRef.current);

    // Clear previous rendering
    container.selectAll('*').remove();

    // D3 Zoom setup
    const zoomBehavior = d3
      .zoom<SVGSVGElement, unknown>()
      .scaleExtent([0.3, 4])
      .on('zoom', (event) => {
        container.attr('transform', event.transform);
      });

    svg.call(zoomBehavior);
    svg.call(zoomBehavior.transform, d3.zoomIdentity.translate(0, 0).scale(1));

    // Force simulation
    const simulation = d3
      .forceSimulation<GraphNodeItem>(subNodes)
      .force(
        'link',
        d3
          .forceLink<GraphNodeItem, GraphLinkItem>(subLinks)
          .id((d) => d.id)
          .distance(60)
      )
      .force('charge', d3.forceManyBody().strength(-160))
      .force('center', d3.forceCenter(width / 2, height / 2))
      .force('collide', d3.forceCollide(20));

    // Draw links
    const linkGroup = container.append('g').attr('class', 'links');

    const linkElements = linkGroup
      .selectAll<SVGLineElement, GraphLinkItem>('line')
      .data(subLinks)
      .enter()
      .append('line')
      .attr('stroke', (d) => {
        const sId = typeof d.source === 'object' ? d.source.id : d.source;
        const tId = typeof d.target === 'object' ? d.target.id : d.target;
        const isFlaggedEdge = flaggedIds.has(sId) || flaggedIds.has(tId);
        return isFlaggedEdge ? '#D9391F' : '#2E2E2C';
      })
      .attr('stroke-width', (d) => {
        const sId = typeof d.source === 'object' ? d.source.id : d.source;
        const tId = typeof d.target === 'object' ? d.target.id : d.target;
        return flaggedIds.has(sId) || flaggedIds.has(tId) ? 1.8 : 1.0;
      })
      .attr('stroke-opacity', 0.6)
      .style('transition', 'stroke-opacity 200ms ease');

    // Draw nodes
    const nodeGroup = container.append('g').attr('class', 'nodes');

    const nodeElements = nodeGroup
      .selectAll<SVGGElement, GraphNodeItem>('g')
      .data(subNodes)
      .enter()
      .append('g')
      .attr('cursor', 'pointer')
      .style('opacity', 0)
      .call(
        d3
          .drag<SVGGElement, GraphNodeItem>()
          .on('start', (event, d) => {
            if (!event.active) simulation.alphaTarget(0.3).restart();
            d.fx = d.x;
            d.fy = d.y;
          })
          .on('drag', (event, d) => {
            d.fx = event.x;
            d.fy = event.y;
          })
          .on('end', (event, d) => {
            if (!event.active) simulation.alphaTarget(0);
            d.fx = null;
            d.fy = null;
          })
      );

    // Fade-in stagger animation for nodes
    nodeElements
      .transition()
      .duration(400)
      .delay((_, i) => Math.min(i * 15, 600))
      .style('opacity', 1);

    // Node shape and style rendering
    nodeElements.each(function (d) {
      const g = d3.select(this);
      const isCustomer = d.type === 'customer';
      const isFlagged = flaggedIds.has(d.id);

      const color = isFlagged
        ? '#D9391F'
        : isCustomer
        ? '#FAFAF8'
        : d.type === 'device'
        ? '#EF9F27'
        : d.type === 'payment'
        ? '#1D9E75'
        : '#9BA3AB';

      // Pulse halo for flagged nodes
      if (isFlagged) {
        g.append('circle')
          .attr('r', 12)
          .attr('fill', 'none')
          .attr('stroke', '#D9391F')
          .attr('stroke-width', 1.5)
          .attr('opacity', 0.6)
          .attr('class', 'pulse-cluster');
      }

      if (isCustomer) {
        g.append('circle')
          .attr('r', 7)
          .attr('fill', color);
      } else {
        const size = d.type === 'device' || d.type === 'payment' ? 12 : 10;
        g.append('rect')
          .attr('x', -size / 2)
          .attr('y', -size / 2)
          .attr('width', size)
          .attr('height', size)
          .attr('rx', 2)
          .attr('fill', color);
      }

      // Label text below node
      g.append('text')
        .text(d.type === 'customer' ? d.id.replace('c_', '').slice(0, 6) : d.label)
        .attr('dy', 18)
        .attr('text-anchor', 'middle')
        .attr('font-size', 9)
        .attr('fill', '#8A8A88')
        .attr('font-family', 'JetBrains Mono, monospace');
    });

    // Hover interactions
    nodeElements
      .on('mouseenter', (event, d) => {
        const bounds = svgRef.current?.getBoundingClientRect();
        if (bounds) {
          setActiveTooltip({
            x: event.clientX - bounds.left,
            y: event.clientY - bounds.top - 40,
            node: d,
          });
        }

        // Highlight connected edges & nodes
        const connectedNodeIds = new Set<string>([d.id]);
        linkElements.each(function (l) {
          const sId = typeof l.source === 'object' ? l.source.id : l.source;
          const tId = typeof l.target === 'object' ? l.target.id : l.target;
          if (sId === d.id) connectedNodeIds.add(tId);
          if (tId === d.id) connectedNodeIds.add(sId);
        });

        nodeElements.style('opacity', (n) => (connectedNodeIds.has(n.id) ? 1 : 0.25));
        linkElements.style('stroke-opacity', (l) => {
          const sId = typeof l.source === 'object' ? l.source.id : l.source;
          const tId = typeof l.target === 'object' ? l.target.id : l.target;
          return sId === d.id || tId === d.id ? 1 : 0.15;
        });
      })
      .on('mouseleave', () => {
        setActiveTooltip(null);
        nodeElements.style('opacity', 1);
        linkElements.style('stroke-opacity', 0.6);
      })
      .on('click', (_, d) => {
        const foundCluster = clusters.find((c) =>
          c.customers.some((cid: string) => `c_${cid}` === d.id) || c.entities.includes(d.id)
        );
        if (foundCluster) onSelectCluster(foundCluster);
      });

    // Simulation tick handler
    simulation.on('tick', () => {
      linkElements
        .attr('x1', (d) => (d.source as GraphNodeItem).x ?? 0)
        .attr('y1', (d) => (d.source as GraphNodeItem).y ?? 0)
        .attr('x2', (d) => (d.target as GraphNodeItem).x ?? 0)
        .attr('y2', (d) => (d.target as GraphNodeItem).y ?? 0);

      nodeElements.attr('transform', (d) => `translate(${d.x ?? 0},${d.y ?? 0})`);
    });

    return () => {
      simulation.stop();
    };
  }, [subNodes, subLinks, flaggedIds, clusters, onSelectCluster]);

  return (
    <div style={{ position: 'relative', width: '100%', height: '100%' }}>
      <svg
        ref={svgRef}
        viewBox="0 0 500 480"
        width="100%"
        height="100%"
        style={{ background: '#0A0A0A', display: 'block' }}
      >
        <g ref={containerRef} />

        {/* Legend Overlay */}
        <g transform="translate(14, 455)">
          <circle cx={4} cy={0} r={4} fill="#FAFAF8" />
          <text x={12} y={3} fontSize="9" fill="#8A8A88" fontFamily="JetBrains Mono">
            CUSTOMER
          </text>

          <circle cx={76} cy={0} r={4} fill="#D9391F" />
          <text x={84} y={3} fontSize="9" fill="#8A8A88" fontFamily="JetBrains Mono">
            FLAGGED
          </text>

          <rect x={138} y={-4} width={8} height={8} rx={1} fill="#EF9F27" />
          <text x={150} y={3} fontSize="9" fill="#8A8A88" fontFamily="JetBrains Mono">
            DEVICE
          </text>

          <rect x={198} y={-4} width={8} height={8} rx={1} fill="#1D9E75" />
          <text x={210} y={3} fontSize="9" fill="#8A8A88" fontFamily="JetBrains Mono">
            PAYMENT
          </text>

          <rect x={262} y={-4} width={8} height={8} rx={1} fill="#9BA3AB" />
          <text x={274} y={3} fontSize="9" fill="#8A8A88" fontFamily="JetBrains Mono">
            ADDRESS/IP
          </text>
        </g>
      </svg>

      {/* Floating Tooltip */}
      {activeTooltip && (
        <div
          style={{
            position: 'absolute',
            top: activeTooltip.y,
            left: activeTooltip.x,
            transform: 'translateX(-50%)',
            background: '#1A1A1A',
            border: '1px solid #2E2E2C',
            borderRadius: 6,
            padding: '6px 12px',
            pointerEvents: 'none',
            fontSize: 11,
            color: '#FAFAF8',
            fontFamily: 'JetBrains Mono, monospace',
            boxShadow: '0 4px 14px rgba(0,0,0,0.6)',
            whiteSpace: 'nowrap',
            zIndex: 20,
          }}
        >
          <span style={{ color: activeTooltip.node.type === 'customer' ? '#FAFAF8' : '#EF9F27', textTransform: 'uppercase', marginRight: 6 }}>
            [{activeTooltip.node.type}]
          </span>
          {activeTooltip.node.id}
        </div>
      )}
    </div>
  );
}
