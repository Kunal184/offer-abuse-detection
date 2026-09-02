import { useEffect, useState } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { useAppStore, getRiskLevel } from '../store/appStore';
import { loadAllData, loadCustomerPrediction, scoreCustomer } from '../api/client';
import RiskBadge from '../components/shared/RiskBadge';
import VerdictCard, { AnimatedNumber } from '../components/shared/VerdictCard';
import type {
  CustomerRaw,
  OrderRaw,
  RedemptionRaw,
  DeviceRaw,
  AddressRaw,
  PaymentRaw,
  IpRaw,
  GraphNode,
  GraphLink,
  EntityType,
} from '../types/index';

const SIGNAL_LABELS: Record<string, { label: string; description: string }> = {
  account_age_days: { label: 'Account Age', description: 'Days since account creation' },
  order_count: { label: 'Order Count', description: 'Total orders placed' },
  total_spend: { label: 'Total Spend', description: 'Lifetime spending' },
  average_spend: { label: 'Average Spend', description: 'Mean order value' },
  time_to_first_order_hours: { label: 'Time to First Order', description: 'Hours from signup to first order' },
  redemption_count: { label: 'Redemption Count', description: 'Offers redeemed' },
  time_to_first_redemption_hours: { label: 'Time to First Redemption', description: 'Hours from signup to first redemption' },
  order_redemption_rate: { label: 'Redemption Rate', description: 'Orders with offers / total orders' },
  max_device_user_count: { label: 'Shared Device Count', description: 'Max users on same device' },
  max_address_user_count: { label: 'Shared Address Count', description: 'Max users at same address' },
  max_payment_user_count: { label: 'Shared Payment Count', description: 'Max users on same payment' },
  max_ip_user_count: { label: 'Shared IP Count', description: 'Max users on same IP' },
  unique_connected_customers: { label: 'Connected Customers', description: 'Customers sharing entities' },
  avg_entity_degree: { label: 'Avg Entity Degree', description: 'Mean connections per entity' },
  max_entity_degree: { label: 'Max Entity Degree', description: 'Max connections on single entity' },
  cluster_size: { label: 'Cluster Size', description: 'Customers in same network' },
};

export default function CustomerDetailPage() {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const demoData = useAppStore((s) => s.demoData);
  const setDemoData = useAppStore((s) => s.setDemoData);
  const setSelectedCustomer = useAppStore((s) => s.setSelectedCustomer);

  const [customer, setCustomer] = useState<CustomerRaw | null>(null);
  const [prediction, setPrediction] = useState<any | null>(null);
  const [orders, setOrders] = useState<OrderRaw[]>([]);
  const [redemptions, setRedemptions] = useState<RedemptionRaw[]>([]);
  const [graphData, setGraphData] = useState<{ nodes: GraphNode[]; links: GraphLink[] }>({ nodes: [], links: [] });
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!demoData) {
      loadAllData().then(setDemoData);
    }
  }, []);

  useEffect(() => {
    if (!id || !demoData) return;

    const found = demoData.customers.find((c: CustomerRaw) => c.customer_id === id);
    if (!found) {
      setError('Customer not found');
      setLoading(false);
      return;
    }

    setCustomer(found);
    setOrders(demoData.orders.filter((o: OrderRaw) => o.customer_id === id));
    setRedemptions(demoData.redemptions.filter((r: RedemptionRaw) => r.customer_id === id));

    // Build local graph
    const nodes: GraphNode[] = [];
    const links: GraphLink[] = [];
    const nodeIds = new Set<string>();
    const customerId = `c_${id}`;
    nodes.push({ id: customerId, type: 'customer', label: found.name, index: 0 });
    nodeIds.add(customerId);

    const addEntity = (entId: string, type: EntityType) => {
      const eid = `${type}_${entId}`;
      if (!nodeIds.has(eid)) {
        let label = entId.slice(0, 12);
        if (type === 'device') label = `Device ${entId.slice(0, 6)}`;
        else if (type === 'address') label = `Addr ${entId.slice(0, 6)}`;
        else if (type === 'payment') label = `Pay ${entId.slice(0, 6)}`;
        else if (type === 'ip') label = entId;
        nodes.push({ id: eid, type, label, index: nodes.length });
        nodeIds.add(eid);
      }
      links.push({ source: customerId, target: eid, sourceType: 'customer', targetType: type });
    };

    demoData.devices.filter((d: DeviceRaw) => d.customer_id === id).forEach((d: DeviceRaw) => {
      addEntity(d.device_id, 'device');
      // Add 2-hop neighbors
      const otherCustomers = demoData.devices.filter((od: DeviceRaw) => od.device_id === d.device_id && od.customer_id !== id);
      otherCustomers.slice(0, 3).forEach((oc: DeviceRaw) => {
        const c2Id = `c_${oc.customer_id}`;
        if (!nodeIds.has(c2Id)) {
          const c2 = demoData.customers.find((c: CustomerRaw) => c.customer_id === oc.customer_id);
          nodes.push({ id: c2Id, type: 'customer', label: c2?.name || oc.customer_id.slice(0, 8), index: nodes.length });
          nodeIds.add(c2Id);
        }
        links.push({ source: `device_${d.device_id}`, target: c2Id, sourceType: 'device', targetType: 'customer' });
      });
    });

    demoData.addresses.filter((a: AddressRaw) => a.customer_id === id).slice(0, 2).forEach((a: AddressRaw) => {
      addEntity(a.address_id, 'address');
    });

    demoData.payments.filter((p: PaymentRaw) => p.customer_id === id).slice(0, 2).forEach((p: PaymentRaw) => {
      addEntity(p.payment_id, 'payment');
    });

    demoData.ips.filter((i: IpRaw) => i.customer_id === id).slice(0, 2).forEach((i: IpRaw) => {
      addEntity(i.ip_address, 'ip');
    });

    setGraphData({ nodes, links });

    // Fetch live prediction + SHAP explanation from backend GET /v1/predictions/{id}?explain=true
    loadCustomerPrediction(id, true)
      .then((p) => {
        setPrediction(p);
        setSelectedCustomer({
          customer_id: id,
          name: found.name,
          email: found.email,
          phone: found.phone,
          created_at: found.created_at,
          risk: getRiskLevel(p.abuse_probability),
          abuse_probability: p.abuse_probability,
          cluster_size: p.feature_snapshot.cluster_size,
          connected_customers: p.feature_snapshot.unique_connected_customers,
          last_activity: found.created_at,
          status: p.predicted_label === 1 ? 'flagged' : 'active',
          features: p.feature_snapshot as any,
          prediction: p,
        });
      })
      .catch(() => {
        scoreCustomer(id, demoData).then((p) => setPrediction(p));
      })
      .finally(() => setLoading(false));
  }, [id, demoData]);

  if (loading) {
    return (
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', minHeight: 400 }}>
        <div style={{ color: 'var(--text-muted)', fontSize: 12, fontFamily: 'var(--font-mono)' }}>SCORING CUSTOMER & GENERATING SHAP EXPLANATION...</div>
      </div>
    );
  }

  if (error || !customer) {
    return (
      <div>
        <button className="btn btn-ghost" style={{ marginBottom: 16 }} onClick={() => navigate('/customers')}>
          ← Back to Customers
        </button>
        <div className="empty-state">
          <div className="empty-state-title">Error</div>
          <div className="empty-state-desc">{error || 'Customer not found'}</div>
        </div>
      </div>
    );
  }

  const risk = prediction ? getRiskLevel(prediction.abuse_probability) : 'clear';
  const riskStatement = risk === 'high' ? 'COORDINATED ACTIVITY DETECTED' : risk === 'medium' ? 'UNDER OBSERVATION' : 'NO ABNORMAL PATTERN';
  const initials = customer.name.split(' ').map((n) => n[0]).join('').slice(0, 2).toUpperCase();

  return (
    <div>
      <button className="btn btn-ghost" style={{ marginBottom: 20 }} onClick={() => navigate('/customers')}>
        ← Back to Customers
      </button>

      {/* Header */}
      <div className="customer-header">
        <div className={`customer-avatar ${risk === 'high' ? 'flagged' : ''}`} style={{ width: 48, height: 48, fontSize: 16 }}>
          {initials}
        </div>
        <div className="customer-header-info">
          <h2>{customer.name}</h2>
          <div className="mono" style={{ marginTop: 2 }}>{customer.customer_id}</div>
        </div>
        <div style={{ marginLeft: 'auto' }}>
          <RiskBadge level={risk} />
        </div>
      </div>

      {/* Verdict card */}
      <div style={{ marginBottom: 28 }}>
        <VerdictCard
          number={
            prediction
              ? <AnimatedNumber target={prediction.abuse_probability * 100} decimals={1} suffix="%" />
              : '—'
          }
          statement={riskStatement}
          label="ABUSE PROBABILITY"
          meta={prediction ? `Model: ${prediction.model_name} · Threshold: ${prediction.decision_threshold} · Label: ${prediction.predicted_label} (${prediction.predicted_label === 1 ? 'FLAGGED' : 'CLEAR'}) · Version: ${prediction.model_version}` : ''}
        />
      </div>

      {/* SHAP ML Explanation */}
      {prediction?.explanation && (
        <div className="card" style={{ marginBottom: 24 }}>
          <div className="card-body">
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', marginBottom: 12 }}>
              <h2>Tree SHAP ML Explanation</h2>
              <span style={{ fontSize: 11, color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
                Base Value: {prediction.explanation.base_value.toFixed(4)} log-odds
              </span>
            </div>
            <p style={{ fontSize: 12, color: 'var(--text-secondary)', marginBottom: 16 }}>
              Feature attribution computed from the frozen XGBoost model using Tree SHAP marginal contributions.
            </p>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 20 }}>
              {/* Positive Risk Contributors */}
              <div style={{ background: 'var(--bg-page)', borderRadius: 6, padding: 14 }}>
                <div style={{ fontSize: 12, fontWeight: 700, color: '#D9391F', marginBottom: 10, display: 'flex', alignItems: 'center', gap: 6 }}>
                  <span>▲ Risk Increasing Contributors (Positive SHAP)</span>
                </div>
                {prediction.explanation.top_positive_contributors.length === 0 ? (
                  <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>No positive risk factors</div>
                ) : (
                  prediction.explanation.top_positive_contributors.map((c: any) => (
                    <div key={c.feature_name} style={{ marginBottom: 10, fontSize: 12 }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 2 }}>
                        <span style={{ fontWeight: 600, color: 'var(--text-primary)' }}>
                          {SIGNAL_LABELS[c.feature_name]?.label || c.feature_name}
                        </span>
                        <span style={{ fontFamily: 'var(--font-mono)', fontWeight: 700, color: '#D9391F' }}>
                          +{c.shap_value.toFixed(4)}
                        </span>
                      </div>
                      <div style={{ fontSize: 11, color: 'var(--text-secondary)' }}>
                        Value: <span className="mono">{c.feature_value}</span> · {c.impact}
                      </div>
                    </div>
                  ))
                )}
              </div>

              {/* Negative Risk Contributors */}
              <div style={{ background: 'var(--bg-page)', borderRadius: 6, padding: 14 }}>
                <div style={{ fontSize: 12, fontWeight: 700, color: '#2E7D32', marginBottom: 10, display: 'flex', alignItems: 'center', gap: 6 }}>
                  <span>▼ Risk Decreasing Contributors (Negative SHAP)</span>
                </div>
                {prediction.explanation.top_negative_contributors.length === 0 ? (
                  <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>No negative risk factors</div>
                ) : (
                  prediction.explanation.top_negative_contributors.map((c: any) => (
                    <div key={c.feature_name} style={{ marginBottom: 10, fontSize: 12 }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 2 }}>
                        <span style={{ fontWeight: 600, color: 'var(--text-primary)' }}>
                          {SIGNAL_LABELS[c.feature_name]?.label || c.feature_name}
                        </span>
                        <span style={{ fontFamily: 'var(--font-mono)', fontWeight: 700, color: '#2E7D32' }}>
                          {c.shap_value.toFixed(4)}
                        </span>
                      </div>
                      <div style={{ fontSize: 11, color: 'var(--text-secondary)' }}>
                        Value: <span className="mono">{c.feature_value}</span> · {c.impact}
                      </div>
                    </div>
                  ))
                )}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Why flagged - feature signals */}
      {prediction && (
        <div className="card" style={{ marginBottom: 24 }}>
          <div className="card-body">
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', marginBottom: 12 }}>
              <h2>Investigation Signals</h2>
              <span style={{ fontSize: 11, color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
                16 features observed at scoring time
              </span>
            </div>
            <p style={{ fontSize: 12, color: 'var(--text-secondary)', marginBottom: 16 }}>
              Observed customer signals available to the model. These are model features, not causal explanations.
            </p>
            <div className="signal-grid">
              {Object.entries(prediction.feature_snapshot).map(([key, value]) => {
                const meta = SIGNAL_LABELS[key] || { label: key, description: '' };
                const display = formatSignalValue(key, value as number);
                return (
                  <div className="signal-item" key={key}>
                    <div className="signal-label">{meta.label}</div>
                    <div className="signal-value">
                      {display.value}
                      {display.unit && <span className="signal-unit">{display.unit}</span>}
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      )}

      {/* Local graph + activity side by side */}
      <div style={{ display: 'grid', gridTemplateColumns: '1.2fr 1fr', gap: 24, marginBottom: 24 }}>
        {/* Local relationship graph */}
        <div className="card">
          <div className="card-body">
            <h2 style={{ marginBottom: 12 }}>Local Relationship Graph</h2>
            <p style={{ fontSize: 12, color: 'var(--text-secondary)', marginBottom: 16 }}>
              Customer → shared entities → connected customers
            </p>
            <LocalGraphView nodes={graphData.nodes} links={graphData.links} />
          </div>
        </div>

        {/* Risk + verdict breakdown */}
        <div>
          <div className="card" style={{ marginBottom: 16 }}>
            <div className="card-body">
              <h3 className="section-label">RELATIONSHIP EVIDENCE</h3>
              {prediction && (
                <>
                  <div style={{ marginTop: 8 }}>
                    <div className="metric-row">
                      <span className="metric-name">Connected customers</span>
                      <span className="metric-value mono">{prediction.feature_snapshot.unique_connected_customers || 0}</span>
                    </div>
                    <div className="metric-row">
                      <span className="metric-name">Shared devices</span>
                      <span className="metric-value mono">{prediction.feature_snapshot.max_device_user_count || 0}</span>
                    </div>
                    <div className="metric-row">
                      <span className="metric-name">Shared addresses</span>
                      <span className="metric-value mono">{prediction.feature_snapshot.max_address_user_count || 0}</span>
                    </div>
                    <div className="metric-row">
                      <span className="metric-name">Shared payments</span>
                      <span className="metric-value mono">{prediction.feature_snapshot.max_payment_user_count || 0}</span>
                    </div>
                    <div className="metric-row">
                      <span className="metric-name">Shared IPs</span>
                      <span className="metric-value mono">{prediction.feature_snapshot.max_ip_user_count || 0}</span>
                    </div>
                    <div className="metric-row">
                      <span className="metric-name">Cluster size</span>
                      <span className="metric-value mono">{prediction.feature_snapshot.cluster_size || 0}</span>
                    </div>
                  </div>
                </>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Orders + redemptions */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 24 }}>
        <div className="card">
          <div className="card-body">
            <h3 className="section-label">ORDERS</h3>
            <table className="data-table" style={{ marginTop: 8 }}>
              <thead>
                <tr>
                  <th>Order</th>
                  <th>Time</th>
                  <th>Amount</th>
                  <th>Status</th>
                </tr>
              </thead>
              <tbody>
                {orders.length === 0 ? (
                  <tr><td colSpan={4}><div className="empty-state" style={{ padding: 24 }}><div className="empty-state-desc">No orders</div></div></td></tr>
                ) : (
                  orders.slice(0, 10).map((o) => (
                    <tr key={o.order_id}>
                      <td className="cell-mono" style={{ fontSize: 11 }}>{o.order_id.slice(0, 8)}…</td>
                      <td className="mono" style={{ fontSize: 11 }}>{new Date(o.timestamp).toLocaleDateString('en-IN', { day: '2-digit', month: 'short' })}</td>
                      <td className="mono" style={{ fontSize: 12, color: 'var(--text-primary)' }}>₹{Number(o.amount).toFixed(2)}</td>
                      <td>
                        <span className={`badge ${o.status === 'completed' ? 'badge-clear' : 'badge-neutral'}`}>
                          {o.status.toUpperCase()}
                        </span>
                      </td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>
        </div>

        <div className="card">
          <div className="card-body">
            <h3 className="section-label">OFFER REDEMPTIONS</h3>
            <table className="data-table" style={{ marginTop: 8 }}>
              <thead>
                <tr>
                  <th>Redemption</th>
                  <th>Time</th>
                  <th>Discount</th>
                </tr>
              </thead>
              <tbody>
                {redemptions.length === 0 ? (
                  <tr><td colSpan={3}><div className="empty-state" style={{ padding: 24 }}><div className="empty-state-desc">No redemptions</div></div></td></tr>
                ) : (
                  redemptions.slice(0, 10).map((r) => (
                    <tr key={r.redemption_id}>
                      <td className="cell-mono" style={{ fontSize: 11 }}>{r.redemption_id.slice(0, 8)}…</td>
                      <td className="mono" style={{ fontSize: 11 }}>{new Date(r.timestamp).toLocaleDateString('en-IN', { day: '2-digit', month: 'short' })}</td>
                      <td className="mono" style={{ fontSize: 12, color: 'var(--text-primary)' }}>₹{Number(r.discount_amount).toFixed(2)}</td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
}

function formatSignalValue(key: string, value: number): { value: string; unit?: string } {
  if (key === 'total_spend' || key === 'average_spend') return { value: `₹${Number(value).toFixed(2)}`, unit: '' };
  if (key === 'order_redemption_rate') return { value: `${(value * 100).toFixed(1)}`, unit: '%' };
  if (key.includes('time_to_first') || key === 'account_age_days') {
    if (key === 'account_age_days') return { value: `${value.toFixed(0)}`, unit: 'd' };
    return { value: `${value.toFixed(1)}`, unit: 'h' };
  }
  if (key === 'avg_entity_degree') return { value: value.toFixed(2), unit: '' };
  return { value: value.toFixed(0), unit: '' };
}

function LocalGraphView({ nodes, links }: { nodes: GraphNode[]; links: GraphLink[] }) {
  const [hovered, setHovered] = useState<string | null>(null);
  const centerNode = nodes[0];
  const otherNodes = nodes.slice(1);
  const radius = 90;

  const nodeColor = (n: GraphNode) => {
    if (n.id === centerNode?.id) return '#FAFAF8';
    if (hovered === n.id) return '#D9391F';
    if (hovered) {
      const isConnected = links.some((l) =>
        (l.source === centerNode?.id && l.target === n.id) ||
        (l.target === centerNode?.id && l.source === n.id)
      );
      return isConnected ? '#D9391F' : '#3E3E3C';
    }
    return '#9BA3AB';
  };

  const nodeOpacity = (n: GraphNode) => {
    if (hovered && n.id !== hovered) {
      const isConnected = links.some((l) =>
        (l.source === n.id) || (l.target === n.id)
      );
      return isConnected ? 1 : 0.25;
    }
    return 1;
  };

  return (
    <div className="graph-area" style={{ background: 'var(--bg-page)', borderRadius: 8, padding: 16, height: 360 }}>
      <svg viewBox="0 0 360 360" width="100%" height="100%">
        {/* Edges */}
        {links.map((l, i) => {
          const sx = 180;
          const sy = 180;
          const tx = 180 + Math.cos((i / links.length) * 2 * Math.PI) * radius;
          const ty = 180 + Math.sin((i / links.length) * 2 * Math.PI) * radius;
          const isHighlighted = hovered && (l.source === hovered || l.target === hovered);
          return (
            <line
              key={i}
              x1={sx}
              y1={sy}
              x2={tx}
              y2={ty}
              stroke={isHighlighted ? '#D9391F' : '#2E2E2C'}
              strokeWidth={isHighlighted ? 1.5 : 1}
              opacity={hovered && !isHighlighted ? 0.25 : 1}
            />
          );
        })}
        {/* Center customer node */}
        {centerNode && (
          <g>
            <circle cx={180} cy={180} r={18} fill="#FAFAF8" />
            <text x={180} y={184} textAnchor="middle" fontSize="10" fontWeight="700" fill="#0A0A0A">
              {centerNode.label.split(' ').map(n => n[0]).join('').slice(0, 2)}
            </text>
            <text x={180} y={210} textAnchor="middle" fontSize="9" fill="#8A8A88" fontFamily="JetBrains Mono">
              {centerNode.label.split(' ')[0]}
            </text>
          </g>
        )}
        {/* Surrounding nodes */}
        {otherNodes.map((n, i) => {
          const angle = (i / Math.max(otherNodes.length, 1)) * 2 * Math.PI;
          const x = 180 + Math.cos(angle) * radius;
          const y = 180 + Math.sin(angle) * radius;
          const color = nodeColor(n);
          const opacity = nodeOpacity(n);
          return (
            <g
              key={n.id}
              opacity={opacity}
              onMouseEnter={() => setHovered(n.id)}
              onMouseLeave={() => setHovered(null)}
              style={{ cursor: 'pointer', transition: 'opacity 200ms ease' }}
            >
              {n.type === 'customer' ? (
                <rect x={x - 12} y={y - 10} width={24} height={20} rx={3} fill={color} />
              ) : (
                <circle cx={x} cy={y} r={9} fill={color} />
              )}
              <text x={x} y={y + (n.type === 'customer' ? 22 : 20)} textAnchor="middle" fontSize="8" fill="#8A8A88">
                {n.type.slice(0, 3).toUpperCase()}
              </text>
            </g>
          );
        })}
        {/* Legend */}
        <g transform="translate(8, 340)">
          <rect x={0} y={-6} width={10} height={10} fill="#FAFAF8" />
          <text x={14} y={2} fontSize="8" fill="#8A8A88">CUSTOMER</text>
          <circle cx={70} cy={-1} r={5} fill="#9BA3AB" />
          <text x={80} y={2} fontSize="8" fill="#8A8A88">ENTITY</text>
        </g>
      </svg>
    </div>
  );
}