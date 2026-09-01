import { useEffect, useState, useMemo } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAppStore, getRiskLevel } from '../store/appStore';
import { loadAllData, scoreCustomers } from '../api/client';
import RiskBadge from '../components/shared/RiskBadge';
import type { CustomerEnriched, CustomerRaw } from '../types/index';

export default function CustomersPage() {
  const navigate = useNavigate();
  const demoData = useAppStore((s) => s.demoData);
  const customers = useAppStore((s) => s.customers);
  const setCustomers = useAppStore((s) => s.setCustomers);
  const setDemoData = useAppStore((s) => s.setDemoData);
  const loading = useAppStore((s) => s.loading.customers);
  const setLoading = useAppStore((s) => s.setLoading);

  const [search, setSearch] = useState('');
  const [riskFilter, setRiskFilter] = useState('all');
  const [sortBy, setSortBy] = useState<'risk' | 'name' | 'last'>('risk');

  useEffect(() => {
    if (!demoData) {
      setLoading('customers', true);
      loadAllData()
        .then(async (data) => {
          setDemoData(data);
          try {
            const preds = await scoreCustomers(
              data.customers.map((c: CustomerRaw) => c.customer_id),
              data.asOf
            );
            const predMap = new Map(preds.map((p) => [p.customer_id, p]));
            const enriched: CustomerEnriched[] = data.customers.map((c: CustomerRaw) => {
              const pred = predMap.get(c.customer_id);
              const risk = pred ? getRiskLevel(pred.abuse_probability) : 'clear';
              const features = pred?.feature_snapshot;
              return {
                customer_id: c.customer_id,
                name: c.name,
                email: c.email,
                phone: c.phone,
                created_at: c.created_at,
                risk,
                abuse_probability: pred?.abuse_probability ?? 0,
                cluster_size: (features as any)?.cluster_size ?? 1,
                connected_customers: (features as any)?.unique_connected_customers ?? 0,
                last_activity: c.created_at,
                status: risk === 'high' ? 'flagged' : 'active',
                features: {
                  customer_id: c.customer_id,
                  account_age_days: 0, order_count: 0, total_spend: 0,
                  average_spend: 0, time_to_first_order_hours: 0,
                  redemption_count: 0, time_to_first_redemption_hours: 0,
                  order_redemption_rate: 0, max_device_user_count: 0,
                  max_address_user_count: 0, max_payment_user_count: 0,
                  max_ip_user_count: 0, unique_connected_customers: 0,
                  avg_entity_degree: 0, max_entity_degree: 0, cluster_size: 1,
                },
                prediction: pred ?? null,
              };
            });
            setCustomers(enriched);
          } catch {
            // Fallback: just show customers without scores
            const enriched: CustomerEnriched[] = data.customers.map((c: CustomerRaw) => ({
              customer_id: c.customer_id,
              name: c.name,
              email: c.email,
              phone: c.phone,
              created_at: c.created_at,
              risk: 'clear' as const,
              abuse_probability: 0,
              cluster_size: 1,
              connected_customers: 0,
              last_activity: c.created_at,
              status: 'active',
              features: {
                customer_id: c.customer_id,
                account_age_days: 0, order_count: 0, total_spend: 0,
                average_spend: 0, time_to_first_order_hours: 0,
                redemption_count: 0, time_to_first_redemption_hours: 0,
                order_redemption_rate: 0, max_device_user_count: 0,
                max_address_user_count: 0, max_payment_user_count: 0,
                max_ip_user_count: 0, unique_connected_customers: 0,
                avg_entity_degree: 0, max_entity_degree: 0, cluster_size: 1,
              },
              prediction: null,
            }));
            setCustomers(enriched);
          }
        })
        .finally(() => setLoading('customers', false));
    }
  }, []);

  const filtered = useMemo(() => {
    let list = [...customers];
    if (search) {
      const q = search.toLowerCase();
      list = list.filter(
        (c) =>
          c.name.toLowerCase().includes(q) ||
          c.customer_id.toLowerCase().includes(q) ||
          c.email.toLowerCase().includes(q)
      );
    }
    if (riskFilter !== 'all') {
      list = list.filter((c) => c.risk === riskFilter);
    }
    list.sort((a, b) => {
      if (sortBy === 'risk') return b.abuse_probability - a.abuse_probability;
      if (sortBy === 'name') return a.name.localeCompare(b.name);
      return new Date(b.last_activity).getTime() - new Date(a.last_activity).getTime();
    });
    return list;
  }, [customers, search, riskFilter, sortBy]);

  const flaggedCount = customers.filter((c) => c.risk === 'high').length;
  const watchCount = customers.filter((c) => c.risk === 'medium').length;
  const clearCount = customers.filter((c) => c.risk === 'clear').length;

  return (
    <div>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 24 }}>
        <div>
          <h1 style={{ fontSize: 24, fontWeight: 700, color: 'var(--text-primary)', marginBottom: 4 }}>Customers</h1>
          <p style={{ fontSize: 13, color: 'var(--text-secondary)' }}>
            {loading ? 'Loading...' : `${customers.length} customers analyzed`}
          </p>
        </div>
        <div style={{ display: 'flex', gap: 8 }}>
          <span className="badge badge-high">{flaggedCount} flagged</span>
          <span className="badge badge-medium">{watchCount} watch</span>
          <span className="badge badge-clear">{clearCount} clear</span>
        </div>
      </div>

      <div className="filter-bar" style={{ marginBottom: 20 }}>
        <input
          className="search-input"
          placeholder="Search by name, email, or ID..."
          value={search}
          onChange={(e) => setSearch(e.target.value)}
        />
        <select
          className="filter-select"
          value={riskFilter}
          onChange={(e) => setRiskFilter(e.target.value)}
        >
          <option value="all">All risks</option>
          <option value="high">Flagged</option>
          <option value="medium">Watch</option>
          <option value="clear">Clear</option>
        </select>
        <select
          className="filter-select"
          value={sortBy}
          onChange={(e) => setSortBy(e.target.value as 'risk' | 'name' | 'last')}
        >
          <option value="risk">Sort: Risk</option>
          <option value="name">Sort: Name</option>
          <option value="last">Sort: Last Activity</option>
        </select>
      </div>

      <div className="card">
        <table className="data-table">
          <thead>
            <tr>
              <th>Customer</th>
              <th>ID</th>
              <th>Abuse Probability</th>
              <th>Risk Level</th>
              <th>Connections</th>
              <th>Cluster Size</th>
              <th>Last Activity</th>
              <th>Status</th>
            </tr>
          </thead>
          <tbody>
            {loading ? (
              Array.from({ length: 5 }).map((_, i) => (
                <tr key={i}>
                  {Array.from({ length: 8 }).map((_, j) => (
                    <td key={j}>
                      <div className="skeleton" style={{ height: 16, width: j === 1 ? 160 : 80 }} />
                    </td>
                  ))}
                </tr>
              ))
            ) : filtered.length === 0 ? (
              <tr>
                <td colSpan={8}>
                  <div className="empty-state">
                    <div className="empty-state-title">No customers found</div>
                    <div className="empty-state-desc">Try adjusting your search or filters</div>
                  </div>
                </td>
              </tr>
            ) : (
              filtered.map((customer) => (
                <tr key={customer.customer_id} onClick={() => navigate(`/customers/${customer.customer_id}`)}>
                  <td>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                      <div className={`customer-avatar ${customer.risk === 'high' ? 'flagged' : ''}`}>
                        {customer.name.split(' ').map((n) => n[0]).join('').slice(0, 2).toUpperCase()}
                      </div>
                      <div>
                        <div style={{ fontSize: 13, fontWeight: 500, color: 'var(--text-primary)' }}>{customer.name}</div>
                        <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>{customer.email}</div>
                      </div>
                    </div>
                  </td>
                  <td>
                    <span className="mono" style={{ fontSize: 11 }}>
                      {customer.customer_id.slice(0, 8)}…
                    </span>
                  </td>
                  <td>
                    <span
                      style={{
                        fontFamily: 'var(--font-mono)',
                        fontSize: 13,
                        fontWeight: 600,
                        color: customer.risk === 'high' ? 'var(--risk-high)'
                          : customer.risk === 'medium' ? 'var(--risk-medium)'
                          : 'var(--risk-low)',
                      }}
                    >
                      {customer.abuse_probability > 0 ? `${(customer.abuse_probability * 100).toFixed(1)}%` : '—'}
                    </span>
                  </td>
                  <td><RiskBadge level={customer.risk} /></td>
                  <td><span className="mono">{customer.connected_customers}</span></td>
                  <td><span className="mono">{customer.cluster_size}</span></td>
                  <td>
                    <span className="mono">
                      {new Date(customer.last_activity).toLocaleDateString('en-IN', {
                        day: '2-digit', month: 'short', year: '2-digit'
                      })}
                    </span>
                  </td>
                  <td>
                    <span
                      className="badge"
                      style={{
                        background: customer.status === 'flagged' ? 'rgba(217,57,31,0.15)' : 'rgba(29,158,117,0.12)',
                        color: customer.status === 'flagged' ? 'var(--risk-high)' : 'var(--risk-low)',
                        border: `1px solid ${customer.status === 'flagged' ? 'rgba(217,57,31,0.3)' : 'rgba(29,158,117,0.25)'}`,
                      }}
                    >
                      {customer.status === 'flagged' ? 'FLAGGED' : 'ACTIVE'}
                    </span>
                  </td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
