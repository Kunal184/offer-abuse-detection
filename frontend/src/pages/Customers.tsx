import { useEffect, useState, useMemo } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAppStore, getRiskLevel } from '../store/appStore';
import { loadScoredCustomers } from '../api/client';
import RiskBadge from '../components/shared/RiskBadge';
import type { CustomerEnriched } from '../types/index';
import './Customers.css';

export default function CustomersPage() {
  const navigate = useNavigate();
  const customers = useAppStore((s) => s.customers);
  const setCustomers = useAppStore((s) => s.setCustomers);
  const loading = useAppStore((s) => s.loading.customers);
  const setLoading = useAppStore((s) => s.setLoading);
  const error = useAppStore((s) => s.error.customers);
  const setError = useAppStore((s) => s.setError);

  const [search, setSearch] = useState('');
  const [riskFilter, setRiskFilter] = useState('all');
  const [sortBy, setSortBy] = useState<'risk' | 'name' | 'last'>('risk');

  useEffect(() => {
    const store = useAppStore.getState();
    if (store.customers.length > 0 || store.loading.customers) return;

    setLoading('customers', true);
    setError('customers', null);

    loadScoredCustomers()
      .then((scored) => {
        const enriched: CustomerEnriched[] = scored.map((c) => {
          const risk = getRiskLevel(c.abuse_probability);
          return {
            customer_id: c.customer_id,
            name: c.name,
            email: c.email,
            phone: String(c.phone),
            created_at: c.created_at,
            risk,
            abuse_probability: c.abuse_probability,
            cluster_size: c.cluster_size,
            connected_customers: c.unique_connected_customers,
            last_activity: c.created_at,
            status: c.predicted_label === 1 ? 'flagged' : 'active',
            features: {
              customer_id: c.customer_id,
              account_age_days: 0,
              order_count: 0,
              total_spend: 0,
              spend_to_discount_ratio: 0,
              order_amount_std: 0,
              time_to_first_order_hours: 0,
              redemption_count: 0,
              time_to_first_redemption_hours: 0,
              order_redemption_rate: 0,
              max_device_user_count: 0,
              max_address_user_count: 0,
              max_payment_user_count: 0,
              max_ip_user_count: 0,
              unique_connected_customers: c.unique_connected_customers,
              avg_entity_degree: 0,
              max_entity_degree: 0,
              high_value_promo_ratio: 0,
              shared_entity_ratio: 0,
              cluster_creation_span_hours: 0,
              cluster_redemptions_1h: 0,
              min_account_creation_delta_minutes: 0,
              cluster_size: c.cluster_size,
            },
            prediction: null,
          };
        });
        setCustomers(enriched);
      })
      .catch((e: unknown) => {
        setError('customers', e instanceof Error ? e.message : String(e));
      })
      .finally(() => setLoading('customers', false));
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
    <div className="customers-container">
      <div className="customers-header">
        <div>
          <h1 className="customers-title">CUSTOMER INVESTIGATION</h1>
          <p className="customers-subtitle">
            {loading
              ? 'Loading customer scores...'
              : error
              ? `Error: ${error}`
              : `${customers.length} CUSTOMER IDENTITIES ANALYZED`}
          </p>
        </div>
        <div className="header-meta-pipe">
          {flaggedCount} FLAGGED | {watchCount} WATCH | {clearCount} CLEAR
        </div>
      </div>

      <div className="editorial-filter-bar">
        <input
          className="editorial-search-input"
          placeholder="SEARCH BY NAME, EMAIL, OR CUSTOMER ID..."
          value={search}
          onChange={(e) => setSearch(e.target.value)}
        />
        <select
          className="editorial-select"
          value={riskFilter}
          onChange={(e) => setRiskFilter(e.target.value)}
        >
          <option value="all">ALL RISKS</option>
          <option value="high">FLAGGED</option>
          <option value="medium">WATCH</option>
          <option value="clear">CLEAR</option>
        </select>
        <select
          className="editorial-select"
          value={sortBy}
          onChange={(e) => setSortBy(e.target.value as 'risk' | 'name' | 'last')}
        >
          <option value="risk">SORT: RISK SCORE</option>
          <option value="name">SORT: NAME</option>
          <option value="last">SORT: LAST ACTIVITY</option>
        </select>
      </div>

      <div className="editorial-table-card">
        <table className="editorial-data-table">
          <thead>
            <tr>
              <th>Customer Name</th>
              <th>Customer ID</th>
              <th>Abuse Probability</th>
              <th>Risk Assessment</th>
              <th>Connections</th>
              <th>Cluster Size</th>
              <th>Status</th>
            </tr>
          </thead>
          <tbody>
            {loading ? (
              Array.from({ length: 8 }).map((_, i) => (
                <tr key={i}>
                  {Array.from({ length: 7 }).map((_, j) => (
                    <td key={j}>
                      <div className="skeleton" style={{ height: 16, width: j === 1 ? 160 : 80 }} />
                    </td>
                  ))}
                </tr>
              ))
            ) : error ? (
              <tr>
                <td colSpan={7}>
                  <div className="empty-state">
                    <div className="empty-state-title" style={{ color: '#E5341C' }}>Error loading customers</div>
                    <div className="empty-state-desc">{error}</div>
                  </div>
                </td>
              </tr>
            ) : filtered.length === 0 ? (
              <tr>
                <td colSpan={7}>
                  <div className="empty-state">
                    <div className="empty-state-title">No customers found</div>
                    <div className="empty-state-desc">Try adjusting your search or filters</div>
                  </div>
                </td>
              </tr>
            ) : (
              filtered.map((c) => {
                const pct = (c.abuse_probability * 100).toFixed(1);
                const barColor = c.risk === 'high' ? '#E5341C' : c.risk === 'medium' ? '#EF9F27' : '#1D9E75';
                return (
                  <tr key={c.customer_id} onClick={() => navigate(`/customers/${c.customer_id}`)}>
                    <td style={{ fontWeight: 600, color: '#FAFAF8' }}>{c.name}</td>
                    <td style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: '0.82rem', color: 'rgba(250,250,248,0.7)' }}>{c.customer_id}</td>
                    <td>
                      <div className="prob-bar-container">
                        <span style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: '0.88rem', fontWeight: 700, width: '48px', color: barColor }}>
                          {pct}%
                        </span>
                        <div className="prob-bar-bg">
                          <div className="prob-bar-fill" style={{ width: `${pct}%`, backgroundColor: barColor }} />
                        </div>
                      </div>
                    </td>
                    <td>
                      <RiskBadge level={c.risk} />
                    </td>
                    <td style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: '0.88rem' }}>{c.connected_customers}</td>
                    <td style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: '0.88rem' }}>{c.cluster_size}</td>
                    <td>
                      <span style={{
                        fontFamily: 'JetBrains Mono, monospace',
                        fontSize: '0.75rem',
                        fontWeight: 700,
                        textTransform: 'uppercase',
                        color: c.status === 'flagged' ? '#E5341C' : '#1D9E75'
                      }}>
                        {c.status}
                      </span>
                    </td>
                  </tr>
                );
              })
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
