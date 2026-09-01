import { useEffect } from 'react';
import { useAppStore } from '../store/appStore';

export default function ActivityPage() {
  const events = useAppStore((s) => s.activityEvents);
  const setEvents = useAppStore((s) => s.appendActivity);
  const loading = useAppStore((s) => s.loading.activity);
  const setLoading = useAppStore((s) => s.setLoading);

  useEffect(() => {
    if (events.length === 0) {
      setLoading('activity', true);
      // Generate demo events
      const demoEvents = [
        {
          id: '1', timestamp: new Date().toISOString(),
          type: 'customer_created',
          description: 'New customer registered in system',
          severity: 'neutral' as const,
        },
        {
          id: '2', timestamp: new Date(Date.now() - 3600000).toISOString(),
          type: 'order_placed',
          description: 'Customer placed order #ORD-2341, amount ₹1,247.89',
          severity: 'neutral' as const,
        },
        {
          id: '3', timestamp: new Date(Date.now() - 7200000).toISOString(),
          type: 'offer_redeemed',
          description: 'Customer redeemed WELCOME50 discount ₹500',
          severity: 'medium' as const,
        },
        {
          id: '4', timestamp: new Date(Date.now() - 900000).toISOString(),
          type: 'suspicious_connection',
          description: 'Customer shares device with flagged account',
          severity: 'high' as const,
        },
        {
          id: '5', timestamp: new Date(Date.now() - 540000).toISOString(),
          type: 'risk_score',
          description: 'Abuse probability updated from 34% to 52%',
          severity: 'medium' as const,
        },
        {
          id: '6', timestamp: new Date(Date.now() - 270000).toISOString(),
          type: 'customer_flagged',
          description: 'Customer flagged as coordinated abuse participant',
          severity: 'high' as const,
        },
        {
          id: '7', timestamp: new Date(Date.now() - 180000).toISOString(),
          type: 'cluster_updated',
          description: '3 new accounts added to existing abuse cluster',
          severity: 'medium' as const,
        },
        {
          id: '8', timestamp: new Date(Date.now() - 120000).toISOString(),
          type: 'order_placed',
          description: 'Customer placed order #ORD-2342, amount ₹892.45',
          severity: 'neutral' as const,
        },
        {
          id: '9', timestamp: new Date(Date.now() - 60000).toISOString(),
          type: 'risk_score',
          description: 'Abuse probability increased to 87.3%',
          severity: 'high' as const,
        },
      ];
      setEvents(demoEvents);
      setLoading('activity', false);
    }
  }, []);

  if (loading && events.length === 0) {
    return (
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', minHeight: 500, padding: 40 }}>
        <div style={{ color: 'var(--text-muted)', fontSize: 14, fontFamily: 'var(--font-mono)' }}>LOADING ACTIVITY FEED...</div>
      </div>
    );
  }

  return (
    <div className="activity-page">
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 28 }}>
        <h1 style={{ fontSize: 24, fontWeight: 700, color: 'var(--text-primary)', marginBottom: 4 }}>Activity</h1>
        <div style={{ fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--text-muted)' }}>
          {events.length} events
        </div>
      </div>

      <div className="activity-feed">
        {events.map((event) => {
          const sevClass = `sev-${event.severity}`;
          const time = new Date(event.timestamp).toLocaleTimeString('en-US', {
            hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit',
          });
          return (
            <div key={event.id} className="activity-event">
              <div className={`activity-time`}>{time}</div>
              <div className={`activity-type ${sevClass}`}>{event.type.replace('_', ' ').toUpperCase()}</div>
              <div className="activity-desc">{event.description}</div>
              <div className={`activity-severity-dot ${sevClass}`} />
            </div>
          );
        })}
      </div>
    </div>
  );
}
