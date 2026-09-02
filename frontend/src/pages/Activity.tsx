import { useEffect } from 'react';
import { useAppStore } from '../store/appStore';
import { connectEventStream } from '../api/client';
import './Activity.css';

export default function ActivityPage() {
  const events = useAppStore((s) => s.activityEvents);
  const appendActivity = useAppStore((s) => s.appendActivity);
  const loading = useAppStore((s) => s.loading.activity);
  const setLoading = useAppStore((s) => s.setLoading);

  useEffect(() => {
    setLoading('activity', false);

    const disconnectStream = connectEventStream((newEvent) => {
      appendActivity([newEvent]);
    });

    return () => {
      disconnectStream();
    };
  }, [appendActivity, setLoading]);

  if (loading && events.length === 0) {
    return (
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', minHeight: 500, padding: 40 }}>
        <div style={{ color: '#E5341C', fontSize: 14, fontFamily: 'JetBrains Mono, monospace' }}>
          CONNECTING TO LIVE SSE EVENT STREAM...
        </div>
      </div>
    );
  }

  return (
    <div className="activity-container">
      <div className="activity-header">
        <div>
          <h1 className="activity-title">LIVE ACTIVITY STREAM</h1>
          <p className="activity-subtitle">REAL-TIME SSE EVENT MONITORING & INGESTION</p>
        </div>
        <div className="badge-tape-group">
          <span className="badge-tape-high">{events.length} EVENTS</span>
          <span className="badge-tape-clear">LIVE SSE STREAM</span>
        </div>
      </div>

      <div className="activity-feed-card">
        {events.length === 0 ? (
          <div style={{ color: 'rgba(250,250,248,0.5)', fontSize: 13, fontFamily: 'JetBrains Mono, monospace', padding: '32px 0', textTransform: 'uppercase' }}>
            LISTENING FOR INCOMING MERCHANT EVENTS...
          </div>
        ) : (
          events.map((event) => {
            const time = new Date(event.timestamp).toLocaleTimeString('en-US', {
              hour12: false,
              hour: '2-digit',
              minute: '2-digit',
              second: '2-digit',
            });
            return (
              <div key={event.id} className="activity-row">
                <div className="activity-row-time">{time}</div>
                <div className={`activity-row-type sev-${event.severity}`}>
                  {event.type.replace(/_/g, ' ').toUpperCase()}
                </div>
                <div className="activity-row-desc">{event.description}</div>
                <div className={`activity-row-dot sev-${event.severity}`} />
              </div>
            );
          })
        )}
      </div>
    </div>
  );
}
