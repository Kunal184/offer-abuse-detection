import { useEffect } from 'react';
import { useAppStore } from '../store/appStore';
import { connectEventStream, loadActivityFeed } from '../api/client';
import './Activity.css';

export default function ActivityPage() {
  const events = useAppStore((s) => s.activityEvents);
  const setActivity = useAppStore((s) => s.setActivityEvents);
  const appendActivity = useAppStore((s) => s.appendActivity);
  const loading = useAppStore((s) => s.loading.activity);
  const setLoading = useAppStore((s) => s.setLoading);

  useEffect(() => {
    setLoading('activity', true);

    // Initial fetch of activity log
    loadActivityFeed()
      .then((feed) => {
        setActivity(feed);
      })
      .catch(() => {})
      .finally(() => setLoading('activity', false));

    // Connect to live SSE stream for real-time incoming events
    const disconnectStream = connectEventStream((newEvent) => {
      appendActivity([newEvent]);
    });

    return () => {
      disconnectStream();
    };
  }, [setActivity, appendActivity, setLoading]);

  if (loading && events.length === 0) {
    return (
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', minHeight: 500, padding: 40 }}>
        <div style={{ color: '#E5341C', fontSize: 14, fontFamily: 'JetBrains Mono, monospace' }}>
          LOADING ACTIVITY LOG FEED...
        </div>
      </div>
    );
  }

  return (
    <div className="activity-container">
      <div className="activity-header">
        <div>
          <h1 className="activity-title">LIVE ACTIVITY STREAM</h1>
          <p className="activity-subtitle">REAL-TIME WEBHOOK EVENT MONITORING & INGESTION</p>
        </div>
        <div className="header-meta-pipe">
          {events.length} EVENTS | LIVE SSE STREAM
        </div>
      </div>

      <div className="activity-feed-card">
        {events.length === 0 ? (
          <div style={{ color: 'rgba(250,250,248,0.5)', fontSize: 13, fontFamily: 'JetBrains Mono, monospace', padding: '32px 0', textTransform: 'uppercase' }}>
            LISTENING FOR INCOMING MERCHANT WEBHOOK EVENTS...
          </div>
        ) : (
          events.map((event) => {
            const time = new Date(event.timestamp).toLocaleTimeString('en-US', {
              hour12: false,
              hour: '2-digit',
              minute: '2-digit',
              second: '2-digit',
            });
            const evType = (event.type || event.event_type || 'EVENT').replace(/_/g, ' ').toUpperCase();
            const evDesc = event.description || event.message || 'Activity event logged';
            return (
              <div key={event.id || Math.random()} className="activity-row">
                <div className="activity-row-time">{time}</div>
                <div className={`activity-row-type sev-${event.severity || 'info'}`}>
                  {evType}
                </div>
                <div className="activity-row-desc">{evDesc}</div>
                <div className={`activity-row-dot sev-${event.severity || 'info'}`} />
              </div>
            );
          })
        )}
      </div>
    </div>
  );
}
