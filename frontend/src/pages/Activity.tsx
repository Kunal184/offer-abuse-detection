/**
 * Activity feed page.
 *
 * ─── Event source architecture ───────────────────────────────────────────────
 * Events arrive through an "event source" abstraction:
 *
 *   EVENT SOURCE          NORMALIZATION         PRESENTATION
 *   ──────────────        ──────────────        ──────────────
 *   getDemoEvents()  -->  ActivityEvent[]  -->  ActivityFeed
 *
 * To replace demo data with a real WebSocket stream in the future:
 *   1. Create a new event source (e.g. useWebSocketEvents())
 *   2. Normalise each incoming message into ActivityEvent shape
 *   3. Call appendActivity() with the normalised events
 *   4. Remove the getDemoEvents() call below
 *
 * The presentation layer (ActivityFeed) does not know or care about the source.
 *
 * NOTE: Current events are SIMULATED DEMO DATA — they do NOT reflect real
 * backend activity. They are intentional stand-ins until a WebSocket/event
 * stream is wired in.
 * ─────────────────────────────────────────────────────────────────────────────
 */

import { useEffect, useRef } from 'react';
import { useAppStore } from '../store/appStore';
import { getDemoEvents } from '../utils/demoEvents';

export default function ActivityPage() {
  const events = useAppStore((s) => s.activityEvents);
  const appendActivity = useAppStore((s) => s.appendActivity);
  const loading = useAppStore((s) => s.loading.activity);
  const setLoading = useAppStore((s) => s.setLoading);
  const seededRef = useRef(false);

  useEffect(() => {
    // useRef guard prevents React Strict Mode double-invocation from duplicating events
    if (seededRef.current) return;
    seededRef.current = true;

    setLoading('activity', true);

    // ── Demo event source ──────────────────────────────────────────────────
    // Replace this block with a WebSocket/SSE connection when ready.
    // The appendActivity() call is the only coupling point with the store.
    const demoEvents = getDemoEvents();
    appendActivity(demoEvents);
    // ── End demo event source ──────────────────────────────────────────────

    setLoading('activity', false);
  }, []);

  if (loading && events.length === 0) {
    return (
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', minHeight: 500, padding: 40 }}>
        <div style={{ color: 'var(--text-muted)', fontSize: 14, fontFamily: 'var(--font-mono)' }}>
          LOADING ACTIVITY FEED...
        </div>
      </div>
    );
  }

  return (
    <div className="activity-page">
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 28 }}>
        <div>
          <h1 style={{ fontSize: 24, fontWeight: 700, color: 'var(--text-primary)', marginBottom: 4 }}>Activity</h1>
          <p style={{ fontSize: 13, color: 'var(--text-secondary)' }}>
            Simulated demo events · not live backend activity
          </p>
        </div>
        <div style={{ fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--text-muted)' }}>
          {events.length} events · DEMO MODE
        </div>
      </div>

      <div className="activity-feed">
        {events.map((event) => {
          const sevClass = `sev-${event.severity}`;
          const time = new Date(event.timestamp).toLocaleTimeString('en-US', {
            hour12: false,
            hour: '2-digit',
            minute: '2-digit',
            second: '2-digit',
          });
          return (
            <div key={event.id} className="activity-event">
              <div className="activity-time">{time}</div>
              <div className={`activity-type ${sevClass}`}>
                {event.type.replace(/_/g, ' ').toUpperCase()}
              </div>
              <div className="activity-desc">{event.description}</div>
              <div className={`activity-severity-dot ${sevClass}`} />
            </div>
          );
        })}
      </div>
    </div>
  );
}
