import { useEffect, useState } from 'react';
import { loadMetrics, loadFeatureImportance } from '../api/client';
import SimpleBarChart from '../components/shared/SimpleBarChart';
import type { ModelMetrics, FeatureImportance } from '../types/index';

const CHART_COLORS = ['#D9391F', '#EF9F27', '#1D9E75', '#9BA3AB', '#8A8A88', '#5E5E5C'];

export default function AnalyticsPage() {
  const [metrics, setMetrics] = useState<ModelMetrics | null>(null);
  const [featureImportance, setFeatureImportance] = useState<FeatureImportance[]>([]);
  const [loadingMetrics, setLoadingMetrics] = useState(true);
  const [loadingFeatures, setLoadingFeatures] = useState(true);
  const [errorMetrics, setErrorMetrics] = useState<string | null>(null);
  const [errorFeatures, setErrorFeatures] = useState<string | null>(null);

  useEffect(() => {
    setLoadingMetrics(true);
    setErrorMetrics(null);
    loadMetrics()
      .then(setMetrics)
      .catch((e: unknown) => {
        setErrorMetrics(e instanceof Error ? e.message : String(e));
        // Fallback to known baseline values so the page remains useful
        setMetrics({
          f1: 0.9351,
          precision: 0.9730,
          recall: 0.9000,
          rocAuc: 0.9989,
          prAuc: 0.9961,
          confusionMatrix: [[132, 1], [4, 36]],
        });
      })
      .finally(() => setLoadingMetrics(false));

    setLoadingFeatures(true);
    setErrorFeatures(null);
    loadFeatureImportance()
      .then(setFeatureImportance)
      .catch((e: unknown) => {
        setErrorFeatures(e instanceof Error ? e.message : String(e));
        setFeatureImportance([
          { feature: 'average_spend', importance: 0.7220 },
          { feature: 'time_to_first_redemption_hours', importance: 0.1178 },
          { feature: 'unique_connected_customers', importance: 0.0426 },
          { feature: 'max_device_user_count', importance: 0.0353 },
          { feature: 'max_ip_user_count', importance: 0.0271 },
          { feature: 'order_count', importance: 0.0220 },
          { feature: 'order_redemption_rate', importance: 0.0091 },
          { feature: 'time_to_first_order_hours', importance: 0.0115 },
          { feature: 'account_age_days', importance: 0.0008 },
          { feature: 'max_address_user_count', importance: 0.0 },
          { feature: 'max_entity_degree', importance: 0.0 },
          { feature: 'cluster_size', importance: 0.0067 },
          { feature: 'redemption_count', importance: 0.0032 },
          { feature: 'total_spend', importance: 0.0017 },
        ]);
      })
      .finally(() => setLoadingFeatures(false));
  }, []);

  const topFeatures = featureImportance.slice(0, 10);

  const perfBarData = metrics
    ? [
        { name: 'Precision', value: metrics.precision, color: '#EF9F27' },
        { name: 'Recall', value: metrics.recall, color: '#1D9E75' },
        { name: 'F1', value: metrics.f1, color: '#D9391F' },
        { name: 'ROC-AUC', value: metrics.rocAuc, color: '#9BA3AB' },
      ]
    : [];

  return (
    <div>
      <div style={{ marginBottom: 28 }}>
        <h1 style={{ fontSize: 24, fontWeight: 700, color: 'var(--text-primary)', marginBottom: 4 }}>Analytics</h1>
        <p style={{ fontSize: 13, color: 'var(--text-secondary)' }}>Model performance and feature analysis</p>
        {(errorMetrics || errorFeatures) && (
          <p style={{ fontSize: 11, color: 'var(--risk-medium)', marginTop: 4, fontFamily: 'var(--font-mono)' }}>
            ⚠ Using baseline values · {errorMetrics || errorFeatures}
          </p>
        )}
      </div>

      {/* Main metrics cards */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: 16, marginBottom: 28 }}>
        <div className="kpi-card">
          <div className="kpi-label">F1 SCORE</div>
          <div className="kpi-value mono" style={{ color: 'var(--risk-high)' }}>{loadingMetrics ? '—' : metrics?.f1.toFixed(4)}</div>
          <div className="kpi-sub">F1 on group-aware test</div>
        </div>
        <div className="kpi-card">
          <div className="kpi-label">PRECISION</div>
          <div className="kpi-value mono" style={{ color: 'var(--risk-medium)' }}>{loadingMetrics ? '—' : metrics?.precision.toFixed(4)}</div>
          <div className="kpi-sub">TP / (TP+FP)</div>
        </div>
        <div className="kpi-card">
          <div className="kpi-label">RECALL</div>
          <div className="kpi-value mono" style={{ color: 'var(--risk-low)' }}>{loadingMetrics ? '—' : metrics?.recall.toFixed(4)}</div>
          <div className="kpi-sub">TP / (TP+FN)</div>
        </div>
        <div className="kpi-card">
          <div className="kpi-label">ROC-AUC</div>
          <div className="kpi-value mono" style={{ color: 'var(--text-primary)' }}>{loadingMetrics ? '—' : metrics?.rocAuc.toFixed(4)}</div>
          <div className="kpi-sub">Area under ROC curve</div>
        </div>
      </div>

      {/* Charts grid */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 24, marginBottom: 24 }}>
        {/* Confusion matrix */}
        <div className="card">
          <div className="card-body">
            <h3 className="section-label">CONFUSION MATRIX (XGBoost)</h3>
            {loadingMetrics || !metrics ? (
              <div style={{ height: 200, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                <span style={{ color: 'var(--text-muted)' }}>Loading...</span>
              </div>
            ) : (
              <div style={{ display: 'grid', gridTemplateColumns: 'auto 1fr 1fr', gap: 8, padding: 8, fontFamily: 'var(--font-mono)' }}>
                <div></div>
                <div style={{ textAlign: 'center', fontSize: 11, color: 'var(--text-muted)', textTransform: 'uppercase' }}>Predicted Clear</div>
                <div style={{ textAlign: 'center', fontSize: 11, color: 'var(--text-muted)', textTransform: 'uppercase' }}>Predicted Flag</div>
                <div style={{ fontSize: 11, color: 'var(--text-muted)', textTransform: 'uppercase', alignSelf: 'center' }}>Actual Clear</div>
                <div style={{ background: 'rgba(29,158,117,0.12)', border: '1px solid var(--risk-low)', borderRadius: 4, padding: '14px 0', textAlign: 'center', color: 'var(--risk-low)', fontSize: 18, fontWeight: 700 }}>{metrics.confusionMatrix[0][0]}</div>
                <div style={{ background: 'rgba(239,159,39,0.12)', border: '1px solid var(--risk-medium)', borderRadius: 4, padding: '14px 0', textAlign: 'center', color: 'var(--risk-medium)', fontSize: 18, fontWeight: 700 }}>{metrics.confusionMatrix[0][1]}</div>
                <div style={{ fontSize: 11, color: 'var(--text-muted)', textTransform: 'uppercase', alignSelf: 'center' }}>Actual Flag</div>
                <div style={{ background: 'rgba(239,159,39,0.12)', border: '1px solid var(--risk-medium)', borderRadius: 4, padding: '14px 0', textAlign: 'center', color: 'var(--risk-medium)', fontSize: 18, fontWeight: 700 }}>{metrics.confusionMatrix[1][0]}</div>
                <div style={{ background: 'rgba(217,57,31,0.15)', border: '1px solid var(--risk-high)', borderRadius: 4, padding: '14px 0', textAlign: 'center', color: 'var(--risk-high)', fontSize: 18, fontWeight: 700 }}>{metrics.confusionMatrix[1][1]}</div>
              </div>
            )}
          </div>
        </div>

        {/* Performance metrics bar chart */}
        <div className="card">
          <div className="card-body">
            <h3 className="section-label">PERFORMANCE METRICS</h3>
            <div style={{ height: 200, display: 'flex', alignItems: 'center' }}>
              {loadingMetrics ? (
                <span style={{ color: 'var(--text-muted)', fontSize: 12 }}>Loading...</span>
              ) : (
                <SimpleBarChart data={perfBarData} height={200} domain={[0, 1]} />
              )}
            </div>
          </div>
        </div>

        {/* Feature importance */}
        <div className="card">
          <div className="card-body">
            <h3 className="section-label">FEATURE IMPORTANCE</h3>
            <p style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 12 }}>XGBoost gain · group-aware split</p>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
              {loadingFeatures ? (
                Array.from({ length: 8 }).map((_, i) => (
                  <div key={i} style={{ display: 'flex', justifyContent: 'space-between', padding: '6px 0' }}>
                    <div className="skeleton" style={{ width: 120, height: 16 }} />
                    <div className="skeleton" style={{ width: 60, height: 16 }} />
                  </div>
                ))
              ) : (
                topFeatures.map((f, i) => {
                  const max = Math.max(...topFeatures.map((x) => x.importance));
                  const width = max ? (f.importance / max) * 100 : 0;
                  return (
                    <div key={f.feature} style={{ display: 'flex', alignItems: 'center', gap: 12, padding: '8px 0' }}>
                      <div style={{ fontSize: 11, color: 'var(--text-secondary)', width: 140, textTransform: 'uppercase', letterSpacing: '0.04em' }}>{f.feature}</div>
                      <div style={{ width: 100, height: 8, background: 'var(--border)', borderRadius: 4, overflow: 'hidden' }}>
                        <div style={{ height: '100%', background: CHART_COLORS[i % CHART_COLORS.length], width: `${width}%`, transition: 'width 800ms ease' }} />
                      </div>
                      <div style={{ fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--text-secondary)', width: 60, textAlign: 'right' }}>{f.importance.toFixed(4)}</div>
                    </div>
                  );
                })
              )}
            </div>
          </div>
        </div>

        {/* Model details */}
        <div className="card">
          <div className="card-body">
            <h3 className="section-label">MODEL DETAILS</h3>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '8px 0', borderBottom: '1px solid var(--border)' }}>
                <span style={{ fontSize: 13, color: 'var(--text-secondary)' }}>Model</span>
                <span style={{ fontFamily: 'var(--font-mono)', fontSize: 12, color: 'var(--text-primary)' }}>xgboost_groupaware</span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '8px 0', borderBottom: '1px solid var(--border)' }}>
                <span style={{ fontSize: 13, color: 'var(--text-secondary)' }}>Status</span>
                <span style={{ fontSize: 12, color: 'var(--risk-low)' }}>FROZEN</span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '8px 0', borderBottom: '1px solid var(--border)' }}>
                <span style={{ fontSize: 13, color: 'var(--text-secondary)' }}>Split</span>
                <span style={{ fontSize: 12, color: 'var(--risk-low)' }}>Group-aware</span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '8px 0', borderBottom: '1px solid var(--border)' }}>
                <span style={{ fontSize: 13, color: 'var(--text-secondary)' }}>Test groups</span>
                <span style={{ fontFamily: 'var(--font-mono)', fontSize: 12, color: 'var(--text-primary)' }}>7</span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '8px 0', borderBottom: '1px solid var(--border)' }}>
                <span style={{ fontSize: 13, color: 'var(--text-secondary)' }}>Threshold</span>
                <span style={{ fontFamily: 'var(--font-mono)', fontSize: 12, color: 'var(--text-primary)' }}>0.5</span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '8px 0' }}>
                <span style={{ fontSize: 13, color: 'var(--text-secondary)' }}>Features</span>
                <span style={{ fontFamily: 'var(--font-mono)', fontSize: 12, color: 'var(--text-primary)' }}>16</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
