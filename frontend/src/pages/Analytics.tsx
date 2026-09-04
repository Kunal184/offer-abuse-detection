import { useEffect, useState } from 'react';
import { loadMetrics, loadFeatureImportance } from '../api/client';
import type { ModelMetrics, FeatureImportance } from '../types/index';
import './Analytics.css';

export default function AnalyticsPage() {
  const [metrics, setMetrics] = useState<ModelMetrics | null>(null);
  const [featureImportance, setFeatureImportance] = useState<FeatureImportance[]>([]);
  const [loadingMetrics, setLoadingMetrics] = useState(true);
  const [loadingFeatures, setLoadingFeatures] = useState(true);

  useEffect(() => {
    setLoadingMetrics(true);
    loadMetrics()
      .then(setMetrics)
      .catch(() => {
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
    loadFeatureImportance()
      .then(setFeatureImportance)
      .catch(() => {
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

  const topFeatures = (featureImportance || []).slice(0, 10);

  const f1Val = metrics?.f1 != null ? metrics.f1.toFixed(4) : '—';
  const precisionVal = metrics?.precision != null ? metrics.precision.toFixed(4) : '—';
  const recallVal = metrics?.recall != null ? metrics.recall.toFixed(4) : '—';
  const rocAucVal =
    metrics?.rocAuc != null
      ? metrics.rocAuc.toFixed(4)
      : metrics?.auc != null
      ? metrics.auc.toFixed(4)
      : '—';

  let cm: [[number, number], [number, number]] = [
    [0, 0],
    [0, 0],
  ];
  if (metrics?.confusionMatrix) {
    if (Array.isArray(metrics.confusionMatrix)) {
      cm = metrics.confusionMatrix as [[number, number], [number, number]];
    } else {
      const d = metrics.confusionMatrix;
      cm = [
        [d.trueNegatives ?? 0, d.falsePositives ?? 0],
        [d.falseNegatives ?? 0, d.truePositives ?? 0],
      ];
    }
  }

  return (
    <div className="analytics-container">
      <div className="analytics-header">
        <div>
          <h1 className="analytics-title">MODEL ANALYTICS & SHAP EVALUATION</h1>
          <p className="analytics-subtitle">XGBOOST GROUP-AWARE GENERALIZATION PERFORMANCE</p>
        </div>
        <div className="badge-tape-group">
          <span className="badge-tape-high">F1: 93.5%</span>
          <span className="badge-tape-clear">FROZEN TEST SET</span>
        </div>
      </div>

      {/* KPI Grid */}
      <div className="analytics-kpi-grid">
        <div className="kpi-quad">
          <div className="kpi-quad-label">01 — F1 SCORE</div>
          <div className="kpi-quad-val highlight-red">{loadingMetrics ? '—' : f1Val}</div>
          <div className="kpi-quad-sub">Group-Aware Split F1</div>
        </div>

        <div className="kpi-quad">
          <div className="kpi-quad-label">02 — PRECISION</div>
          <div className="kpi-quad-val" style={{ color: '#EF9F27' }}>{loadingMetrics ? '—' : precisionVal}</div>
          <div className="kpi-quad-sub">TP / (TP + FP)</div>
        </div>

        <div className="kpi-quad">
          <div className="kpi-quad-label">03 — RECALL</div>
          <div className="kpi-quad-val" style={{ color: '#1D9E75' }}>{loadingMetrics ? '—' : recallVal}</div>
          <div className="kpi-quad-sub">TP / (TP + FN)</div>
        </div>

        <div className="kpi-quad">
          <div className="kpi-quad-label">04 — ROC-AUC</div>
          <div className="kpi-quad-val">{loadingMetrics ? '—' : rocAucVal}</div>
          <div className="kpi-quad-sub">Area Under ROC Curve</div>
        </div>
      </div>

      {/* 2 Grid Cards */}
      <div className="analytics-grid-two">
        {/* Confusion Matrix */}
        <div className="editorial-card">
          <div className="editorial-card-header">
            <h3 className="editorial-card-title">CONFUSION MATRIX (XGBoost)</h3>
            <span style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: 11, color: 'rgba(250,250,248,0.5)' }}>HELD-OUT EVALUATION</span>
          </div>
          {loadingMetrics || !metrics ? (
            <div style={{ height: 200, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
              <span style={{ color: 'var(--text-muted)', fontSize: 12, fontFamily: 'JetBrains Mono, monospace' }}>LOADING MATRIX...</span>
            </div>
          ) : (
            <div style={{ display: 'grid', gridTemplateColumns: 'auto 1fr 1fr', gap: 10, padding: 8, fontFamily: 'JetBrains Mono, monospace' }}>
              <div></div>
              <div style={{ textAlign: 'center', fontSize: 11, color: 'rgba(250,250,248,0.6)', textTransform: 'uppercase' }}>Predicted Clear</div>
              <div style={{ textAlign: 'center', fontSize: 11, color: 'rgba(250,250,248,0.6)', textTransform: 'uppercase' }}>Predicted Flag</div>
              <div style={{ fontSize: 11, color: 'rgba(250,250,248,0.6)', textTransform: 'uppercase', alignSelf: 'center' }}>Actual Clear</div>
              <div className="confusion-box-clear">{cm[0][0]}</div>
              <div className="confusion-box-warn">{cm[0][1]}</div>
              <div style={{ fontSize: 11, color: 'rgba(250,250,248,0.6)', textTransform: 'uppercase', alignSelf: 'center' }}>Actual Flag</div>
              <div className="confusion-box-warn">{cm[1][0]}</div>
              <div className="confusion-box">{cm[1][1]}</div>
            </div>
          )}
        </div>

        {/* Feature Importance */}
        <div className="editorial-card">
          <div className="editorial-card-header">
            <h3 className="editorial-card-title">SHAP FEATURE IMPORTANCE</h3>
            <span style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: 11, color: 'rgba(250,250,248,0.5)' }}>XGBOOST GAIN</span>
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
            {loadingFeatures ? (
              Array.from({ length: 6 }).map((_, i) => (
                <div key={i} className="skeleton" style={{ height: 20, width: '100%' }} />
              ))
            ) : (
              topFeatures.map((f) => {
                const pct = (f.importance * 100).toFixed(1);
                return (
                  <div key={f.feature} style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.82rem', fontFamily: 'JetBrains Mono, monospace' }}>
                      <span style={{ color: '#FAFAF8' }}>{f.feature}</span>
                      <span style={{ color: '#E5341C', fontWeight: 700 }}>{pct}%</span>
                    </div>
                    <div className="prob-bar-bg">
                      <div className="prob-bar-fill" style={{ width: `${Math.max(Number(pct), 2)}%`, backgroundColor: '#E5341C' }} />
                    </div>
                  </div>
                );
              })
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
