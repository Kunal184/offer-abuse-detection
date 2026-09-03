import { useState } from 'react';
import './Settings.css';

export default function SettingsPage() {
  const [copiedKey, setCopiedKey] = useState(false);
  const [copiedUrl, setCopiedUrl] = useState(false);
  const [activeTab, setActiveTab] = useState<'customer_created' | 'order' | 'redemption'>('customer_created');

  const webhookUrl = 'http://localhost:8000/v1/events';
  const apiKey = 'demo_api_key_acme_2026';

  const handleCopyUrl = () => {
    navigator.clipboard.writeText(webhookUrl);
    setCopiedUrl(true);
    setTimeout(() => setCopiedUrl(false), 2000);
  };

  const handleCopyKey = () => {
    navigator.clipboard.writeText(apiKey);
    setCopiedKey(true);
    setTimeout(() => setCopiedKey(false), 2000);
  };

  const payloadExamples = {
    customer_created: {
      event_type: 'customer_created',
      customer_id: 'cust_98a7f1',
      timestamp: '2026-09-04T12:00:00Z',
      payload: {
        name: 'Aarav Sharma',
        email: 'aarav@example.com',
        phone: '+919876543210',
        device_id: 'dev_android_9912',
        address_id: 'addr_mumbai_400001',
        payment_id: 'pay_upi_aarav@upi',
        ip_address: '103.21.244.12',
      },
    },
    order: {
      event_type: 'order',
      customer_id: 'cust_98a7f1',
      timestamp: '2026-09-04T12:05:00Z',
      payload: {
        order_id: 'ord_88192a',
        amount: 2499.0,
        status: 'completed',
        device_id: 'dev_android_9912',
        ip_address: '103.21.244.12',
      },
    },
    redemption: {
      event_type: 'redemption',
      customer_id: 'cust_98a7f1',
      timestamp: '2026-09-04T12:05:01Z',
      payload: {
        redemption_id: 'red_77192a',
        order_id: 'ord_88192a',
        offer_code: 'FESTIVE500',
        discount_amount: 500.0,
      },
    },
  };

  return (
    <div className="settings-container">
      <div className="settings-header">
        <div>
          <h1 className="settings-title">WEBHOOK INTEGRATION</h1>
          <p className="settings-subtitle">API KEYS & REAL-TIME EVENT INGESTION SPECIFICATION</p>
        </div>
        <div className="header-meta-pipe">
          DEMO MERCHANT | SINGLE API KEY | FASTAPI WEBHOOK
        </div>
      </div>

      {/* Framing Banner Callout */}
      <div className="settings-framing-banner">
        <p className="framing-text">
          Connect this once in your order/checkout flow — after that, every order and redemption is scored automatically. No manual data upload.
        </p>
      </div>

      {/* Credentials Card */}
      <div className="settings-card">
        <div className="settings-card-header">
          <h2 className="settings-card-title">MERCHANT CREDENTIALS</h2>
          <span className="settings-tag">DEMO MERCHANT</span>
        </div>

        <div className="credential-row">
          <div className="credential-label">Webhook Endpoint URL</div>
          <div className="credential-value-box">
            <code>{webhookUrl}</code>
            <button className="btn-copy" onClick={handleCopyUrl}>
              {copiedUrl ? 'COPIED ✓' : 'COPY URL'}
            </button>
          </div>
        </div>

        <div className="credential-row">
          <div className="credential-label">Header Authentication (X-API-Key)</div>
          <div className="credential-value-box">
            <code>{apiKey}</code>
            <button className="btn-copy" onClick={handleCopyKey}>
              {copiedKey ? 'COPIED ✓' : 'COPY API KEY'}
            </button>
          </div>
        </div>
      </div>

      {/* Example Payload Schemas */}
      <div className="settings-card">
        <div className="settings-card-header">
          <h2 className="settings-card-title">WEBHOOK PAYLOAD SCHEMAS</h2>
          <div className="payload-tabs">
            {(['customer_created', 'order', 'redemption'] as const).map((tab) => (
              <button
                key={tab}
                className={`tab-btn ${activeTab === tab ? 'active' : ''}`}
                onClick={() => setActiveTab(tab)}
              >
                {tab.replace('_', ' ').toUpperCase()}
              </button>
            ))}
          </div>
        </div>

        <div className="code-block-wrapper">
          <pre className="json-code-block">
            {JSON.stringify(payloadExamples[activeTab], null, 2)}
          </pre>
        </div>
      </div>
    </div>
  );
}
