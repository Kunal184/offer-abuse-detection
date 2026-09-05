import type {
  CustomerRaw,
  OrderRaw,
  RedemptionRaw,
  DeviceRaw,
  AddressRaw,
  PaymentRaw,
  IpRaw,
  PredictionRequest,
  PredictionResponse,
  OverviewStats,
  ClusterInfo,
  GraphNode,
  GraphLink,
  ModelMetrics,
  FeatureImportance,
} from '../types/index';

const BASE = import.meta.env.VITE_API_BASE || '';

function getAuthApiKey(): string | null {
  try {
    const stored = localStorage.getItem('hex_currentUser');
    if (stored) {
      const parsed = JSON.parse(stored);
      if (parsed.apiKey) return parsed.apiKey;
    }
  } catch {}
  return null;
}

async function apiGet<T>(path: string): Promise<T> {
  const apiKey = getAuthApiKey();
  const headers: Record<string, string> = {};
  if (apiKey) {
    headers['X-API-Key'] = apiKey;
  }
  const res = await fetch(`${BASE}${path}`, { headers });
  if (!res.ok) throw new Error(`API ${path}: ${res.status}`);
  return res.json();
}

async function apiPost<T, B = unknown>(path: string, body: B): Promise<T> {
  const apiKey = getAuthApiKey();
  const headers: Record<string, string> = { 'Content-Type': 'application/json' };
  if (apiKey) {
    headers['X-API-Key'] = apiKey;
  }
  const res = await fetch(`${BASE}${path}`, {
    method: 'POST',
    headers,
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`API ${path}: ${res.status}`);
  return res.json();
}

/* ─── Data loading ─────────────────────────────────────────────── */

export async function loadCustomers(): Promise<CustomerRaw[]> {
  return apiGet<CustomerRaw[]>('/v1/data/customers');
}

/** Returns customers pre-joined with ML abuse scores (bulk vectorised call). */
export async function loadScoredCustomers(): Promise<ScoredCustomer[]> {
  return apiGet<ScoredCustomer[]>('/v1/data/scored-customers');
}

export interface ScoredCustomer {
  customer_id: string;
  name: string;
  email: string;
  phone: number;
  created_at: string;
  abuse_probability: number;
  predicted_label: number;
  cluster_size: number;
  unique_connected_customers: number;
}

export async function loadOrders(): Promise<OrderRaw[]> {
  return apiGet<OrderRaw[]>('/v1/data/orders');
}

export async function loadRedemptions(): Promise<RedemptionRaw[]> {
  return apiGet<RedemptionRaw[]>('/v1/data/redemptions');
}

export async function loadDevices(): Promise<DeviceRaw[]> {
  return apiGet<DeviceRaw[]>('/v1/data/devices');
}

export async function loadAddresses(): Promise<AddressRaw[]> {
  return apiGet<AddressRaw[]>('/v1/data/addresses');
}

export async function loadPayments(): Promise<PaymentRaw[]> {
  return apiGet<PaymentRaw[]>('/v1/data/payments');
}

export async function loadIPs(): Promise<IpRaw[]> {
  return apiGet<IpRaw[]>('/v1/data/ips');
}

export async function loadAllData(): Promise<{
  customers: CustomerRaw[];
  orders: OrderRaw[];
  redemptions: RedemptionRaw[];
  devices: DeviceRaw[];
  addresses: AddressRaw[];
  payments: PaymentRaw[];
  ips: IpRaw[];
  asOf: string;
}> {
  const [customers, orders, redemptions, devices, addresses, payments, ips] =
    await Promise.all([
      loadCustomers(),
      loadOrders(),
      loadRedemptions(),
      loadDevices(),
      loadAddresses(),
      loadPayments(),
      loadIPs(),
    ]);

  // Compute as_of as max timestamp
  const timestamps = [
    ...customers.map((c: CustomerRaw) => c.created_at),
    ...orders.map((o: OrderRaw) => o.timestamp),
    ...redemptions.map((r: RedemptionRaw) => r.timestamp),
  ];
  const asOf = timestamps
    .map((t: string) => new Date(t).getTime())
    .reduce((a: number, b: number) => Math.max(a, b), 0);

  return {
    customers,
    orders,
    redemptions,
    devices,
    addresses,
    payments,
    ips,
    asOf: new Date(asOf).toISOString(),
  };
}

/* ─── ML predictions ──────────────────────────────────────────── */

export async function scoreCustomer(
  customerId: string,
  allData: {
    customers: CustomerRaw[];
    orders: OrderRaw[];
    redemptions: RedemptionRaw[];
    devices: DeviceRaw[];
    addresses: AddressRaw[];
    payments: PaymentRaw[];
    ips: IpRaw[];
    asOf: string;
  }
): Promise<PredictionResponse> {
  return apiPost<PredictionResponse, PredictionRequest>('/v1/predictions', {
    customer_id: customerId,
    customers: allData.customers,
    orders: allData.orders,
    offer_redemptions: allData.redemptions,
    customer_devices: allData.devices,
    customer_addresses: allData.addresses,
    customer_payments: allData.payments,
    customer_ips: allData.ips,
    as_of: allData.asOf,
  });
}

export async function scoreCustomers(customerIds: string[], asOf: string): Promise<PredictionResponse[]> {
  const res = await apiPost<{ predictions: PredictionResponse[]; scored_at: string }, { customer_ids: string[]; as_of: string }>(
    '/v1/predictions/batch',
    { customer_ids: customerIds, as_of: asOf }
  );
  return res.predictions;
}

export async function loadCustomerPrediction(customerId: string, explain = true): Promise<PredictionResponse> {
  return apiGet<PredictionResponse>(`/v1/predictions/${customerId}?explain=${explain}`);
}

/* ─── Overview & Activity ────────────────────────────────────────── */

export async function loadOverview(): Promise<OverviewStats> {
  return apiGet<OverviewStats>('/v1/overview');
}

export async function loadActivityFeed(): Promise<import('../types').ActivityEvent[]> {
  return apiGet<import('../types').ActivityEvent[]>('/v1/activity');
}


/* ─── Graph & Clusters ─────────────────────────────────────────── */

export async function loadGraph(): Promise<{ nodes: GraphNode[]; links: GraphLink[] }> {
  return apiGet<{ nodes: GraphNode[]; links: GraphLink[] }>('/v1/graph');
}

export async function loadClusters(): Promise<{ clusters: ClusterInfo[] }> {
  return apiGet<{ clusters: ClusterInfo[] }>('/v1/clusters');
}

/* ─── Analytics ───────────────────────────────────────────────── */

export async function loadMetrics(): Promise<ModelMetrics> {
  return apiGet<ModelMetrics>('/v1/analytics/metrics');
}

export async function loadFeatureImportance(): Promise<FeatureImportance[]> {
  return apiGet<FeatureImportance[]>('/v1/analytics/feature-importance');
}

/* ─── Health ─────────────────────────────────────────────────── */

export async function healthCheck(): Promise<boolean> {
  try {
    const res = await fetch(`${BASE}/health`);
    return res.ok;
  } catch {
    return false;
  }
}

export async function simulateCustomers(): Promise<{ status: string; simulatedCount: number; overview: OverviewStats }> {
  return apiPost('/v1/simulate', {});
}

export async function resetDatabase(): Promise<{ status: string; message: string }> {
  const res = await fetch(`${BASE}/v1/reset`, { method: 'POST' });
  if (!res.ok) {
    throw new Error('Failed to reset database');
  }
  return res.json();
}

/* ─── Real-time Event Stream ──────────────────────────────────── */

export function connectEventStream(onEvent: (event: import('../types').ActivityEvent) => void): () => void {
  const url = `${BASE}/v1/events/stream`;
  const eventSource = new EventSource(url);

  eventSource.onmessage = (e) => {
    try {
      const data = JSON.parse(e.data);
      if (data && data.id && data.type && data.description) {
        onEvent({
          id: data.id,
          timestamp: data.timestamp || new Date().toISOString(),
          type: data.type,
          description: data.description,
          severity: data.severity || 'neutral',
          entityType: data.entityType,
          entityId: data.entityId,
        });
      }
    } catch {
      // Ignore heartbeat or non-JSON messages
    }
  };

  return () => {
    eventSource.close();
  };
}