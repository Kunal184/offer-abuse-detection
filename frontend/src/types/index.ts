/* ─── Core domain types ─────────────────────────────────────── */

export type RiskLevel = 'high' | 'medium' | 'clear';
export type Severity = 'high' | 'medium' | 'neutral';
export type EntityType = 'customer' | 'device' | 'address' | 'payment' | 'ip';

/* ─── Raw CSV rows (HistoricalData on backend) ─────────────── */

export interface CustomerRaw {
  customer_id: string;
  name: string;
  email: string;
  phone: string;
  created_at: string;
}

export interface OrderRaw {
  order_id: string;
  customer_id: string;
  amount: number;
  timestamp: string;
  status: string;
}

export interface RedemptionRaw {
  redemption_id: string;
  customer_id: string;
  order_id: string;
  offer_id: string;
  discount_amount: number;
  timestamp: string;
}

export interface DeviceRaw {
  customer_id: string;
  device_id: string;
}

export interface AddressRaw {
  customer_id: string;
  address_id: string;
}

export interface PaymentRaw {
  customer_id: string;
  payment_id: string;
}

export interface IpRaw {
  customer_id: string;
  ip_address: string;
}

export interface OfferRaw {
  offer_id: string;
  code: string;
  type: string;
  is_stackable: boolean;
  max_discount: number;
}

export interface ShapContributor {
  feature_name: string;
  feature_value: number;
  shap_value: number;
  direction: string;
  impact: string;
}

export interface ShapExplanation {
  base_value: number;
  top_positive_contributors: ShapContributor[];
  top_negative_contributors: ShapContributor[];
  all_contributions: ShapContributor[];
}

export interface PredictionResponse {
  customer_id: string;
  abuse_probability: number;
  predicted_label: number;
  decision_threshold: number;
  model_name: string;
  model_version: string;
  feature_snapshot: Record<string, number>;
  graph_signals: Record<string, number>;
  as_of: string;
  scored_at: string;
  explanation?: ShapExplanation | null;
}

/* ─── Feature row (16 columns) ──────────────────────────────── */

export interface FeatureRow {
  customer_id: string;
  account_age_days: number;
  order_count: number;
  total_spend: number;
  average_spend: number;
  time_to_first_order_hours: number;
  redemption_count: number;
  time_to_first_redemption_hours: number;
  order_redemption_rate: number;
  max_device_user_count: number;
  max_address_user_count: number;
  max_payment_user_count: number;
  max_ip_user_count: number;
  unique_connected_customers: number;
  avg_entity_degree: number;
  max_entity_degree: number;
  cluster_size: number;
}

/* ─── Enriched customer (UI view) ───────────────────────────── */

export interface CustomerEnriched {
  customer_id: string;
  name: string;
  email: string;
  phone: string;
  created_at: string;
  risk: RiskLevel;
  abuse_probability: number;
  cluster_size: number;
  connected_customers: number;
  last_activity: string;
  status: string;
  features: FeatureRow;
  prediction: PredictionResponse | null;
}

/* ─── Graph types ───────────────────────────────────────────── */

export interface GraphNode {
  id: string;
  type: EntityType;
  label: string;
  risk?: RiskLevel;
  flagged?: boolean;
  index: number;
}

export interface GraphLink {
  source: string;
  target: string;
  sourceType: EntityType;
  targetType: EntityType;
}

export interface ClusterInfo {
  id: string;
  customerCount: number;
  flaggedCustomerCount: number;
  sharedEntities: { type: string; count: number }[];
  overallRisk: RiskLevel;
  customers: string[];
  entities: string[];
}

/* ─── Activity types ────────────────────────────────────────── */

export interface ActivityEvent {
  id: string;
  timestamp: string;
  type: string;
  description: string;
  severity: Severity;
  entityType?: string;
  entityId?: string;
  event_type?: string;
  message?: string;
}

/* ─── Analytics types ───────────────────────────────────────── */

export interface ConfusionMatrixDict {
  truePositives?: number;
  falsePositives?: number;
  trueNegatives?: number;
  falseNegatives?: number;
}

export interface ModelMetrics {
  f1: number;
  precision: number;
  recall: number;
  rocAuc?: number;
  auc?: number;
  prAuc?: number;
  confusionMatrix: [[number, number], [number, number]] | ConfusionMatrixDict;
}

export interface FeatureImportance {
  feature: string;
  importance: number;
}

/* ─── Overview stats ────────────────────────────────────────── */

export interface OverviewStats {
  customersAnalyzed: number;
  customersFlagged: number;
  abuseClusters: number;
  totalExposure: number;
  flaggedRatio: number;
  riskDistribution?: { high: number; medium: number; clear: number };
  abuseGroupCount?: number;
  asOf?: string;
}

/* ─── Prediction request ────────────────────────────────────── */

export interface PredictionRequest {
  customer_id: string;
  customers: CustomerRaw[];
  orders: OrderRaw[];
  offer_redemptions: RedemptionRaw[];
  customer_devices: DeviceRaw[];
  customer_addresses: AddressRaw[];
  customer_payments: PaymentRaw[];
  customer_ips: IpRaw[];
  as_of: string;
}