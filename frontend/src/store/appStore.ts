import { create } from 'zustand';
import type {
  CustomerEnriched,
  OverviewStats,
  ActivityEvent,
  ClusterInfo,
  GraphNode,
  GraphLink,
  RiskLevel,
  Severity,
  CustomerRaw,
  OrderRaw,
  RedemptionRaw,
  DeviceRaw,
  AddressRaw,
  PaymentRaw,
  IpRaw,
} from '../types/index';

interface DemoData {
  customers: CustomerRaw[];
  orders: OrderRaw[];
  redemptions: RedemptionRaw[];
  devices: DeviceRaw[];
  addresses: AddressRaw[];
  payments: PaymentRaw[];
  ips: IpRaw[];
  asOf: string;
}

interface AppState {
  // Data
  demoData: DemoData | null;
  customers: CustomerEnriched[];
  overview: OverviewStats | null;
  clusters: ClusterInfo[];
  graphNodes: GraphNode[];
  graphLinks: GraphLink[];
  activityEvents: ActivityEvent[];
  selectedCustomer: CustomerEnriched | null;
  selectedCluster: ClusterInfo | null;
  activeView: string;
  loading: {
    customers: boolean;
    overview: boolean;
    clusters: boolean;
    graph: boolean;
    activity: boolean;
    prediction: boolean;
  };
  error: { [key: string]: string | null };

  // Actions
  setDemoData: (data: DemoData) => void;
  setCustomers: (customers: CustomerEnriched[]) => void;
  setOverview: (overview: OverviewStats) => void;
  setClusters: (clusters: ClusterInfo[]) => void;
  setGraph: (nodes: GraphNode[], links: GraphLink[]) => void;
  appendActivity: (events: ActivityEvent[]) => void;
  setSelectedCustomer: (customer: CustomerEnriched | null) => void;
  setSelectedCluster: (cluster: ClusterInfo | null) => void;
  setActiveView: (view: string) => void;
  setLoading: (key: keyof AppState['loading'], value: boolean) => void;
  setError: (key: string, value: string | null) => void;
  updateCustomerPrediction: (customer: CustomerEnriched) => void;
}

export const useAppStore = create<AppState>((set) => ({
  demoData: null,
  customers: [],
  overview: null,
  clusters: [],
  graphNodes: [],
  graphLinks: [],
  activityEvents: [],
  selectedCustomer: null,
  selectedCluster: null,
  activeView: 'overview',
  loading: {
    customers: false,
    overview: false,
    clusters: false,
    graph: false,
    activity: false,
    prediction: false,
  },
  error: {},

  setDemoData: (data) => set({ demoData: data }),
  setCustomers: (customers) => set({ customers }),
  setOverview: (overview) => set({ overview }),
  setClusters: (clusters) => set({ clusters }),
  setGraph: (nodes, links) => set({ graphNodes: nodes, graphLinks: links }),
  appendActivity: (events) =>
    set((state) => ({ activityEvents: [...state.activityEvents, ...events] })),
  setSelectedCustomer: (customer) => set({ selectedCustomer: customer }),
  setSelectedCluster: (cluster) => set({ selectedCluster: cluster }),
  setActiveView: (view) => set({ activeView: view }),
  setLoading: (key, value) =>
    set((state) => ({ loading: { ...state.loading, [key]: value } })),
  setError: (key, value) =>
    set((state) => ({ error: { ...state.error, [key]: value } })),
  updateCustomerPrediction: (customer) =>
    set((state) => ({
      customers: state.customers.map((c) =>
        c.customer_id === customer.customer_id ? customer : c
      ),
    })),
}));

/* ─── Risk helpers ─────────────────────────────────────────────── */

export function getRiskLevel(probability: number): RiskLevel {
  if (probability >= 0.5) return 'high';
  if (probability >= 0.3) return 'medium';
  return 'clear';
}

export function getRiskColor(level: RiskLevel): string {
  switch (level) {
    case 'high': return '#D9391F';
    case 'medium': return '#EF9F27';
    case 'clear': return '#1D9E75';
  }
}

export function getRiskLabel(level: RiskLevel): string {
  switch (level) {
    case 'high': return 'FLAGGED';
    case 'medium': return 'WATCH';
    case 'clear': return 'CLEAR';
  }
}

export function getSeverityColor(severity: Severity): string {
  switch (severity) {
    case 'high': return '#D9391F';
    case 'medium': return '#EF9F27';
    case 'neutral': return '#8A8A88';
  }
}