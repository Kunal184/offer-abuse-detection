import type { ActivityEvent } from '../types/index';

export function getDemoEvents(): ActivityEvent[] {
  return [
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
}
