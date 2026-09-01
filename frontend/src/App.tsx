import { BrowserRouter, Routes, Route } from 'react-router-dom';
import Shell from './components/layout/Shell';
import OverviewPage from './pages/Overview';
import CustomersPage from './pages/Customers';
import CustomerDetailPage from './pages/CustomerDetail';
import AbuseClustersPage from './pages/AbuseClusters';
import ActivityPage from './pages/Activity';
import AnalyticsPage from './pages/Analytics';

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route element={<Shell />}>
          <Route path="/" element={<OverviewPage />} />
          <Route path="/overview" element={<OverviewPage />} />
          <Route path="/customers" element={<CustomersPage />} />
          <Route path="/customers/:id" element={<CustomerDetailPage />} />
          <Route path="/clusters" element={<AbuseClustersPage />} />
          <Route path="/activity" element={<ActivityPage />} />
          <Route path="/analytics" element={<AnalyticsPage />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}