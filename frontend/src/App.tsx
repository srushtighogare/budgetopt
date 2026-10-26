import { lazy, Suspense } from "react";
import { Navigate, Route, Routes } from "react-router-dom";

import { useAuth } from "./auth/AuthContext";
import { Layout } from "./components/Layout";
import { TeamProvider } from "./components/TeamContext";
import { Spinner } from "./components/ui";
import LoginPage from "./pages/Login";

const AlertsPage = lazy(() => import("./pages/Alerts"));
const ForecastPage = lazy(() => import("./pages/Forecast"));
const ModelsPage = lazy(() => import("./pages/Models"));
const OptimizationPage = lazy(() => import("./pages/Optimization"));
const OverviewPage = lazy(() => import("./pages/Overview"));
const PlaygroundPage = lazy(() => import("./pages/Playground"));
const RequestsPage = lazy(() => import("./pages/Requests"));
const TeamsPage = lazy(() => import("./pages/Teams"));
const WhatIfPage = lazy(() => import("./pages/WhatIf"));

export default function App() {
  const { user, loading, isManager } = useAuth();

  if (loading) return <div className="grid h-full place-items-center"><Spinner /></div>;
  if (!user) {
    return (
      <Routes>
        <Route path="*" element={<LoginPage />} />
      </Routes>
    );
  }

  return (
    <TeamProvider>
      <Suspense fallback={<Spinner />}>
        <Routes>
          <Route element={<Layout />}>
            <Route index element={<OverviewPage />} />
            <Route path="playground" element={<PlaygroundPage />} />
            <Route path="requests" element={<RequestsPage />} />
            <Route path="forecast" element={<ForecastPage />} />
            <Route path="alerts" element={<AlertsPage />} />
            <Route path="models" element={<ModelsPage />} />
            <Route path="teams" element={<TeamsPage />} />
            {isManager && <Route path="optimization" element={<OptimizationPage />} />}
            {isManager && <Route path="whatif" element={<WhatIfPage />} />}
            <Route path="*" element={<Navigate to="/" replace />} />
          </Route>
        </Routes>
      </Suspense>
    </TeamProvider>
  );
}
