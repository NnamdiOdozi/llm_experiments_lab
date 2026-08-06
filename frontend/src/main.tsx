import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";
import ErrorBoundary from "./components/ErrorBoundary";
import "./index.css";

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <ErrorBoundary>
      {/* A fluid wrapper keeps the dashboard responsive while leaving browser
          zoom entirely under the user's control. */}
      <div id="app-scale">
        <App />
      </div>
    </ErrorBoundary>
  </React.StrictMode>
);
