import { Component, type ErrorInfo, type ReactNode } from "react";

interface State {
  error: Error | null;
}

/** Product-grade guard: a crash renders a recovery card, never a white screen. */
export default class ErrorBoundary extends Component<
  { children: ReactNode },
  State
> {
  state: State = { error: null };

  static getDerivedStateFromError(error: Error): State {
    return { error };
  }

  componentDidCatch(error: Error, info: ErrorInfo): void {
    console.error("UI crashed:", error, info.componentStack);
  }

  render() {
    if (!this.state.error) return this.props.children;
    return (
      <main className="placeholder glass">
        <h2>Something broke on our side</h2>
        <p>
          The page hit an unexpected error. Reloading usually fixes it; if it
          keeps happening, open an issue on GitHub.
        </p>
        <p style={{ marginTop: "1.2rem" }}>
          <button className="btn btn-primary" onClick={() => window.location.reload()}>
            Reload
          </button>
        </p>
      </main>
    );
  }
}
