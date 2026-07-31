import { Component, type ErrorInfo, type ReactNode } from "react";

interface Props {
  children: ReactNode;
}

interface State {
  error: Error | null;
}

export class AppErrorBoundary extends Component<Props, State> {
  state: State = { error: null };

  static getDerivedStateFromError(error: Error): State {
    return { error };
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error("K2 Region Lab render failure", error, info.componentStack);
  }

  render() {
    if (!this.state.error) return this.props.children;
    return <main className="fatal-shell">
      <p className="kicker">Interface error</p>
      <h1>K2 Region Lab could not render this view</h1>
      <p>The current project state triggered an interface error. Reload to recover; project and cloud files are not deleted.</p>
      <pre className="fatal-error-detail">{this.state.error.stack || this.state.error.message}</pre>
      <button className="primary-button" onClick={() => window.location.reload()}>Reload interface</button>
    </main>;
  }
}
