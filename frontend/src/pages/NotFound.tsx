import { Link } from "react-router-dom";

export default function NotFound() {
  return (
    <main className="placeholder glass">
      <span className="chip">Error 404</span>
      <h2 style={{ marginTop: "0.8rem" }}>This page cites nothing</h2>
      <p>The address does not exist, so there is no source to show for it.</p>
      <p style={{ marginTop: "1.2rem" }}>
        <Link className="btn btn-primary" to="/">
          Back home
        </Link>
      </p>
    </main>
  );
}
