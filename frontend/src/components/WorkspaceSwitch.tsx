export type Workspace = "legal" | "academic";

export const WS_ACCENT: Record<Workspace, string> = {
  legal: "#1e3a5f",
  academic: "#0d9488",
};

export default function WorkspaceSwitch({
  value,
  onChange,
}: {
  value: Workspace;
  onChange: (workspace: Workspace) => void;
}) {
  return (
    <div className="chat-ws" role="tablist" aria-label="Workspace">
      {(["legal", "academic"] as Workspace[]).map((ws) => (
        <button
          key={ws}
          role="tab"
          aria-selected={value === ws}
          className={`chat-ws-btn${value === ws ? " is-active" : ""}`}
          style={value === ws ? { background: WS_ACCENT[ws] } : undefined}
          onClick={() => onChange(ws)}
        >
          {ws}
        </button>
      ))}
    </div>
  );
}
