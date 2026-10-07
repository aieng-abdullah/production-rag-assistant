export default function Privacy() {
  return (
    <main className="shell legal">
      <h1>Privacy Policy</h1>
      <p className="legal-updated">Last updated: 7 October 2026</p>

      <h2>1. What we process</h2>
      <p>
        GroundedAI is a citation-verified research assistant. To answer your
        questions we process: the documents you upload (PDF files), your
        queries, and basic account data (email address, sign-in method,
        usage counters).
      </p>

      <h2>2. How your documents are used</h2>
      <p>
        Uploaded PDFs are parsed into page-aware chunks and embedded into
        vectors. Embedding requests are sent to <strong>Voyage AI</strong>;
        your question and document excerpts are sent to <strong>Groq</strong>{" "}
        to generate answers. Retrieval runs against a tenant-isolated vector
        store; your documents are never visible to other users.
      </p>

      <h2>3. What we do not do</h2>
      <ul>
        <li>We do not sell or share your documents with third parties beyond the processing above.</li>
        <li>We do not use your documents to train models.</li>
        <li>We do not put your content in logs, URLs, or error messages.</li>
      </ul>

      <h2>4. Cookies and local storage</h2>
      <p>
        We store your sign-in token and guest device id in your
        browser's local storage so your session survives refreshes. We do not
        use advertising or cross-site tracking cookies.
      </p>

      <h2>5. Retention</h2>
      <p>
        Documents and their embeddings are kept until you delete them from the
        Documents page. Usage counters reset daily (00:00 UTC). Guest sessions
        are tied to an anonymous device id and hold no personal data.
      </p>

      <h2>6. Optional tracing</h2>
      <p>
        When enabled by the operator, request traces are recorded in Langfuse
        for debugging and evaluation. Traces contain retrieval context and
        model outputs of your own requests, never other users' data.
      </p>

      <h2>7. Contact</h2>
      <p>
        Questions? Open an issue on{" "}
        <a href="https://github.com/aieng-abdullah/production-rag-assistant/issues" target="_blank" rel="noopener noreferrer">
          GitHub
        </a>.
      </p>
    </main>
  );
}
