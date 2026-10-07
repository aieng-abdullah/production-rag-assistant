export default function Terms() {
  return (
    <main className="shell legal">
      <h1>Terms &amp; Conditions</h1>
      <p className="legal-updated">Last updated: 7 October 2026</p>

      <h2>1. The service</h2>
      <p>
        GroundedAI is a citation-enforced research assistant: it answers
        questions from documents you upload, citing page-level sources. It is a research aid, not legal, financial, or
        medical advice.
      </p>

      <h2>2. Your responsibilities</h2>
      <ul>
        <li>Only upload documents you have the right to process.</li>
        <li>Keep your sign-in credentials to yourself; you are responsible for activity under your account.</li>
        <li>Do not attempt to access other users' tenants or overload the service.</li>
      </ul>

      <h2>3. Accuracy of answers</h2>
      <p>
        Every claim is validated against retrieved sources and answers are
        marked when the corpus does not support them (abstain). Even so,
        outputs can be incomplete or wrong; verify citations against the
        original pages before relying on them.
      </p>

      <h2>4. Free tier and quotas</h2>
      <p>
        Guest access: 3 questions and 1 document. Signed-in free tier: daily
        query and document limits as displayed in the app. Quotas exist to
        keep the service available for everyone and may change with notice.
      </p>

      <h2>5. Availability</h2>
      <p>
        The service is provided "as is", without warranty. We may limit or
        suspend access to protect the service or comply with law.
      </p>

      <h2>6. Termination</h2>
      <p>
        You may delete your documents and stop using the service at any time.
        We may terminate abusive use of the service. Deleting a document
        removes its chunks and embeddings from the index.
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
