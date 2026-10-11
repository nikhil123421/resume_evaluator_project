import { useEffect, useRef, useState } from "react";

type Evaluation = {
  requirement: string;
  importance: "Required" | "Preferred";
  status: "Supported" | "Unverified" | "Missing";
  evidence_quote: string | null;
  reasoning: string;
};

type MatchResult = {
  candidate_name: string;
  overall_score: number;
  evaluations: Evaluation[];
  evaluation_complete: boolean;
  warnings: string[];
};

const cn = (...classes: (string | false | undefined)[]) =>
  classes.filter(Boolean).join(" ");

/* ------------------------------------------------------------------ */
/*  UI-only helpers (no business logic)                                */
/* ------------------------------------------------------------------ */

function useCountUp(target: number, duration = 1400) {
  const [value, setValue] = useState(0);
  useEffect(() => {
    let raf = 0;
    const start = performance.now();
    const tick = (now: number) => {
      const t = Math.min(1, (now - start) / duration);
      const eased = 1 - Math.pow(1 - t, 3);
      setValue(target * eased);
      if (t < 1) raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [target, duration]);
  return value;
}

function ScoreNumber({ value }: { value: number }) {
  const shown = useCountUp(value);
  return <>{Math.round(shown)}</>;
}

function Eyebrow({ children }: { children: React.ReactNode }) {
  return (
    <p className="text-[10px] font-medium uppercase tracking-[0.28em] text-white/45">
      {children}
    </p>
  );
}

function ArrowIcon({ diagonal = false }: { diagonal?: boolean }) {
  return (
    <svg
      width="16"
      height="16"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.5"
      aria-hidden="true"
    >
      {diagonal ? (
        <path d="M7 17 17 7M7 7h10v10" />
      ) : (
        <path d="M5 12h14m-6-6 6 6-6 6" />
      )}
    </svg>
  );
}

function UploadIcon() {
  return (
    <svg
      width="24"
      height="24"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.4"
      aria-hidden="true"
    >
      <path d="M12 15V3m0 0L7 8m5-5 5 5" />
      <path d="M4 15v4a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-4" />
    </svg>
  );
}

function StatusBadge({ status }: { status: Evaluation["status"] }) {
  const styles = {
    Supported: "border-white bg-white text-black",
    Unverified: "border-white/35 bg-transparent text-white/85",
    Missing: "border-white/12 bg-transparent text-white/40",
  };

  const dot = {
    Supported: "bg-black",
    Unverified: "bg-white",
    Missing: "bg-white/30",
  };

  return (
    <span
      className={cn(
        "inline-flex shrink-0 items-center gap-2 border px-3 py-1.5 text-[9px] font-semibold uppercase tracking-[0.18em]",
        styles[status]
      )}
    >
      <span className={cn("h-1.5 w-1.5 rounded-full", dot[status])} />
      {status}
    </span>
  );
}

/* ------------------------------------------------------------------ */
/*  App                                                                */
/* ------------------------------------------------------------------ */

export default function App() {
  const [jobDescription, setJobDescription] = useState("");
  const [resume, setResume] = useState<File | null>(null);
  const [result, setResult] = useState<MatchResult | null>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [dragging, setDragging] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const resultsRef = useRef<HTMLElement>(null);

  function selectResume(file?: File) {
    if (!file) return;

    if (
      file.type !== "application/pdf" &&
      !file.name.toLowerCase().endsWith(".pdf")
    ) {
      setError("This file isn't a PDF. Please select a PDF resume.");
      return;
    }

    if (file.size === 0) {
      setError("This file is empty. Please select another PDF.");
      return;
    }

    setResume(file);
    setError("");
  }

  async function handleSubmit(e: React.FormEvent<HTMLFormElement>) {
    e.preventDefault();
    setError("");
    setResult(null);

    if (!resume) {
      setError("Upload your resume PDF to continue.");
      return;
    }

    if (
      resume.type !== "application/pdf" &&
      !resume.name.toLowerCase().endsWith(".pdf")
    ) {
      setError("Please upload a PDF file.");
      return;
    }

    if (jobDescription.trim().length < 30) {
      setError("The job description must contain at least 30 characters.");
      return;
    }

    const formData = new FormData();
    formData.append("job_description", jobDescription);
    formData.append("resume_pdf", resume);

    setLoading(true);

    try {
      const API_BASE_URL = import.meta.env.VITE_API_URL;

      if (!API_BASE_URL) {
        throw new Error("Backend URL is not configured.");
      }

      const response = await fetch(
        `${API_BASE_URL.replace(/\/+$/, "")}/api/v1/evaluate`,
        {
          method: "POST",
          body: formData,
        }
      );

      if (!response.ok) {
        const data = await response.json().catch(() => null);
        throw new Error(
          typeof data?.detail === "string"
            ? data.detail
            : "Evaluation failed. Please try again."
        );
      }

      const data: MatchResult = await response.json();
      setResult(data);

      window.setTimeout(() => {
        resultsRef.current?.scrollIntoView({
          behavior: "smooth",
          block: "start",
        });
      }, 100);
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Could not connect to the backend."
      );
    } finally {
      setLoading(false);
    }
  }

  const supported =
    result?.evaluations.filter((item) => item.status === "Supported").length ??
    0;
  const unverified =
    result?.evaluations.filter((item) => item.status === "Unverified")
      .length ?? 0;
  const missing =
    result?.evaluations.filter((item) => item.status === "Missing").length ??
    0;
  const total = result?.evaluations.length ?? 0;

  const score = Math.max(
    0,
    Math.min(100, Number(result?.overall_score ?? 0))
  );

  return (
    <main className="relative min-h-screen overflow-x-hidden bg-black text-white antialiased selection:bg-white selection:text-black">
      {/* ---------- global styles: animations only ---------- */}
      <style>{`
        @keyframes fadeUp {
          from { opacity: 0; transform: translateY(22px); }
          to   { opacity: 1; transform: translateY(0); }
        }
        @keyframes fadeIn {
          from { opacity: 0; }
          to   { opacity: 1; }
        }
        @keyframes barGrow {
          from { width: 0%; }
        }
        @keyframes floatSlow {
          0%, 100% { transform: translate(0, 0); }
          50%      { transform: translate(30px, -24px); }
        }
        @keyframes scanSweep {
          0%   { transform: translateX(-100%); opacity: 0; }
          20%  { opacity: 1; }
          80%  { opacity: 1; }
          100% { transform: translateX(100%); opacity: 0; }
        }
        .a-fade-up { animation: fadeUp 0.9s cubic-bezier(0.22, 1, 0.36, 1) both; }
        .a-fade-in { animation: fadeIn 1.1s ease-out both; }
        .a-bar     { animation: barGrow 1.4s cubic-bezier(0.22, 1, 0.36, 1) both; }
        .a-float   { animation: floatSlow 22s ease-in-out infinite; }
        .a-scan    { animation: scanSweep 1.6s cubic-bezier(0.4, 0, 0.4, 1) infinite; }
        @media (prefers-reduced-motion: reduce) {
          .a-fade-up, .a-fade-in, .a-bar, .a-float, .a-scan { animation: none !important; }
        }
      `}</style>

      {/* ---------- ambient background ---------- */}
      <div className="pointer-events-none fixed inset-0 z-0">
        <div
          className="absolute inset-0 opacity-[0.035]"
          style={{
            backgroundImage:
              "linear-gradient(rgba(255,255,255,0.7) 1px, transparent 1px), linear-gradient(90deg, rgba(255,255,255,0.7) 1px, transparent 1px)",
            backgroundSize: "72px 72px",
            maskImage:
              "radial-gradient(circle at 50% 20%, black 30%, transparent 80%)",
            WebkitMaskImage:
              "radial-gradient(circle at 50% 20%, black 30%, transparent 80%)",
          }}
        />
        <div className="a-float absolute -left-40 top-[-10%] h-[560px] w-[560px] rounded-full bg-white/[0.045] blur-[140px]" />
        <div className="a-float absolute -right-40 top-[40%] h-[520px] w-[520px] rounded-full bg-white/[0.03] blur-[140px]" style={{ animationDelay: "-8s" }} />
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_50%_0%,transparent_0%,rgba(0,0,0,0.6)_70%)]" />
      </div>


      <div className="relative z-10 mx-auto max-w-[1240px] px-5 pb-20 pt-14 sm:px-8 sm:pb-24 sm:pt-20 lg:px-12 lg:pb-28 lg:pt-24">
        {/* ---------- Hero ---------- */}
        <header className="grid gap-12 border-b border-white/10 pb-14 sm:pb-20 lg:grid-cols-[1.4fr_0.6fr] lg:items-end lg:gap-20">
          <div>
            <div className="a-fade-up mb-7 flex items-center gap-3" style={{ animationDelay: "60ms" }}>
              <Eyebrow> RESUME INTELLIGENCE</Eyebrow>
            </div>

            <h1
              className="a-fade-up max-w-4xl font-serif text-[52px] leading-[0.96] tracking-[-0.055em] text-white sm:text-7xl lg:text-[92px]"
              style={{ animationDelay: "140ms" }}
            >
              Your resume,
              <br />
              <span className="italic text-white/45">under the evidence.</span>
            </h1>

            <p
              className="a-fade-up mt-8 max-w-[520px] text-sm leading-8 text-white/55 sm:text-base sm:leading-8"
              style={{ animationDelay: "240ms" }}
            >
              See how your resume matches the role. Explore the evidence behind each requirement, understand which qualifications are demonstrated, and identify where your experience needs stronger evidence.
            </p>

            {/* No-login callout inside the hero */}
            <div
              className="a-fade-up mt-9 inline-flex flex-wrap items-center gap-x-7 gap-y-3 border border-white/15 bg-white/[0.02] px-5 py-3.5"
              style={{ animationDelay: "340ms" }}
            >
              {[
                "No account created",
                "No email required",
                "Runs in one session",
              ].map((item, i) => (
                <span key={item} className="flex items-center gap-2.5 text-[10px] uppercase tracking-[0.22em] text-white/65">
                  <span className={cn("h-1 w-1 rounded-full", i === 0 ? "bg-white" : "bg-white/40")} />
                  {item}
                </span>
              ))}
            </div>
          </div>

          <div
            className="a-fade-up grid grid-cols-3 border border-white/10 bg-white/[0.02] lg:mb-2"
            style={{ animationDelay: "400ms" }}
          >
            {[
              ["01", "Add role"],
              ["02", "Add CV"],
              ["03", "Review"],
            ].map(([number, label], index) => (
              <div
                key={number}
                className={cn(
                  "group relative px-3.5 py-6 transition-colors hover:bg-white/[0.03] sm:px-5",
                  index !== 2 && "border-r border-white/10"
                )}
              >
                <p className="font-serif text-2xl text-white transition-transform duration-500 group-hover:-translate-y-0.5">
                  {number}
                </p>
                <p className="mt-2.5 text-[9px] uppercase tracking-[0.16em] text-white/50 sm:text-[10px]">
                  {label}
                </p>
              </div>
            ))}
          </div>
        </header>

        {/* ---------- Input form ---------- */}
        <form
          onSubmit={handleSubmit}
          className="a-fade-up mt-14 sm:mt-16"
          style={{ animationDelay: "480ms" }}
        >
          <div className="mb-7 flex items-end justify-between gap-4">
            <div>
              <h2 className="font-serif text-3xl tracking-tight text-white sm:text-4xl">
                 The Workspace.
              </h2>
            </div>
            
          </div>

          <div className="grid gap-5 lg:grid-cols-[1.2fr_0.8fr]">
            {/* Job description */}
            <section className="group flex flex-col border border-white/10 bg-white/[0.015] p-5 transition-colors hover:border-white/20 sm:p-7 lg:p-8">
              <div className="mb-7 flex items-start justify-between">
                <div>
                  <Eyebrow>01 / The role</Eyebrow>
                  <h3 className="mt-2.5 text-lg font-medium text-white">
                    Job description
                  </h3>
                </div>

                <span className="border border-white/15 px-2.5 py-1.5 text-[9px] uppercase tracking-[0.18em] text-white/55">
                  Required
                </span>
              </div>

              <textarea
                id="jobDescription"
                value={jobDescription}
                onChange={(e) => setJobDescription(e.target.value)}
                minLength={30}
                maxLength={30000}
                required
                rows={12}
                placeholder={`Paste the job description here...\n\nInclude the responsibilities, technical skills, qualifications, and experience requirements.`}
                className="min-h-[280px] w-full flex-1 resize-y border border-white/10 bg-black p-4 text-sm leading-7 text-white outline-none transition placeholder:text-white/25 focus:border-white/45 focus:bg-white/[0.015] focus:shadow-[0_0_0_4px_rgba(255,255,255,0.05)] sm:p-5"
              />

              <div className="mt-3.5 flex items-center justify-between gap-3">
                <p className="text-[10px] text-white/40">
                  Minimum 30 characters
                </p>
                <p className="font-mono text-[10px] tabular-nums text-white/40">
                  {jobDescription.length.toLocaleString("en-US")} / 30,000
                </p>
              </div>
            </section>

            {/* Resume upload */}
            <section className="group flex flex-col border border-white/10 bg-white/[0.015] p-5 transition-colors hover:border-white/20 sm:p-7 lg:p-8">
              <div className="mb-7 flex items-start justify-between">
                <div>
                  <Eyebrow>02 / The candidate</Eyebrow>
                  <h3 className="mt-2.5 text-lg font-medium text-white">
                    Your resume
                  </h3>
                </div>

                <span className="border border-white/15 px-2.5 py-1.5 text-[9px] uppercase tracking-[0.18em] text-white/55">
                  PDF only
                </span>
              </div>

              <input
                ref={fileInputRef}
                id="resume"
                type="file"
                accept=".pdf,application/pdf"
                onChange={(e) => selectResume(e.target.files?.[0])}
                className="sr-only"
              />

              <label
                htmlFor="resume"
                onDragOver={(e) => {
                  e.preventDefault();
                  setDragging(true);
                }}
                onDragLeave={() => setDragging(false)}
                onDrop={(e) => {
                  e.preventDefault();
                  setDragging(false);
                  selectResume(e.dataTransfer.files?.[0]);
                }}
                className={cn(
                  "relative flex min-h-[280px] flex-1 cursor-pointer flex-col items-center justify-center overflow-hidden border border-dashed px-5 py-8 text-center transition-all duration-300",
                  dragging
                    ? "border-white bg-white/[0.06] scale-[1.01]"
                    : "border-white/20 bg-black hover:border-white/55 hover:bg-white/[0.02]"
                )}
              >
                <div className="mb-6 flex h-14 w-14 items-center justify-center border border-white/25 bg-white/[0.04] text-white transition-transform duration-500 group-hover:scale-105">
                  {resume ? (
                    <span className="text-xs font-semibold tracking-widest">
                      PDF
                    </span>
                  ) : (
                    <UploadIcon />
                  )}
                </div>

                <p className="max-w-full break-all text-sm font-medium text-white">
                  {resume ? resume.name : "Choose your resume"}
                </p>

                <p className="mt-2.5 max-w-xs text-xs leading-6 text-white/45">
                  {resume
                    ? `${(resume.size / 1024).toFixed(0)} KB · Select to replace`
                    : "Drop your PDF here or browse your files. Keep it readable and up to date."}
                </p>

                <span className="mt-7 inline-flex items-center gap-3 border border-white/25 px-4 py-3 text-[9px] font-semibold uppercase tracking-[0.20em] text-white transition hover:bg-white hover:text-black">
                  {resume ? "Replace file" : "Browse files"}
                  <ArrowIcon />
                </span>
              </label>

              <div className="mt-4 flex items-start gap-2 text-[10px] leading-5 text-white/40">
                <span className="mt-0.5"></span>
                <p>
                  PDF text extraction limits apply. No file is tied to an
                  account, nothing to log into.
                </p>
              </div>
            </section>
          </div>

          {error && (
            <div
              role="alert"
              className="a-fade-in mt-5 border border-white/40 bg-white/[0.04] px-5 py-4 text-sm text-white/85"
            >
              <span className="mr-2 font-semibold text-white">Notice —</span>
              {error}
            </div>
          )}

          <div className="mt-6 flex flex-col gap-6 border-b border-white/10 pb-10 sm:flex-row sm:items-center sm:justify-between">
            <div className="max-w-lg">
              <p className="text-xs leading-6 text-white/45">
                Your score reflects weighted requirement evidence, not the
                probability of receiving an interview or job offer.
              </p>
            </div>

            <button
              type="submit"
              disabled={loading}
              className="group relative flex min-h-14 w-full items-center justify-between gap-10 overflow-hidden bg-white px-6 py-4 text-left text-black transition-all duration-300 hover:bg-white/90 hover:shadow-[0_0_40px_-10px_rgba(255,255,255,0.4)] disabled:cursor-not-allowed disabled:opacity-40 sm:w-auto sm:min-w-[260px]"
            >
              {/* sweeping shimmer when loading */}
              {loading && (
                <span className="pointer-events-none absolute inset-0 overflow-hidden">
                  <span className="a-scan absolute inset-y-0 w-1/3 bg-gradient-to-r from-transparent via-black/10 to-transparent" />
                </span>
              )}

              <span className="relative text-[10px] font-semibold uppercase tracking-[0.20em]">
                {loading ? "Analysing your profile..." : "Analyse my resume"}
              </span>
              <span className="relative transition-transform duration-300 group-hover:translate-x-1 group-hover:-translate-y-0.5">
                {loading ? (
                  <span className="inline-block h-4 w-4 animate-spin rounded-full border border-black/25 border-t-black" />
                ) : (
                  <ArrowIcon diagonal />
                )}
              </span>
            </button>
          </div>
        </form>

        {/* ---------- Results ---------- */}
        {result && (
          <section
            ref={resultsRef}
            aria-live="polite"
            className="scroll-mt-24 pt-20 sm:pt-28"
          >
            <div
              className="a-fade-up mb-10 flex flex-col justify-between gap-6 border-b border-white/15 pb-8 sm:flex-row sm:items-end"
              style={{ animationDelay: "60ms" }}
            >
              <div>
                <Eyebrow>03 / Your analysis</Eyebrow>
                <h2 className="mt-3.5 font-serif text-4xl tracking-tight text-white sm:text-6xl">
                  The assessment.
                </h2>
                <p className="mt-4 text-sm text-white/55">
                  {result.candidate_name || "Candidate"}{" "}
                  <span className="mx-2.5 text-white/25">/</span>
                  {total} requirements reviewed
                </p>
              </div>

              <button
                type="button"
                onClick={() => {
                  setResult(null);
                  setError("");
                  window.scrollTo({ top: 0, behavior: "smooth" });
                }}
                className="inline-flex w-fit items-center gap-3 border border-white/25 px-4 py-3 text-[9px] uppercase tracking-[0.20em] text-white transition hover:bg-white hover:text-black"
              >
                New assessment <ArrowIcon diagonal />
              </button>
            </div>

            {/* Score */}
            <div className="grid gap-5 lg:grid-cols-[1.05fr_0.95fr]">
              <div
                className="a-fade-up border border-white/10 bg-white/[0.02] p-6 transition-colors hover:border-white/20 sm:p-10"
                style={{ animationDelay: "120ms" }}
              >
                <Eyebrow>Weighted evidence coverage</Eyebrow>

                <div className="mt-8 flex items-baseline gap-2">
                  <span className="font-serif text-8xl leading-none tracking-[-0.07em] text-white tabular-nums sm:text-9xl">
                    <ScoreNumber value={result.overall_score} />
                  </span>
                  <span className="font-serif text-3xl text-white/40">%</span>
                </div>

                <div className="mt-10 h-1 w-full overflow-hidden bg-white/[0.08]">
                  <div
                    className="a-bar h-full bg-white"
                    style={{ width: `${score}%` }}
                  />
                </div>

                <p className="mt-6 max-w-md text-xs leading-6 text-white/50">
                  The weighted share of job requirements supported by evidence
                  in your resume. This is an evidence coverage score, not a
                  hiring prediction.
                </p>
              </div>

              <div
                className="a-fade-up border border-white/10 bg-white/[0.015] p-6 transition-colors hover:border-white/20 sm:p-10"
                style={{ animationDelay: "200ms" }}
              >
                <div className="flex items-start justify-between gap-4">
                  <Eyebrow>Requirement breakdown</Eyebrow>
                  <span
                    className={cn(
                      "border px-2.5 py-1.5 text-[9px] uppercase tracking-[0.18em]",
                      result.evaluation_complete
                        ? "border-white bg-white text-black"
                        : "border-white/35 text-white/70"
                    )}
                  >
                    {result.evaluation_complete ? "Complete" : "Incomplete"}
                  </span>
                </div>

                <div className="mt-10 space-y-7">
                  {[
                    {
                      label: "Supported",
                      count: supported,
                      color: "bg-white",
                    },
                    {
                      label: "Unverified",
                      count: unverified,
                      color: "bg-white/45",
                    },
                    {
                      label: "Missing",
                      count: missing,
                      color: "bg-white/15",
                    },
                  ].map((item, i) => (
                    <div key={item.label}>
                      <div className="mb-2.5 flex items-center justify-between">
                        <span className="text-xs text-white/70">
                          {item.label}
                        </span>
                        <span className="font-mono text-xs tabular-nums text-white">
                          {String(item.count).padStart(2, "0")}
                        </span>
                      </div>

                      <div className="h-1 overflow-hidden bg-white/[0.06]">
                        <div
                          className={cn("a-bar h-full", item.color)}
                          style={{
                            width: `${total ? (item.count / total) * 100 : 0}%`,
                            animationDelay: `${200 + i * 120}ms`,
                          }}
                        />
                      </div>
                    </div>
                  ))}
                </div>

                <p className="mt-10 border-t border-white/10 pt-5 text-xs leading-6 text-white/45">
                  {result.evaluation_complete
                    ? "Review the evidence behind each requirement before deciding what to improve."
                    : "Some parts of this assessment may be incomplete. Review the warnings below."}
                </p>
              </div>
            </div>

            {/* Warnings */}
            {(!result.evaluation_complete ||
              (result.warnings?.length ?? 0) > 0) && (
              <div
                className="a-fade-up mt-5 border border-white/20 bg-white/[0.02] p-6 sm:p-7"
                style={{ animationDelay: "280ms" }}
              >
                <Eyebrow>Review notes</Eyebrow>

                {!result.evaluation_complete && (
                  <p className="mt-4 text-sm leading-6 text-white/80">
                    The evaluation did not complete fully. Treat the score
                    cautiously.
                  </p>
                )}

                {result.warnings?.map((warning, index) => (
                  <p
                    key={`${warning}-${index}`}
                    className="mt-3 text-sm leading-6 text-white/65"
                  >
                    <span className="mr-2 text-white/40">—</span>
                    {warning}
                  </p>
                ))}
              </div>
            )}

            {/* Detailed requirements */}
            <div className="mt-16">
              <div
                className="a-fade-up mb-7 flex items-end justify-between gap-4"
                style={{ animationDelay: "340ms" }}
              >
                <div>
                  <Eyebrow>Evidence, not assumptions</Eyebrow>
                  <h3 className="mt-3 font-serif text-3xl tracking-tight text-white sm:text-4xl">
                    Requirement by requirement.
                  </h3>
                </div>

                <span className="font-mono text-[10px] text-white/45">
                  {String(total).padStart(2, "0")} ITEMS
                </span>
              </div>

              <div className="border-t border-white/15">
                {result.evaluations.map((item, index) => (
                  <article
                    key={`${item.requirement}-${index}`}
                    className="a-fade-up group grid gap-6 border-b border-white/10 py-8 transition-colors hover:bg-white/[0.02] sm:grid-cols-[48px_1fr] sm:gap-8 sm:py-10"
                    style={{
                      animationDelay: `${Math.min(400 + index * 60, 1200)}ms`,
                    }}
                  >
                    <span className="font-mono text-[10px] text-white/35 transition-all duration-300 group-hover:translate-x-0.5 group-hover:text-white/70">
                      {String(index + 1).padStart(2, "0")}
                    </span>

                    <div className="min-w-0">
                      <div className="flex flex-col justify-between gap-4 sm:flex-row sm:items-start">
                        <div className="max-w-3xl">
                          <h4 className="text-base font-medium leading-7 text-white sm:text-lg">
                            {item.requirement}
                          </h4>
                          <p className="mt-2.5 text-[9px] uppercase tracking-[0.20em] text-white/40">
                            {item.importance} requirement
                          </p>
                        </div>

                        <StatusBadge status={item.status} />
                      </div>

                      {item.evidence_quote && (
                        <blockquote className="mt-7 border-l border-white/40 pl-4 transition-colors group-hover:border-white/70 sm:pl-5">
                          <Eyebrow>Found in your resume</Eyebrow>
                          <p className="mt-2.5 font-serif text-base italic leading-7 text-white/85 sm:text-lg">
                            “{item.evidence_quote}”
                          </p>
                        </blockquote>
                      )}

                      <div className="mt-7">
                        <Eyebrow>Assessment</Eyebrow>
                        <p className="mt-2.5 max-w-3xl text-sm leading-7 text-white/55">
                          {item.reasoning}
                        </p>
                      </div>
                    </div>
                  </article>
                ))}
              </div>
            </div>

            <p className="mt-7 max-w-2xl text-xs leading-6 text-white/35">
              AI-assisted analysis. Results depend on the resume and job
              description provided. They do not represent a hiring decision.
            </p>
          </section>
        )}
      </div>
    </main>
  );
}