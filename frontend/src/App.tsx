import { useMemo, useEffect, useRef, useState } from 'react';
import { ArrowRight, BookOpen, CloudUpload, Cpu, Filter, Search, Sparkles, X } from 'lucide-react';
import { RadialBar, RadialBarChart, ResponsiveContainer, Tooltip } from 'recharts';

type ModelKey =
  | 'ensemble_all'
  | 'svm'
  | 'random_forest'
  | 'xgboost'
  | 'logistic_regression';

type ModelPrediction = {
  label: string;
  confidence: number;
  probabilities?: Record<string, number>;
  model_used?: string;
  latency_ms?: number;
  xai_explanation?: string;
  error?: string;
};

type AnalysisResults = Record<ModelKey, ModelPrediction>;

const baselineModels = [
  { key: 'svm', label: 'Support Vector Machine (SVM)' },
  { key: 'random_forest', label: 'Random Forest (RF)' },
  { key: 'xgboost', label: 'XGBoost' },
  { key: 'logistic_regression', label: 'Logistic Regression (Baseline)' },
];

const sentimentTone = (label: string | undefined | null) => {
  switch ((label ?? '').toLowerCase()) {
    case 'positive':
      return { badge: 'bg-emerald-500/15 text-emerald-300', accent: '#2dd4bf' };
    case 'neutral':
      return { badge: 'bg-amber-400/10 text-amber-300', accent: '#fbbf24' };
    default:
      return { badge: 'bg-rose-500/10 text-rose-300', accent: '#fb7185' };
  }
};

/* Footer inserted to reflect mockup */
// Footer rendered as static content for now
// It appears after the main app container visually
// function Footer(): JSX.Element {
//   return (
//     <footer className="mt-12 border-t border-white/6 pt-8">
//       <div className="mx-auto max-w-7xl px-6 pb-8 text-slate-400">
//         <div className="flex flex-col items-start justify-between gap-6 sm:flex-row">
//           <div>
//             <h4 className="text-xl font-semibold text-slate-100">FinSlang-AI</h4>
//             <p className="mt-2 text-sm">Precision Sentiment for Indonesian Markets.</p>
//           </div>
//           <div className="flex gap-8 text-sm">
//             <div>
//               <p className="font-semibold text-slate-100">Privacy</p>
//               <p className="mt-1">Policy</p>
//             </div>
//             <div>
//               <p className="font-semibold text-slate-100">Terms</p>
//               <p className="mt-1">of Service</p>
//             </div>
//             <div>
//               <p className="font-semibold text-slate-100">API</p>
//               <p className="mt-1">Documentation</p>
//             </div>
//           </div>
//         </div>
//         <div className="mt-6 text-sm text-slate-500">© 2024 FinSlang-AI.</div>
//       </div>
//     </footer>
//   );
// }

type SlangEntry = {
  term: string;
  meaning: string;
  example: string;
  category: string;
};

const CATEGORY_COLORS: Record<string, string> = {
  'Profit & Rugi'      : 'bg-emerald-500/15 text-emerald-300',
  'Strategi'           : 'bg-cyan-500/15 text-cyan-300',
  'Sentimen'           : 'bg-violet-500/15 text-violet-300',
  'Kondisi Pasar'      : 'bg-amber-400/10 text-amber-300',
  'Aksi Pasar'         : 'bg-sky-500/15 text-sky-300',
  'Mekanisme Bursa'    : 'bg-slate-500/20 text-slate-300',
  'Tipe Saham'         : 'bg-orange-500/15 text-orange-300',
  'Pelaku Pasar'       : 'bg-pink-500/15 text-pink-300',
  'Analisis Teknikal'  : 'bg-teal-500/15 text-teal-300',
  'Analisis'           : 'bg-teal-500/15 text-teal-300',
  'Investasi'          : 'bg-lime-500/15 text-lime-300',
  'Indeks'             : 'bg-indigo-500/15 text-indigo-300',
  'Aksi Korporasi'     : 'bg-rose-500/15 text-rose-300',
  'Manipulasi Pasar'   : 'bg-red-500/15 text-red-300',
  'Risiko'             : 'bg-red-500/15 text-red-300',
};

function SlangSearch() {
  const [query, setQuery] = useState('');
  const [allSlang, setAllSlang] = useState<SlangEntry[]>([]);
  const [fetchError, setFetchError] = useState('');
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    fetch('/slang/search')
      .then((res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return res.json();
      })
      .then((data) => {
        if (!Array.isArray(data.results)) throw new Error('Format respons tidak valid');
        setAllSlang(data.results);
      })
      .catch((err: Error) => setFetchError(`Gagal memuat kamus slang: ${err.message}. Pastikan backend sudah di-restart.`));
  }, []);

  const filtered = useMemo(() => {
    if (!query.trim()) return allSlang;
    const q = query.trim().toLowerCase();
    return allSlang.filter(
      (e) =>
        e.term.toLowerCase().includes(q) ||
        e.meaning.toLowerCase().includes(q) ||
        e.category.toLowerCase().includes(q),
    );
  }, [query, allSlang]);

  const categories = useMemo(
    () => Array.from(new Set(allSlang.map((e) => e.category))).sort(),
    [allSlang],
  );

  return (
    <div className="rounded-[2rem] border border-white/10 bg-slate-900/80 p-8 shadow-glow">
      <div className="mb-6 flex flex-col gap-4 sm:flex-row sm:items-end sm:justify-between">
        <div>
          <p className="text-xs uppercase tracking-[0.35em] text-cyan-300/80">Financial Slang Dictionary</p>
          <h2 className="mt-2 text-2xl font-semibold text-slate-100 flex items-center gap-3">
            <BookOpen className="h-6 w-6 text-cyan-400" />
            Kamus Slang Finansial Indonesia
          </h2>
          <p className="mt-2 text-sm text-slate-400">
            Cari arti kata slang yang sering dipakai di komunitas investor dan trader Indonesia.
          </p>
        </div>
        <div className="text-sm text-slate-500">
          {allSlang.length > 0 && `${filtered.length} / ${allSlang.length} istilah`}
        </div>
      </div>

      <div className="relative mb-6">
        <Search className="absolute left-4 top-1/2 h-4 w-4 -translate-y-1/2 text-slate-500 pointer-events-none" />
        <input
          ref={inputRef}
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder='Cari istilah slang... contoh: "cuan", "nyangkut", "ARA"'
          className="w-full rounded-[1.75rem] border border-white/10 bg-slate-950/90 py-3 pl-11 pr-10 text-sm text-slate-100 outline-none transition focus:border-cyan-400 focus:ring-4 focus:ring-cyan-400/10 placeholder:text-slate-600"
        />
        {query && (
          <button
            type="button"
            onClick={() => { setQuery(''); inputRef.current?.focus(); }}
            className="absolute right-4 top-1/2 -translate-y-1/2 text-slate-500 hover:text-slate-300 transition"
          >
            <X className="h-4 w-4" />
          </button>
        )}
      </div>

      {!query && allSlang.length > 0 && (
        <div className="mb-4 flex flex-wrap gap-2">
          {categories.map((cat) => (
            <button
              key={cat}
              type="button"
              onClick={() => setQuery(cat)}
              className={`rounded-full px-3 py-1 text-xs font-medium transition hover:opacity-80 ${CATEGORY_COLORS[cat] ?? 'bg-slate-700/40 text-slate-300'}`}
            >
              {cat}
            </button>
          ))}
        </div>
      )}

      {fetchError ? (
        <p className="text-sm text-rose-300">{fetchError}</p>
      ) : filtered.length === 0 ? (
        <div className="rounded-[1.75rem] border border-white/10 bg-slate-950/60 p-8 text-center text-slate-400">
          {allSlang.length === 0 ? 'Memuat kamus...' : `Tidak ada istilah yang cocok dengan "${query}".`}
        </div>
      ) : (
        <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-3">
          {filtered.map((entry) => (
            <div
              key={entry.term}
              className="flex flex-col gap-3 rounded-[1.75rem] border border-white/10 bg-slate-950/80 p-5 shadow-[0_10px_30px_rgba(15,23,42,0.25)] transition hover:border-cyan-400/20"
            >
              <div className="flex items-start justify-between gap-2">
                <h3 className="text-lg font-semibold text-slate-100">{entry.term}</h3>
                <span className={`shrink-0 rounded-full px-2.5 py-0.5 text-xs font-medium ${CATEGORY_COLORS[entry.category] ?? 'bg-slate-700/40 text-slate-300'}`}>
                  {entry.category}
                </span>
              </div>
              <p className="text-sm leading-6 text-slate-300">{entry.meaning}</p>
              <div className="rounded-[1rem] bg-slate-900/70 px-4 py-3 text-xs leading-5 text-slate-400 italic border border-white/5">
                "{entry.example}"
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

const fetchPrediction = async (text: string, models: string[]) => {
  const url = '/predict/multi';
  const response = await fetch(url, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ text, models }),
  });

  if (!response.ok) {
    const body = await response.text();
    throw new Error(`Backend error: ${response.status} ${body}`);
  }

  return response.json();
};

function App() {
  const [text, setText] = useState('');
  const [analysis, setAnalysis] = useState<AnalysisResults | null>(null);
  const [isLoaded, setIsLoaded] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const ensembleAll = useMemo(() => {
    const r = analysis?.ensemble_all;
    if (!r || r.error) return { label: 'NEUTRAL', confidence: 0 };
    return r;
  }, [analysis]);
  const ensembleAllConfidenceText = analysis?.ensemble_all?.confidence !== undefined
    ? `${Math.round(analysis.ensemble_all.confidence * 100)}%`
    : '—';
  const ensembleAllLatencyText = analysis?.ensemble_all?.latency_ms !== undefined
    ? `${analysis.ensemble_all.latency_ms}ms`
    : '—';
  const ensembleAllConfidenceValue = analysis?.ensemble_all?.confidence !== undefined
    ? Math.round(analysis.ensemble_all.confidence * 100)
    : 0;
  const explanationText = analysis?.ensemble_all?.xai_explanation
    ?? 'Penjelasan XAI belum tersedia. Lakukan analisis untuk memperbarui.';

  const handleAnalyze = async () => {
    if (!text.trim()) {
      setError('Teks tidak boleh kosong.');
      return;
    }

    setLoading(true);
    setError('');

    try {
      const payload = await fetchPrediction(text, [
        'ensemble_all',
        'svm',
        'random_forest',
        'xgboost',
        'logistic_regression',
      ]);

      if (!payload.results) {
        throw new Error('Response backend tidak berisi hasil prediksi');
      }

      const results = Object.fromEntries(
        Object.entries(payload.results).map(([key, value]) => [key, value as ModelPrediction])
      ) as AnalysisResults;

      setAnalysis(results);
      setIsLoaded(true);
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Unknown error';
      setError(`Tidak dapat terhubung ke backend: ${message}`);
      setAnalysis(null);
      setIsLoaded(false);
    } finally {
      setLoading(false);
    }
  };

  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => {
      const content = reader.result as string;
      setText(content.slice(0, 1200));
    };
    reader.readAsText(file);
  };

  const getResult = (key: ModelKey) => {
    const r = analysis?.[key];
    if (!r || r.error) return { label: '-', confidence: 0, latency_ms: 0 };
    return r;
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100">
      <div className="mx-auto max-w-7xl px-6 py-10">
        <section className="mb-8 rounded-[2rem] border border-white/10 bg-slate-900/80 p-8 shadow-[0_30px_90px_rgba(15,23,42,0.45)] backdrop-blur-xl">
          <div className="flex flex-col gap-6 xl:flex-row xl:items-end xl:justify-between">
            <div className="max-w-3xl">
              <p className="text-xs uppercase tracking-[0.35em] text-cyan-300/70">FinSlang.AI</p>
              <h1 className="mt-4 text-4xl font-semibold tracking-tight text-slate-100 sm:text-5xl">
                Indonesian Financial Sentiment Analysis
              </h1>
              <p className="mt-4 max-w-2xl text-slate-400 sm:text-lg">
                Analisis sentimen finansial Indonesia dengan TF-IDF dan ensemble traditional ML untuk mendeteksi slang pasar modal.
              </p>
            </div>
            <div className="grid gap-4 sm:grid-cols-2 xl:w-[38%]">
              <div className="rounded-[1.75rem] border border-white/10 bg-slate-950/70 p-6 shadow-[0_20px_60px_rgba(15,23,42,0.35)]">
                <p className="text-xs uppercase tracking-[0.35em] text-slate-500">Realtime</p>
                <p className="mt-3 text-3xl font-semibold text-cyan-300">Connected</p>
                <p className="mt-2 text-sm text-slate-400">FastAPI backend aktif</p>
              </div>
              <div className="rounded-[1.75rem] border border-white/10 bg-slate-950/70 p-6 shadow-[0_20px_60px_rgba(15,23,42,0.35)]">
                <p className="text-xs uppercase tracking-[0.35em] text-slate-500">Coverage</p>
                <p className="mt-3 text-3xl font-semibold text-amber-300">Traditional ML</p>
                <p className="mt-2 text-sm text-slate-400">SVM · RF · XGB · LR + Ensemble</p>
              </div>
            </div>
          </div>
        </section>

        <main className="space-y-8">
          <div className="grid gap-6 lg:grid-cols-2">
            <div className="rounded-[2rem] border border-white/10 bg-slate-900/80 p-8 shadow-glow">
              <div className="flex flex-col gap-4 sm:flex-row sm:items-end sm:justify-between">
                <div>
                  <p className="text-xs uppercase tracking-[0.35em] text-cyan-300/80">Market Sentiment Input</p>
                  <h2 className="mt-2 text-3xl font-semibold text-slate-100">Text atau tweet pasar modal</h2>
                </div>
                <button
                  type="button"
                  className="inline-flex items-center gap-2 rounded-full bg-cyan-400 px-6 py-3 text-sm font-semibold text-slate-950 shadow-glow shadow-cyan-400/30 transition hover:bg-cyan-300 disabled:cursor-not-allowed disabled:opacity-60"
                  onClick={handleAnalyze}
                  disabled={loading}
                >
                  {loading ? 'Analyzing' : 'Analyze Sentiment'}
                </button>
              </div>
              <textarea
                value={text}
                onChange={(event) => setText(event.target.value)}
                rows={6}
                className="mt-6 w-full rounded-[1.75rem] border border-white/10 bg-slate-950/90 px-5 py-5 text-sm text-slate-100 outline-none transition focus:border-cyan-400 focus:ring-4 focus:ring-cyan-400/10"
                placeholder="Masukkan teks atau slang pasar modal di sini"
              />
              {error ? <p className="mt-4 text-sm text-rose-300">{error}</p> : null}
            </div>

            <div className="rounded-[2rem] border border-white/10 bg-gradient-to-br from-slate-900/95 to-slate-950/95 p-8 shadow-glow">
              <div className="flex flex-col gap-4 sm:flex-row sm:items-end sm:justify-between">
                <div>
                  <p className="text-xs uppercase tracking-[0.35em] text-amber-300/80">Traditional Ensemble</p>
                  <p className="mt-3 text-3xl font-semibold text-slate-100">{ensembleAll.label}</p>
                </div>
                <div className="rounded-[1.5rem] bg-slate-950/90 px-5 py-4 text-right text-slate-300 shadow-inner shadow-slate-950/20">
                  <p className="text-xs uppercase tracking-[0.35em] text-slate-500">Fused Score</p>
                  <p className="mt-2 text-4xl font-semibold text-amber-300">{ensembleAllConfidenceText}</p>
                </div>
              </div>
              <p className="mt-6 max-w-xl text-sm leading-7 text-slate-400">
                Ensemble score of 4 traditional models (SVM, Random Forest, XGBoost, Logistic Regression).
              </p>
            </div>
          </div>

          <div className="rounded-[2rem] border border-white/10 bg-slate-900/80 p-6 shadow-glow">
            <div className="mb-6 flex flex-col gap-2">
              <p className="text-xs uppercase tracking-[0.35em] text-cyan-300/80">Model Comparison Matrix</p>
              <h2 className="text-2xl font-semibold text-slate-100">Performa setiap arsitektur</h2>
            </div>
            <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
              {[
                { key: 'svm', label: 'SVM' },
                { key: 'random_forest', label: 'RF' },
                { key: 'xgboost', label: 'XGBoost' },
                { key: 'logistic_regression', label: 'LOGREG' },
              ].map((model) => {
                const data = getResult(model.key as ModelKey);
                const tone = sentimentTone(data.label);
                return (
                  <div key={model.key} className="rounded-[1.75rem] border border-white/10 bg-slate-950/80 p-4 text-center shadow-[0_20px_50px_rgba(15,23,42,0.25)]">
                    <div className="mx-auto mb-4 h-24 w-24 rounded-full bg-slate-900/80 p-4 ring-1 ring-white/10">
                      <div className="relative h-full w-full rounded-full bg-slate-900/90">
                        <div
                          className="absolute inset-0 rounded-full"
                          style={{
                            background: `conic-gradient(${tone.accent} ${Math.round(data.confidence * 100)}%, rgba(148,163,184,0.08) 0)`,
                          }}
                        />
                        <div className="absolute inset-4 flex items-center justify-center rounded-full bg-slate-950/95 text-slate-100">
                          <span className="text-xl font-semibold">{Math.round(data.confidence * 100)}%</span>
                        </div>
                      </div>
                    </div>
                    <p className="text-sm font-semibold uppercase tracking-[0.3em] text-slate-300">{model.label}</p>
                    <p className={`mt-2 text-sm font-semibold ${tone.badge}`}>{data.label}</p>
                  </div>
                );
              })}
            </div>
          </div>

          <div className="rounded-[2rem] border border-white/10 bg-slate-900/80 p-8 shadow-glow">
              <div className="flex flex-col gap-4">
                <div className="flex items-start justify-between gap-4">
                  <div>
                    <p className="text-xs uppercase tracking-[0.35em] text-cyan-300/80">Explainable AI Interpretation</p>
                    <h2 className="mt-2 text-2xl font-semibold text-slate-100">Insight-driven sentiment narrative</h2>
                  </div>
                  <div className="flex items-center gap-3">
                    <div className="rounded-[0.75rem] bg-slate-950/80 px-3 py-2 text-sm text-slate-300">Latency: {ensembleAllLatencyText}</div>
                  </div>
                </div>

                <div className="grid gap-4 lg:grid-cols-[1fr_220px]">
                  <div>
                    <div className="rounded-[1.75rem] border border-white/10 bg-slate-950/80 p-6 text-sm leading-7 text-slate-300 shadow-inner shadow-slate-950/20">
                      {explanationText}
                    </div>
                  </div>

                  <div className="space-y-4">
                    <div className="rounded-[1.75rem] bg-slate-950/80 p-4 text-sm text-slate-300 shadow-inner shadow-slate-950/20">
                      <p className="text-xs uppercase tracking-[0.35em] text-slate-500">Confidence</p>
                      <div className="mt-3 flex items-center gap-4">
                        <div className="flex-1">
                          <div className="h-3 w-full rounded-full bg-slate-900">
                            <div
                              className="h-full rounded-full bg-cyan-400"
                              style={{ width: `${ensembleAllConfidenceValue}%` }}
                            />
                          </div>
                        </div>
                        <div className="text-xl font-semibold text-slate-100">{ensembleAllConfidenceText}</div>
                      </div>
                    </div>

                    <div className="rounded-[1.75rem] bg-slate-950/80 p-4 text-sm text-slate-300 shadow-inner shadow-slate-950/20">
                      <p className="text-xs uppercase tracking-[0.35em] text-slate-500">Consensus</p>
                      <p className="mt-2 text-xl font-semibold text-slate-100">{ensembleAll.label}</p>
                    </div>
                  </div>
                </div>
              </div>
            </div>

          <SlangSearch />
        </main>
      </div>
    </div>
  );
}

// render footer immediately after main app
// note: kept in-file for simplicity
const AppWithFooter: React.FC = () => (
  <>
    <App />
    {/* <Footer /> */}
  </>
);

export default AppWithFooter;

