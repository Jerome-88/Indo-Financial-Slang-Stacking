import { useMemo, useState } from 'react';
import { ArrowRight, CloudUpload, Cpu, Filter, Sparkles } from 'lucide-react';
import { RadialBar, RadialBarChart, ResponsiveContainer, Tooltip } from 'recharts';

type ModelKey =
  | 'ensemble'
  | 'ensemble_all'
  | 'indobert'
  | 'indoroberta'
  | 'xlmr'
  | 'svm'
  | 'random_forest'
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

const tierModels = [
  { key: 'indobert', label: 'IndoBERT (Base)' },
  { key: 'indoroberta', label: 'IndoBERTweet (Uncased)' },
  { key: 'xlmr', label: 'XLM-RoBERTa (Multilingual)' },
];

const baselineModels = [
  { key: 'svm', label: 'Support Vector Machine (SVM)' },
  { key: 'random_forest', label: 'Random Forest (RF)' },
  { key: 'logistic_regression', label: 'Logistic Regression (Baseline)' },
];

const sentimentTone = (label: string) => {
  switch (label.toLowerCase()) {
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

  const ensemble = useMemo(
    () => analysis?.ensemble ?? { label: 'NEUTRAL', confidence: 0 },
    [analysis]
  );
  const ensembleAll = useMemo(
    () => analysis?.ensemble_all ?? { label: 'NEUTRAL', confidence: 0 },
    [analysis]
  );
  const sentiment = sentimentTone(ensemble.label);
  const ensembleConfidenceText = analysis?.ensemble?.confidence !== undefined
    ? `${Math.round(analysis.ensemble.confidence * 100)}%`
    : '—';
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
        'ensemble',
        'ensemble_all',
        'indobert',
        'indoroberta',
        'xlmr',
        'svm',
        'random_forest',
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

  const getResult = (key: ModelKey) =>
    analysis?.[key] ?? { label: '-', confidence: 0, latency_ms: 0 };

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
                Sentimen finansial Indonesia dengan kombinasi karakter informal atau slang dalam media sosial dengan menggunakan model ensemble.
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
                <p className="mt-3 text-3xl font-semibold text-amber-300">Deep + Baseline</p>
                <p className="mt-2 text-sm text-slate-400">Arsitektur ensemble lengkap</p>
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
            <div className="grid gap-4 sm:grid-cols-3">
              {[
                { key: 'indobert', label: 'IndoBERT' },
                { key: 'indoroberta', label: 'BERTweet' },
                { key: 'xlmr', label: 'XLM-RoB' },
                { key: 'svm', label: 'SVM' },
                { key: 'random_forest', label: 'RF' },
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

          <div className="grid gap-6 lg:grid-cols-2">
            <div className="rounded-[2rem] border border-white/10 bg-gradient-to-br from-slate-900/95 to-slate-950/95 p-8 shadow-glow">
              <div className="flex flex-col gap-4 sm:flex-row sm:items-end sm:justify-between">
                <div>
                  <p className="text-xs uppercase tracking-[0.35em] text-cyan-300/80">Deep Ensemble Consensus</p>
                  <p className="mt-3 text-3xl font-semibold text-slate-100">{ensemble.label}</p>
                </div>
                <div className="rounded-[1.5rem] bg-slate-950/90 px-5 py-4 text-right text-slate-300 shadow-inner shadow-slate-950/20">
                  <p className="text-xs uppercase tracking-[0.35em] text-slate-500">Score</p>
                  <p className="mt-2 text-4xl font-semibold text-cyan-300">{ensembleConfidenceText}</p>
                </div>
              </div>
              <p className="mt-6 max-w-xl text-sm leading-7 text-slate-400">
                Consensus derived from IndoBERT architectures. Hasil ini menunjukkan kekuatan model transformer murni dalam menangkap nuansa sentimen pasar modal Indonesia.
              </p>
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
          </div>
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

