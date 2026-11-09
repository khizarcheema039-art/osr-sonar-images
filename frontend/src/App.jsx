import React, { useState, useEffect } from 'react';
import Rail from './components/Rail';
import Figure from './components/Figure';
import Lightbox from './components/Lightbox';
import ProblemDemo from './components/ProblemDemo';
import DataSplit from './components/DataSplit';
import Baseline from './components/Baseline';
import Methods from './components/Methods';
import ResultsTable from './components/ResultsTable';
import ThresholdExplorer from './components/ThresholdExplorer';
import EmbeddingSpace from './components/EmbeddingSpace';
import ReproduceAndAbout from './components/ReproduceAndAbout';
import ShortcutsModal from './components/ShortcutsModal';
import LivePredictor from './components/LivePredictor';
import { PROJECT_METADATA, FIGURES_CATALOG, OSR_METHODS } from './data/results';

export default function App() {
  const [theme, setTheme] = useState(() => {
    return localStorage.getItem('osr_theme') || 'light';
  });

  const [selectedMethod, setSelectedMethod] = useState(() => {
    return localStorage.getItem('osr_selected_method') || 'MetricLearning';
  });

  const [hoveredMethod, setHoveredMethod] = useState(null);
  const [activeSection, setActiveSection] = useState('abstract');
  const [lightboxIndex, setLightboxIndex] = useState(null);
  const [shortcutsOpen, setShortcutsOpen] = useState(false);

  useEffect(() => {
    if (theme === 'dark') {
      document.documentElement.classList.add('dark');
    } else {
      document.documentElement.classList.remove('dark');
    }
    localStorage.setItem('osr_theme', theme);
  }, [theme]);

  useEffect(() => {
    localStorage.setItem('osr_selected_method', selectedMethod);
  }, [selectedMethod]);

  const toggleTheme = () => {
    setTheme((prev) => (prev === 'light' ? 'dark' : 'light'));
  };

  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;

      if (e.key === '1') {
        setSelectedMethod('MSP');
      } else if (e.key === '2') {
        setSelectedMethod('SVDD');
      } else if (e.key === '3') {
        setSelectedMethod('MetricLearning');
      } else if (e.key === '?') {
        e.preventDefault();
        setShortcutsOpen((prev) => !prev);
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, []);

  useEffect(() => {
    const sectionIds = [
      'abstract',
      'predict',
      'problem',
      'data',
      'baseline',
      'methods',
      'results',
      'explorer',
      'embeddings',
      'reproduce',
    ];

    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            setActiveSection(entry.target.id);
          }
        });
      },
      { rootMargin: '-15% 0px -70% 0px' }
    );

    sectionIds.forEach((id) => {
      const el = document.getElementById(id);
      if (el) observer.observe(el);
    });

    return () => observer.disconnect();
  }, []);

  const handleOpenLightbox = (figure) => {
    const idx = FIGURES_CATALOG.findIndex((f) => f.file === figure.file);
    if (idx !== -1) {
      setLightboxIndex(idx);
    } else {
      setLightboxIndex(0);
    }
  };

  const handleNextLightbox = () => {
    setLightboxIndex((prev) => (prev + 1) % FIGURES_CATALOG.length);
  };

  const handlePrevLightbox = () => {
    setLightboxIndex((prev) => (prev - 1 + FIGURES_CATALOG.length) % FIGURES_CATALOG.length);
  };

  return (
    <div className="min-h-screen bg-paper dark:bg-paper-dark text-ink dark:text-ink-dark transition-colors duration-200">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6 lg:py-12 flex flex-col lg:flex-row gap-8">
        {/* Sticky Left Rail (desktop) / Collapsible Top Bar (mobile) */}
        <Rail
          activeSection={activeSection}
          selectedMethod={selectedMethod}
          onSelectMethod={setSelectedMethod}
          theme={theme}
          onToggleTheme={toggleTheme}
          onOpenShortcuts={() => setShortcutsOpen(true)}
        />

        {/* Main Content Column */}
        <main className="flex-1 max-w-reading mx-auto lg:mx-0 w-full">
          {/* SECTION 1: HEADER & ABSTRACT */}
          <section id="abstract" className="mb-14 pt-2">
            <header className="border-b border-rule dark:border-rule-dark pb-8 mb-8">
              <div className="text-[11px] font-sans uppercase tracking-widest text-accent font-semibold mb-2">
                Department of Computer Science &bull; Bahria University, Islamabad
              </div>

              {/* Title set large, left-aligned, crisp modern human typography */}
              <h1 className="font-display text-3xl sm:text-4xl lg:text-[40px] font-extrabold text-ink dark:text-ink-dark leading-[1.18] tracking-tight text-left">
                {PROJECT_METADATA.title}
              </h1>

              {/* Institution and Program */}
              <p className="text-xs font-sans text-muted dark:text-muted-dark mt-2 font-medium">
                Final Year Research Project &bull; Acoustic Vision & Machine Intelligence Lab
              </p>

              {/* Authors & Supervisor paper byline */}
              <div className="mt-4 pt-4 border-t border-rule/50 dark:border-rule-dark/50 flex flex-wrap items-center gap-x-4 gap-y-1 text-xs font-sans text-muted dark:text-muted-dark">
                <span>
                  By <strong className="text-ink dark:text-ink-dark font-semibold">Mohammad Khizar Hayat Cheema</strong> (01-134231-039)
                </span>
                <span>&bull;</span>
                <span>
                  <strong className="text-ink dark:text-ink-dark font-semibold">Muhammad Haroon</strong> (01-134231-052)
                </span>
                <span>&bull;</span>
                <span>
                  Supervisor: <strong className="text-ink dark:text-ink-dark font-semibold">{PROJECT_METADATA.supervisor}</strong>
                </span>
              </div>
            </header>

            {/* Plain-Language Abstract (Clear, readable, high typographic standard) */}
            <div className="bg-paper dark:bg-panel border-l-2 border-accent p-5 font-sans text-[15px] sm:text-base text-ink/90 dark:text-ink-dark/90 leading-relaxed space-y-3 rounded-r-sm">
              <div className="font-display text-xs uppercase tracking-wider text-accent font-bold">
                Abstract
              </div>
              <p>
                In underwater acoustic perception, autonomous subsea inspection systems rely on closed-set deep classifiers that are forced to assign every acoustic echo to one of several predefined classes. When an autonomous underwater vehicle encounters unmodeled debris, topological clutter, or novel maritime structures, the closed-set network cannot abstain and issues dangerously confident incorrect predictions. Open-set recognition (OSR) mitigates this open-space risk by simultaneously classifying known targets and actively rejecting unmodeled objects.
              </p>
              <p>
                Using the Marine Debris Forward-Looking Sonar (MD-FLS) dataset acquired with an ARIS Explorer 3000 sensor, we benchmark three distinct open-set formulations: Maximum Softmax Probability (MSP), Deep SVDD nearest centroid distance, and Deep Metric Learning with Multi-Similarity triplet loss. Our findings reveal that while Metric Learning optimizes global separation (AUROC 0.9255, AP 0.8448, TPR 0.9410), Deep SVDD establishes the strictest safety barrier against false acceptances (FPR 0.1171, UDR 0.8829), exposing distinct trade-offs for real-world robotic deployment.
              </p>
            </div>
          </section>

          {/* NEW SECTION: LIVE PREDICTION & IMAGE UPLOAD */}
          <section id="predict" className="mb-16 scroll-mt-10">
            <h2 className="font-display text-2xl font-bold tracking-tight text-ink dark:text-ink-dark mb-1">
              Live Sonar Inference & Model Evaluation
            </h2>
            <p className="text-sm text-ink/90 dark:text-ink-dark/90 leading-relaxed font-sans mb-3">
              Upload an acoustic sonar frame below or test with prepared sonar targets. The pipeline executes 3×3 median filtering, 
              local CLAHE contrast equalisation, and passes the 224×224 tensor through the open-set rejection boundaries in real time.
            </p>
            <LivePredictor />
          </section>

          {/* SECTION 2: THE PROBLEM */}
          <section id="problem" className="mb-16 scroll-mt-10">
            <h2 className="font-display text-2xl font-bold tracking-tight text-ink dark:text-ink-dark mb-1">
              2. The Closed-World Failure Mode
            </h2>
            <p className="text-sm text-ink/90 dark:text-ink-dark/90 leading-relaxed font-sans">
              Autonomous subsea vehicles operate in uncontrolled marine environments where acoustic sensor echoes constantly 
              originate from unanticipated objects. Standard deep neural networks assume a static, closed universe: given <span className="font-mono text-accent">K</span> classes, 
              the softmax function normalizes output logits so their sum equals 1.0. Consequently, when presented with a completely novel acoustic object, 
              the model cannot output an unassigned state and is mathematically coerced into misclassifying the unknown sample with unwarranted certainty.
            </p>

            <ProblemDemo />
          </section>

          {/* SECTION 3: DATA & SPLIT */}
          <section id="data" className="mb-16 scroll-mt-10">
            <h2 className="font-display text-2xl font-bold tracking-tight text-ink dark:text-ink-dark mb-1">
              3. Dataset Partition & Preprocessing Pipeline
            </h2>
            <DataSplit onOpenLightbox={handleOpenLightbox} />
          </section>

          {/* SECTION 4: CLOSED-SET BASELINE */}
          <section id="baseline" className="mb-16 scroll-mt-10">
            <h2 className="font-display text-2xl font-bold tracking-tight text-ink dark:text-ink-dark mb-1">
              4. Closed-Set Baseline: The Illusion of Perfection
            </h2>
            <Baseline onOpenLightbox={handleOpenLightbox} />
          </section>

          {/* SECTION 5: OSR METHODS */}
          <section id="methods" className="mb-16 scroll-mt-10">
            <h2 className="font-display text-2xl font-bold tracking-tight text-ink dark:text-ink-dark mb-1">
              5. Open-Set Recognition Formulations
            </h2>
            <Methods
              selectedMethod={selectedMethod}
              onSelectMethod={setSelectedMethod}
            />
          </section>

          {/* SECTION 6: BENCHMARK RESULTS */}
          <section id="results" className="mb-16 scroll-mt-10">
            <h2 className="font-display text-2xl font-bold tracking-tight text-ink dark:text-ink-dark mb-1">
              6. Benchmark Results & Comparative Analysis
            </h2>
            <ResultsTable
              selectedMethod={selectedMethod}
              onSelectMethod={setSelectedMethod}
              hoveredMethod={hoveredMethod}
              onHoverMethod={setHoveredMethod}
              onOpenLightbox={handleOpenLightbox}
            />
          </section>

          {/* SECTION 7: THRESHOLD EXPLORER */}
          <section id="explorer" className="mb-16 scroll-mt-10">
            <h2 className="font-display text-2xl font-bold tracking-tight text-ink dark:text-ink-dark mb-1">
              7. Operational Threshold Explorer
            </h2>
            <ThresholdExplorer
              selectedMethod={selectedMethod}
              onSelectMethod={setSelectedMethod}
            />
          </section>

          {/* SECTION 8: EMBEDDING SPACE */}
          <section id="embeddings" className="mb-16 scroll-mt-10">
            <h2 className="font-display text-2xl font-bold tracking-tight text-ink dark:text-ink-dark mb-1">
              8. Latent Metric Geometry (t-SNE Manifolds)
            </h2>
            <EmbeddingSpace onOpenLightbox={handleOpenLightbox} />
          </section>

          {/* SECTION 9: REPRODUCE & CITATIONS */}
          <section id="reproduce" className="mb-16 scroll-mt-10">
            <h2 className="font-display text-2xl font-bold tracking-tight text-ink dark:text-ink-dark mb-1">
              9. Reproduction Protocol & Academic References
            </h2>
            <ReproduceAndAbout />
          </section>

          {/* Colophon footer */}
          <footer className="pt-8 pb-16 border-t border-rule dark:border-rule-dark text-xs font-sans text-muted dark:text-muted-dark flex flex-wrap justify-between items-center gap-4">
            <div>
              &copy; 2026 Bahria University &bull; Final Year Project
            </div>
            <div>
              Sound Metrics ARIS Explorer 3000 Sonar Research
            </div>
          </footer>
        </main>
      </div>

      {/* Global Lightbox Component */}
      <Lightbox
        isOpen={lightboxIndex !== null}
        figure={lightboxIndex !== null ? FIGURES_CATALOG[lightboxIndex] : null}
        onClose={() => setLightboxIndex(null)}
        onNext={handleNextLightbox}
        onPrev={handlePrevLightbox}
        currentIndex={lightboxIndex || 0}
        totalCount={FIGURES_CATALOG.length}
      />

      {/* Keyboard Shortcuts Dialog */}
      <ShortcutsModal
        isOpen={shortcutsOpen}
        onClose={() => setShortcutsOpen(false)}
      />
    </div>
  );
}
