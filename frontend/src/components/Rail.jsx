import React, { useState } from 'react';

const SECTIONS = [
  { id: 'abstract', label: '1. Abstract & Byline' },
  { id: 'predict', label: '★ Live Inference & Upload' },
  { id: 'problem', label: '2. The Closed-World Failure' },
  { id: 'data', label: '3. Data & Splits' },
  { id: 'baseline', label: '4. Closed-Set Baseline' },
  { id: 'methods', label: '5. OSR Formulations' },
  { id: 'results', label: '6. Benchmark Results' },
  { id: 'explorer', label: '7. Threshold Explorer' },
  { id: 'embeddings', label: '8. Embedding Geometry' },
  { id: 'reproduce', label: '9. Reproduce & Citations' },
];

export default function Rail({ activeSection, selectedMethod, onSelectMethod, theme, onToggleTheme, onOpenShortcuts }) {
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  const scrollTo = (id) => {
    const el = document.getElementById(id);
    if (el) {
      el.scrollIntoView({ behavior: 'smooth' });
      setMobileMenuOpen(false);
    }
  };

  return (
    <>
      {/* Mobile Top Bar */}
      <header className="lg:hidden sticky top-0 z-40 bg-paper/95 dark:bg-paper-dark/95 backdrop-blur-sm border-b border-rule dark:border-rule-dark px-4 py-2.5 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className="font-display text-xs font-bold tracking-tight text-accent">SONAR OSR</span>
          <span className="text-xs text-muted dark:text-muted-dark font-sans truncate max-w-[170px]">
            {activeSection ? SECTIONS.find(s => s.id === activeSection)?.label : 'Research'}
          </span>
        </div>
        <div className="flex items-center gap-3">
          <button
            onClick={onToggleTheme}
            className="text-xs font-sans text-muted dark:text-muted-dark hover:text-ink dark:hover:text-ink-dark transition-colors"
          >
            {theme === 'dark' ? 'Light' : 'Dark'}
          </button>
          <button
            onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
            className="text-xs font-sans px-2.5 py-1 border border-rule dark:border-rule-dark text-ink dark:text-ink-dark"
            aria-label="Toggle Navigation"
          >
            {mobileMenuOpen ? 'Close' : 'Menu'}
          </button>
        </div>
      </header>

      {/* Mobile Dropdown Nav */}
      {mobileMenuOpen && (
        <nav 
          aria-label="Mobile Navigation"
          className="lg:hidden fixed top-[45px] left-0 right-0 z-40 bg-paper dark:bg-paper-dark border-b border-rule dark:border-rule-dark p-4 shadow-md"
        >
          <div className="flex flex-col space-y-1.5 font-sans text-xs">
            {SECTIONS.map((sec) => (
              <button
                key={sec.id}
                onClick={() => scrollTo(sec.id)}
                className={`text-left py-1.5 px-2 border-l-2 transition-colors ${
                  activeSection === sec.id
                    ? 'border-accent text-accent font-semibold bg-accent/5'
                    : 'border-transparent text-muted dark:text-muted-dark hover:text-ink dark:hover:text-ink-dark'
                }`}
              >
                {sec.label}
              </button>
            ))}
            <div className="pt-2 border-t border-rule dark:border-rule-dark flex items-center justify-between text-muted text-[11px]">
              <span>Active: {selectedMethod}</span>
              <button onClick={onOpenShortcuts} className="underline text-accent">
                Keyboard shortcuts [?]
              </button>
            </div>
          </div>
        </nav>
      )}

      {/* Desktop Sticky Left Rail */}
      <aside className="hidden lg:block w-64 shrink-0 pr-8">
        <div className="sticky top-10 flex flex-col justify-between h-[calc(100vh-5rem)]">
          <div>
            {/* Project Header Tag */}
            <div className="pb-4 mb-6 border-b border-rule dark:border-rule-dark">
              <div className="text-[11px] font-display tracking-wider uppercase text-accent font-bold">
                Research Report &bull; 2026
              </div>
              <div className="text-xs font-sans text-muted dark:text-muted-dark mt-0.5 font-medium">
                Bahria University, Islamabad
              </div>
            </div>

            {/* Navigation Sections */}
            <nav className="space-y-1 font-sans text-xs" aria-label="Section Index">
              <div className="text-[10px] uppercase tracking-wider text-muted/80 dark:text-muted-dark/80 mb-2 font-semibold font-display">
                Table of Contents
              </div>
              {SECTIONS.map((sec) => {
                const isActive = activeSection === sec.id;
                const isPredict = sec.id === 'predict';
                return (
                  <button
                    key={sec.id}
                    onClick={() => scrollTo(sec.id)}
                    className={`block w-full text-left py-1 px-2.5 transition-all duration-150 border-l-2 text-[12.5px] leading-snug rounded-sm ${
                      isActive
                        ? 'border-accent text-accent font-semibold bg-accent/5'
                        : isPredict
                        ? 'border-accent/40 text-accent font-medium hover:bg-accent/5'
                        : 'border-transparent text-muted dark:text-muted-dark hover:text-ink dark:hover:text-ink-dark hover:border-rule'
                    }`}
                  >
                    {sec.label}
                  </button>
                );
              })}
            </nav>

            {/* Active Method Switcher in Rail */}
            <div className="mt-7 pt-4 border-t border-rule dark:border-rule-dark font-sans">
              <div className="text-[10px] uppercase tracking-wider text-muted/80 dark:text-muted-dark/80 mb-2 font-semibold font-display">
                Active Benchmark Model
              </div>
              <div className="flex flex-col gap-1 text-xs">
                {[
                  { key: 'MSP', label: '1. Softmax Baseline (MSP)', num: '1' },
                  { key: 'SVDD', label: '2. Deep SVDD (Centroid)', num: '2' },
                  { key: 'MetricLearning', label: '3. Metric Learning (Triplet)', num: '3' },
                ].map((item) => (
                  <button
                    key={item.key}
                    onClick={() => onSelectMethod(item.key)}
                    className={`text-left px-2 py-1 text-xs border transition-colors flex items-center justify-between rounded-sm ${
                      selectedMethod === item.key
                        ? 'border-accent text-accent bg-accent/5 font-semibold'
                        : 'border-transparent text-muted dark:text-muted-dark hover:text-ink dark:hover:text-ink-dark'
                    }`}
                  >
                    <span>{item.label}</span>
                    <span className="font-mono text-[10px] text-muted/60 opacity-60">[{item.num}]</span>
                  </button>
                ))}
              </div>
            </div>
          </div>

          {/* Footer of Rail: Theme Toggle & Shortcut Hint */}
          <div className="pt-4 border-t border-rule dark:border-rule-dark font-sans text-xs flex items-center justify-between text-muted dark:text-muted-dark">
            <button
              onClick={onToggleTheme}
              className="hover:text-ink dark:hover:text-ink-dark transition-colors underline decoration-dotted underline-offset-4"
              title="Toggle theme (Light / Dark)"
            >
              {theme === 'dark' ? 'Theme: Light' : 'Theme: Dark'}
            </button>
            <button
              onClick={onOpenShortcuts}
              className="px-2 py-0.5 border border-rule dark:border-rule-dark hover:border-accent hover:text-accent font-mono transition-colors text-[11px] rounded-sm"
              title="Keyboard shortcuts"
            >
              ?
            </button>
          </div>
        </div>
      </aside>
    </>
  );
}
