import React, { useState, useEffect, useMemo, useRef } from 'react';
import { OSR_METHODS } from '../data/results';

export default function ThresholdExplorer({ selectedMethod, onSelectMethod }) {
  const [scoresData, setScoresData] = useState(null);
  const [isLive, setIsLive] = useState(false);
  const [loadError, setLoadError] = useState(false);

  const currentMethod = OSR_METHODS[selectedMethod] || OSR_METHODS.MetricLearning;

  // Threshold slider state, initialised to the real reported threshold
  const [threshold, setThreshold] = useState(currentMethod.threshold);

  // Sync threshold when selectedMethod changes
  useEffect(() => {
    setThreshold(currentMethod.threshold);
  }, [selectedMethod]);

  // Attempt to load per-sample scores from /data/scores.json
  useEffect(() => {
    let isMounted = true;
    fetch('/data/scores.json')
      .then((res) => {
        if (!res.ok) throw new Error('File not found');
        return res.json();
      })
      .then((data) => {
        if (!isMounted) return;
        if (data && data.MSP && data.SVDD && data.MetricLearning) {
          setScoresData(data);
          setIsLive(true);
        } else {
          setLoadError(true);
        }
      })
      .catch(() => {
        if (isMounted) {
          setLoadError(true);
          setIsLive(false);
        }
      });

    return () => {
      isMounted = false;
    };
  }, []);

  // Keyboard controls: [ and ] nudge threshold
  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.target.tagName === 'INPUT' && e.target.type !== 'range') return;
      if (e.key === '[') {
        e.preventDefault();
        setThreshold((prev) => {
          const step = currentMethod.thresholdStep || 0.005;
          const [min] = currentMethod.thresholdRange;
          return Math.max(min, Number((prev - step).toFixed(4)));
        });
      } else if (e.key === ']') {
        e.preventDefault();
        setThreshold((prev) => {
          const step = currentMethod.thresholdStep || 0.005;
          const [, max] = currentMethod.thresholdRange;
          return Math.min(max, Number((prev + step).toFixed(4)));
        });
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [currentMethod]);

  // When live scores exist, compute empirical TPR, FPR, counts and histograms
  const liveStats = useMemo(() => {
    if (!isLive || !scoresData || !scoresData[selectedMethod]) {
      return null;
    }
    const { known, unknown } = scoresData[selectedMethod];
    if (!known || !unknown || known.length === 0 || unknown.length === 0) return null;

    // Known accepted: score >= threshold
    const knownAccepted = known.filter((s) => s >= threshold).length;
    const tpr = knownAccepted / known.length;

    // Unknown falsely accepted: score >= threshold
    const unknownAccepted = unknown.filter((s) => s >= threshold).length;
    const fpr = unknownAccepted / unknown.length;

    // Unknown rejected: score < threshold
    const unknownRejected = unknown.length - unknownAccepted;
    const udr = unknownRejected / unknown.length;

    // Binning for SVG Histogram (30 bins)
    const allScores = [...known, ...unknown];
    const minScore = Math.min(...allScores);
    const maxScore = Math.max(...allScores);
    const numBins = 32;
    const binWidth = (maxScore - minScore) / numBins;

    const knownBins = new Array(numBins).fill(0);
    const unknownBins = new Array(numBins).fill(0);

    known.forEach((s) => {
      const b = Math.min(numBins - 1, Math.max(0, Math.floor((s - minScore) / binWidth)));
      knownBins[b]++;
    });
    unknown.forEach((s) => {
      const b = Math.min(numBins - 1, Math.max(0, Math.floor((s - minScore) / binWidth)));
      unknownBins[b]++;
    });

    const maxCount = Math.max(...knownBins, ...unknownBins, 1);

    return {
      tpr,
      fpr,
      udr,
      knownTotal: known.length,
      knownAccepted,
      unknownTotal: unknown.length,
      unknownRejected,
      numBins,
      minScore,
      maxScore,
      binWidth,
      knownBins,
      unknownBins,
      maxCount,
    };
  }, [isLive, scoresData, selectedMethod, threshold]);

  const [minBound, maxBound] = currentMethod.thresholdRange;
  const isOriginalThreshold = Math.abs(threshold - currentMethod.threshold) < 0.0001;

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-center justify-between gap-3 pb-3 border-b border-rule dark:border-rule-dark">
        <div>
          <h4 className="font-mono text-xs font-semibold uppercase tracking-wider text-ink dark:text-ink-dark">
            Interactive Operational Threshold Simulator
          </h4>
          <span className="text-[11px] text-muted dark:text-muted-dark font-sans">
            Adjust rejection boundary &tau; to evaluate TPR vs FPR risk trade-offs
          </span>
        </div>

        {/* Method selector */}
        <div className="flex gap-1 font-mono text-xs">
          {Object.values(OSR_METHODS).map((m) => (
            <button
              key={m.key}
              onClick={() => onSelectMethod(m.key)}
              className={`px-2.5 py-1 border transition-colors ${
                selectedMethod === m.key
                  ? 'border-accent bg-accent/10 text-accent font-semibold'
                  : 'border-rule dark:border-rule-dark text-muted dark:text-muted-dark hover:text-ink dark:hover:text-ink-dark'
              }`}
            >
              [{m.index}] {m.shortLabel}
            </button>
          ))}
        </div>
      </div>

      {/* Fallback Notice when scores.json does not exist */}
      {!isLive && (
        <div className="p-4 border border-accent/40 bg-accent/5 dark:bg-panel text-xs space-y-2">
          <div className="flex items-center gap-2 font-mono text-accent font-semibold">
            <span className="w-2 h-2 rounded-full bg-accent inline-block"></span>
            <span>Operating Point View (Export Notice)</span>
          </div>
          <p className="font-mono text-ink/90 dark:text-ink-dark/90 leading-relaxed">
            Per-sample scores have not been exported yet. Run <code className="bg-paper dark:bg-panel border border-rule/60 px-1.5 py-0.5 text-accent font-semibold">python scripts/export_scores.py</code> to enable the live slider.
          </p>
          <p className="text-[11px] text-muted dark:text-muted-dark font-sans leading-normal">
            Displaying the three verified benchmark operating thresholds below. The codebase includes the complete <code className="font-mono">scripts/export_scores.py</code> exporter to extract all per-sample test inferences directly from PyTorch checkpoints.
          </p>
        </div>
      )}

      {/* 3 Real Operating Points Comparison Grid */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-3 font-mono text-xs">
        {Object.values(OSR_METHODS).map((m) => {
          const isSelected = selectedMethod === m.key;
          return (
            <div
              key={m.key}
              onClick={() => onSelectMethod(m.key)}
              className={`cursor-pointer border p-3 transition-colors ${
                isSelected
                  ? 'border-accent bg-accent/10 dark:bg-panel'
                  : 'border-rule dark:border-rule-dark bg-paper dark:bg-panel/50 hover:border-rule-dark'
              }`}
            >
              <div className="flex items-center justify-between pb-1.5 mb-2 border-b border-rule/40 dark:border-rule-dark/40">
                <span className={`font-semibold ${isSelected ? 'text-accent' : 'text-ink dark:text-ink-dark'}`}>
                  {m.shortLabel}
                </span>
                <span className="text-[10px] text-muted">τ = {m.threshold > 0 ? `+${m.threshold}` : m.threshold}</span>
              </div>
              <div className="space-y-1 text-[11px]">
                <div className="flex justify-between">
                  <span className="text-muted dark:text-muted-dark">AUROC:</span>
                  <span className="font-semibold">{m.auroc.toFixed(4)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted dark:text-muted-dark">Reported TPR:</span>
                  <span className="text-teal-sonar font-semibold">{(m.tpr * 100).toFixed(2)}%</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted dark:text-muted-dark">Reported FPR:</span>
                  <span className="text-accent font-semibold">{(m.fpr * 100).toFixed(2)}%</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted dark:text-muted-dark">Reported UDR:</span>
                  <span className="font-semibold">{(m.udr * 100).toFixed(2)}%</span>
                </div>
              </div>
            </div>
          );
        })}
      </div>

      {/* Slider Controls Container */}
      <div className="border border-rule dark:border-rule-dark p-5 bg-paper dark:bg-panel space-y-4">
        <div className="flex flex-wrap items-center justify-between gap-2 font-mono text-xs">
          <div>
            <span className="text-muted dark:text-muted-dark mr-2">Decision Threshold &tau;:</span>
            <span className="text-accent font-bold text-sm">
              {Number(threshold).toFixed(4)}
            </span>
            {isOriginalThreshold && (
              <span className="ml-2 text-[10px] px-1.5 py-0.5 border border-rule/60 text-muted">
                (Reported Optimum)
              </span>
            )}
          </div>

          <div className="flex items-center gap-2">
            <span className="text-[11px] text-muted hidden sm:inline">Nudge: [ / ]</span>
            {!isOriginalThreshold && (
              <button
                onClick={() => setThreshold(currentMethod.threshold)}
                className="text-[11px] font-mono px-2 py-0.5 border border-rule dark:border-rule-dark hover:border-accent text-muted hover:text-accent transition-colors"
              >
                Reset to {currentMethod.threshold}
              </button>
            )}
          </div>
        </div>

        {/* Range Slider */}
        <div className="relative pt-2 pb-1">
          <input
            type="range"
            min={minBound}
            max={maxBound}
            step={currentMethod.thresholdStep || 0.005}
            value={threshold}
            onChange={(e) => setThreshold(parseFloat(e.target.value))}
            className="w-full accent-accent cursor-pointer bg-rule dark:bg-rule-dark h-1.5 rounded-none"
            aria-label="Rejection Threshold Slider"
          />
          <div className="flex justify-between text-[10px] font-mono text-muted dark:text-muted-dark mt-1">
            <span>Min: {minBound} (More Unknowns Accepted)</span>
            <span className="text-accent font-semibold">Reported: {currentMethod.threshold}</span>
            <span>Max: {maxBound} (More Unknowns Rejected)</span>
          </div>
        </div>

        {/* Operating Metrics Readout */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 pt-3 border-t border-rule/40 dark:border-rule-dark/40 font-mono text-center">
          <div className="p-2.5 bg-paper dark:bg-panel/80 border border-rule/50">
            <div className="text-[10px] uppercase text-muted dark:text-muted-dark">
              True Positive Rate (TPR)
            </div>
            <div className="text-lg font-bold text-teal-sonar mt-0.5">
              {liveStats 
                ? `${(liveStats.tpr * 100).toFixed(2)}%`
                : `${(currentMethod.tpr * 100).toFixed(2)}%*`}
            </div>
            <div className="text-[10px] text-muted">Known Acceptance</div>
          </div>

          <div className="p-2.5 bg-paper dark:bg-panel/80 border border-rule/50">
            <div className="text-[10px] uppercase text-muted dark:text-muted-dark">
              False Positive Rate (FPR)
            </div>
            <div className="text-lg font-bold text-accent mt-0.5">
              {liveStats 
                ? `${(liveStats.fpr * 100).toFixed(2)}%`
                : `${(currentMethod.fpr * 100).toFixed(2)}%*`}
            </div>
            <div className="text-[10px] text-muted">Unknowns Leaked</div>
          </div>

          <div className="p-2.5 bg-paper dark:bg-panel/80 border border-rule/50">
            <div className="text-[10px] uppercase text-muted dark:text-muted-dark">
              Unknown Detection (UDR)
            </div>
            <div className="text-lg font-bold text-ink dark:text-ink-dark mt-0.5">
              {liveStats 
                ? `${(liveStats.udr * 100).toFixed(2)}%`
                : `${(currentMethod.udr * 100).toFixed(2)}%*`}
            </div>
            <div className="text-[10px] text-muted">Unknowns Rejected</div>
          </div>

          <div className="p-2.5 bg-paper dark:bg-panel/80 border border-rule/50">
            <div className="text-[10px] uppercase text-muted dark:text-muted-dark">
              Live Counts
            </div>
            <div className="text-xs font-semibold text-ink dark:text-ink-dark mt-1">
              {liveStats ? (
                <>
                  <div>{liveStats.knownAccepted} / {liveStats.knownTotal} known</div>
                  <div className="text-accent">{liveStats.unknownRejected} / {liveStats.unknownTotal} unk</div>
                </>
              ) : (
                <div className="text-[11px] text-muted font-sans pt-1">
                  Run export_scores.py
                </div>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* SVG Score Distribution Histogram (Live or Diagnostic Distribution Chart) */}
      <div className="border border-rule dark:border-rule-dark bg-paper dark:bg-panel p-4">
        <div className="flex items-center justify-between pb-2 mb-3 border-b border-rule/40 font-mono text-xs">
          <span className="font-semibold text-ink dark:text-ink-dark uppercase tracking-wider">
            Score Distribution Geometry (Known vs Unknown Partitions)
          </span>
          <div className="flex items-center gap-3 text-[11px]">
            <span className="flex items-center gap-1.5">
              <span className="w-2.5 h-2.5 bg-teal-sonar inline-block"></span>
              <span className="text-muted dark:text-muted-dark">Known Targets</span>
            </span>
            <span className="flex items-center gap-1.5">
              <span className="w-2.5 h-2.5 bg-accent inline-block"></span>
              <span className="text-muted dark:text-muted-dark">Unknown / Novel</span>
            </span>
          </div>
        </div>

        {isLive && liveStats ? (
          /* Live Interactive SVG Histogram */
          <div className="w-full overflow-x-auto">
            <svg
              viewBox="0 0 600 200"
              className="w-full h-auto select-none font-mono text-[10px]"
              role="img"
              aria-label="Score distribution histogram"
            >
              {/* Bins */}
              {Array.from({ length: liveStats.numBins }).map((_, i) => {
                const x = 50 + (i / liveStats.numBins) * 500;
                const w = 500 / liveStats.numBins - 1;
                const kH = (liveStats.knownBins[i] / liveStats.maxCount) * 140;
                const uH = (liveStats.unknownBins[i] / liveStats.maxCount) * 140;

                return (
                  <g key={i}>
                    {/* Known bin (teal) */}
                    <rect
                      x={x}
                      y={160 - kH}
                      width={w}
                      height={kH}
                      fill="#3F7F78"
                      opacity="0.75"
                    />
                    {/* Unknown bin (amber) */}
                    <rect
                      x={x}
                      y={160 - uH}
                      width={w}
                      height={uH}
                      fill="#C9822B"
                      opacity="0.65"
                    />
                  </g>
                );
              })}

              {/* Threshold Vertical Line */}
              {(() => {
                const clamped = Math.max(liveStats.minScore, Math.min(liveStats.maxScore, threshold));
                const ratio = (clamped - liveStats.minScore) / (liveStats.maxScore - liveStats.minScore);
                const lineX = 50 + ratio * 500;
                return (
                  <g>
                    <line
                      x1={lineX}
                      y1="10"
                      x2={lineX}
                      y2="160"
                      stroke="#C9822B"
                      strokeWidth="2"
                      strokeDasharray="4,2"
                    />
                    <text
                      x={lineX}
                      y="175"
                      textAnchor="middle"
                      fill="#C9822B"
                      className="font-bold text-[10px]"
                    >
                      &tau; = {threshold.toFixed(3)}
                    </text>
                  </g>
                );
              })()}

              {/* Baseline axis */}
              <line x1="50" y1="160" x2="550" y2="160" stroke="#6B7075" strokeWidth="1" />
              <text x="50" y="188" fill="#6B7075">
                {liveStats.minScore.toFixed(2)} (Less Familiar)
              </text>
              <text x="550" y="188" textAnchor="end" fill="#6B7075">
                {liveStats.maxScore.toFixed(2)} (More Familiar)
              </text>
            </svg>
          </div>
        ) : (
          /* Real Distribution Figure from outputs/figures when JSON not exported */
          <div className="space-y-3">
            <div className="bg-panel p-2 border border-rule/30">
              <img
                src="/figures/msp_distributions.png"
                alt="Empirical score distributions for known vs unknown acoustic returns"
                className="w-full h-auto max-h-72 object-contain mx-auto"
              />
            </div>
            <p className="text-[11px] text-muted dark:text-muted-dark font-sans leading-normal">
              Showing empirical familiarity score density separating known training classes from novel test classes. 
              Executing <code className="font-mono text-accent">python scripts/export_scores.py</code> will output raw test arrays 
              to enable live per-sample histogram sliding.
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
