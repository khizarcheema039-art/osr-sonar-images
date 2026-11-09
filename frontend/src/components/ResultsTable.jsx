import React from 'react';
import Figure from './Figure';
import PerClassChart from './PerClassChart';
import { OSR_METHODS } from '../data/results';

export default function ResultsTable({ 
  selectedMethod, 
  onSelectMethod, 
  hoveredMethod, 
  onHoverMethod, 
  onOpenLightbox 
}) {
  const methods = Object.values(OSR_METHODS);

  return (
    <div className="space-y-6">
      {/* Method Text Toggles (Keyboard 1/2/3) */}
      <div className="flex flex-wrap items-center justify-between gap-3 pb-3 border-b border-rule dark:border-rule-dark">
        <span className="font-mono text-xs font-medium uppercase tracking-wider text-muted dark:text-muted-dark">
          Highlight Paradigm:
        </span>
        <div className="flex flex-wrap gap-1 font-mono text-xs">
          {methods.map((m) => {
            const isSelected = selectedMethod === m.key;
            return (
              <button
                key={m.key}
                onClick={() => onSelectMethod(m.key)}
                className={`px-3 py-1 border transition-colors flex items-center gap-1.5 ${
                  isSelected
                    ? 'border-accent bg-accent/10 text-accent font-semibold'
                    : 'border-rule dark:border-rule-dark text-muted dark:text-muted-dark hover:text-ink dark:hover:text-ink-dark'
                }`}
              >
                <span>[{m.index}] {m.shortLabel}</span>
              </button>
            );
          })}
        </div>
      </div>

      {/* Real HTML Results Table */}
      <div className="border border-rule dark:border-rule-dark overflow-x-auto">
        <table className="w-full text-left font-mono text-xs">
          <thead className="bg-paper-dark text-paper border-b border-rule/30">
            <tr>
              <th scope="col" className="p-3 font-semibold">Open-Set Method</th>
              <th scope="col" className="p-3 font-semibold text-right" title="Area Under Receiver Operating Characteristic (threshold-free)">
                AUROC &uarr;
              </th>
              <th scope="col" className="p-3 font-semibold text-right" title="Average Precision (Area under PR curve)">
                AP &uarr;
              </th>
              <th scope="col" className="p-3 font-semibold text-right" title="True Positive Rate (Known acceptance rate)">
                TPR &uarr;
              </th>
              <th scope="col" className="p-3 font-semibold text-right" title="False Positive Rate (Unknowns falsely accepted as known)">
                FPR &darr;
              </th>
              <th scope="col" className="p-3 font-semibold text-right" title="Unknown Detection Rate (1 - FPR)">
                UDR &uarr;
              </th>
              <th scope="col" className="p-3 font-semibold text-right" title="Empirical Operating Threshold">
                Threshold &tau;
              </th>
            </tr>
          </thead>
          <tbody className="divide-y divide-rule/40 dark:divide-rule-dark/40 bg-paper dark:bg-panel text-ink dark:text-ink-dark">
            {methods.map((m) => {
              const isSelected = selectedMethod === m.key;
              const isHovered = hoveredMethod === m.key;
              const isHighlighted = isSelected || isHovered;

              return (
                <tr
                  key={m.key}
                  onClick={() => onSelectMethod(m.key)}
                  onMouseEnter={() => onHoverMethod && onHoverMethod(m.key)}
                  onMouseLeave={() => onHoverMethod && onHoverMethod(null)}
                  className={`cursor-pointer transition-colors duration-150 ${
                    isHighlighted 
                      ? 'bg-accent/10 dark:bg-accent/15 font-medium' 
                      : 'hover:bg-rule/10 dark:hover:bg-rule-dark/20'
                  }`}
                >
                  <th scope="row" className="p-3 font-sans font-medium">
                    <div className="flex items-center gap-2">
                      <span className={`w-2 h-2 rounded-full ${isSelected ? 'bg-accent' : 'bg-muted/40'}`}></span>
                      <span className={isSelected ? 'text-accent font-semibold' : ''}>
                        {m.label}
                      </span>
                    </div>
                  </th>

                  {/* AUROC - Best is MetricLearning 0.9255 */}
                  <td className="p-3 text-right">
                    {m.key === 'MetricLearning' ? (
                      <span className="font-bold underline decoration-accent decoration-1 underline-offset-2">
                        {m.auroc.toFixed(4)}
                      </span>
                    ) : (
                      m.auroc.toFixed(4)
                    )}
                  </td>

                  {/* AP - Best is MetricLearning 0.8448 */}
                  <td className="p-3 text-right">
                    {m.key === 'MetricLearning' ? (
                      <span className="font-bold underline decoration-accent decoration-1 underline-offset-2">
                        {m.ap.toFixed(4)}
                      </span>
                    ) : (
                      m.ap.toFixed(4)
                    )}
                  </td>

                  {/* TPR - Best is MetricLearning 0.9410 */}
                  <td className="p-3 text-right">
                    {m.key === 'MetricLearning' ? (
                      <span className="font-bold underline decoration-accent decoration-1 underline-offset-2">
                        {m.tpr.toFixed(4)}
                      </span>
                    ) : (
                      m.tpr.toFixed(4)
                    )}
                  </td>

                  {/* FPR - Best (lowest) is SVDD 0.1171 */}
                  <td className="p-3 text-right">
                    {m.key === 'SVDD' ? (
                      <span className="font-bold underline decoration-accent decoration-1 underline-offset-2 text-teal-sonar">
                        {m.fpr.toFixed(4)}
                      </span>
                    ) : (
                      m.fpr.toFixed(4)
                    )}
                  </td>

                  {/* UDR - Best is SVDD 0.8829 */}
                  <td className="p-3 text-right">
                    {m.key === 'SVDD' ? (
                      <span className="font-bold underline decoration-accent decoration-1 underline-offset-2 text-teal-sonar">
                        {m.udr.toFixed(4)}
                      </span>
                    ) : (
                      m.udr.toFixed(4)
                    )}
                  </td>

                  {/* Threshold tau */}
                  <td className="p-3 text-right text-muted dark:text-muted-dark">
                    {m.threshold > 0 ? `+${m.threshold.toFixed(4)}` : m.threshold.toFixed(4)}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <div className="text-[11px] font-mono text-muted dark:text-muted-dark flex items-center justify-between">
        <span>Bold / underlined numbers indicate the superior empirical value per column.</span>
        <span>Keyboard shortcuts: 1, 2, 3</span>
      </div>

      {/* Per-Class Grouped Bar Chart */}
      <PerClassChart
        selectedMethod={selectedMethod}
        onSelectMethod={onSelectMethod}
        hoveredMethod={hoveredMethod}
        onHoverMethod={onHoverMethod}
      />

      {/* Method-Specific Dashboard Figure */}
      <div className="pt-2">
        <Figure
          figure={{
            file: `/figures/${OSR_METHODS[selectedMethod].dashboardFig}`,
            number: `Fig. 6 (${OSR_METHODS[selectedMethod].shortLabel})`,
            title: `${OSR_METHODS[selectedMethod].label} Comprehensive Diagnostic Dashboard`,
            caption: OSR_METHODS[selectedMethod].dashboardCaption
          }}
          onOpenLightbox={onOpenLightbox}
        />
      </div>

      {/* Short Honest Reading of the Results */}
      <div className="border border-rule dark:border-rule-dark p-4 bg-paper dark:bg-panel space-y-3 text-xs leading-relaxed font-sans">
        <h4 className="font-serif text-sm font-semibold text-ink dark:text-ink-dark">
          Analysis: Critical Synthesis Across Benchmarked Paradigms
        </h4>
        
        <p className="text-ink/90 dark:text-ink-dark/90">
          The benchmark confirms that there is <strong>no single universal winner</strong> across all operational metrics:
        </p>

        <ul className="list-disc pl-5 space-y-1.5 text-ink/85 dark:text-ink-dark/85 font-sans">
          <li>
            <strong>Deep Metric Learning (triplet loss)</strong> establishes the state of the art in overall discriminatory separation, 
            achieving the highest <strong>AUROC (0.9255)</strong>, highest <strong>Average Precision (0.8448)</strong>, and retaining 
            the highest known acceptance <strong>(TPR = 0.9410)</strong>. Its multi-similarity loss pulls known classes into tight geometric clusters.
          </li>
          <li>
            <strong>Deep SVDD (nearest centroid)</strong> achieves the superior safety boundary against false acceptances, demonstrating the 
            lowest <strong>FPR (0.1171)</strong> and the highest overall <strong>Unknown Detection Rate (UDR = 0.8829)</strong>. It is remarkably robust 
            on novel bottle categories (99% on shampoo bottles, 100% on standing bottles).
          </li>
          <li>
            <strong>Maximum Softmax Probability (MSP)</strong> is the weakest paradigm across all metrics (AUROC 0.8165, AP 0.5782, FPR 0.3177). 
            Softmax normalization without distance grounding allows 31.77% of unknown acoustic instances to pass through undetected.
          </li>
        </ul>

        {/* Operating Threshold Caveat */}
        <div className="p-3 bg-paper-dark/5 dark:bg-paper-dark border-l-2 border-accent text-xs font-mono text-muted dark:text-muted-dark mt-3">
          <strong className="text-accent">Critical Threshold Caveat:</strong> The three methods are evaluated at their individual Youden-optimal 
          operating thresholds (&tau; = 0.9887 for MSP, &tau; = -0.684 for SVDD, &tau; = -0.712 for Metric Learning). Therefore, raw TPR and FPR numbers 
          reflect specific chosen trade-offs and are not directly comparable on an absolute scale, whereas <strong>AUROC and AP provide strict, 
          threshold-free ranking</strong> across the complete continuum of decision boundaries.
        </div>
      </div>
    </div>
  );
}
