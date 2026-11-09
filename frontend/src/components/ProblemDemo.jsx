import React, { useState } from 'react';

const DEMO_CASES = {
  chain: {
    name: "Chain (Acoustic link structure)",
    groundTruth: "Unknown / Novel Target",
    closedSetPred: "propeller",
    closedSetConf: 97.4,
    closedSetReason: "Forced classification: high-density acoustic reflectivity matches radial propeller hub features.",
    osrScore: -1.18,
    osrThreshold: -0.712,
    osrDecision: "REJECTED (Unknown)",
    objectPosition: "20% 30%",
  },
  valve: {
    name: "Valve (Subsea valve body)",
    groundTruth: "Unknown / Novel Target",
    closedSetPred: "can",
    closedSetConf: 96.8,
    closedSetReason: "Forced classification: circular acoustic shadow profile matches aluminum can cross-section.",
    osrScore: -1.24,
    osrThreshold: -0.712,
    osrDecision: "REJECTED (Unknown)",
    objectPosition: "50% 30%",
  },
  "shampoo-bottle": {
    name: "Shampoo Bottle (Plastic debris)",
    groundTruth: "Unknown / Novel Target",
    closedSetPred: "bottle",
    closedSetConf: 98.9,
    closedSetReason: "Forced classification: elongated plastic acoustic return confused with glass bottle.",
    osrScore: -0.96,
    osrThreshold: -0.712,
    osrDecision: "REJECTED (Unknown)",
    objectPosition: "80% 30%",
  },
  "standing-bottle": {
    name: "Standing Bottle (Upright target)",
    groundTruth: "Unknown / Novel Target",
    closedSetPred: "bottle",
    closedSetConf: 99.1,
    closedSetReason: "Forced classification: specular highlight matches upright glass bottle signature.",
    osrScore: -1.05,
    osrThreshold: -0.712,
    osrDecision: "REJECTED (Unknown)",
    objectPosition: "50% 80%",
  }
};

export default function ProblemDemo() {
  const [selectedCase, setSelectedCase] = useState("chain");
  const current = DEMO_CASES[selectedCase];

  return (
    <div className="my-8 border border-rule dark:border-rule-dark bg-paper dark:bg-paper-dark p-4 md:p-6">
      {/* Target selector pills */}
      <div className="flex flex-wrap items-center justify-between gap-3 pb-4 border-b border-rule dark:border-rule-dark text-xs">
        <span className="font-mono text-muted dark:text-muted-dark font-medium uppercase tracking-wider">
          Select Novel Test Target:
        </span>
        <div className="flex flex-wrap gap-1 font-mono">
          {Object.keys(DEMO_CASES).map((key) => (
            <button
              key={key}
              onClick={() => setSelectedCase(key)}
              className={`px-2.5 py-1 text-xs border transition-colors ${
                selectedCase === key
                  ? 'border-accent bg-accent/10 text-accent font-semibold'
                  : 'border-rule dark:border-rule-dark text-muted dark:text-muted-dark hover:text-ink dark:hover:text-ink-dark'
              }`}
            >
              {key}
            </button>
          ))}
        </div>
      </div>

      {/* Target Overview Bar */}
      <div className="py-3 flex flex-wrap items-center justify-between gap-2 text-xs font-mono border-b border-rule/50 dark:border-rule-dark/50 mb-6">
        <div>
          <span className="text-muted dark:text-muted-dark mr-2">Acoustic Object:</span>
          <span className="text-ink dark:text-ink-dark font-semibold">{current.name}</span>
        </div>
        <div>
          <span className="text-muted dark:text-muted-dark mr-2">Ground Truth Partition:</span>
          <span className="text-accent font-semibold uppercase">{current.groundTruth}</span>
        </div>
      </div>

      {/* Two Column Side-by-Side Comparison */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Column 1: Closed-Set System */}
        <div className="border border-rule dark:border-rule-dark p-4 bg-paper dark:bg-panel flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between border-b border-rule dark:border-rule-dark pb-2 mb-3 font-mono text-xs">
              <span className="font-semibold text-ink dark:text-ink-dark">Closed-Set Classifier</span>
              <span className="text-red-700 dark:text-red-400 font-bold tracking-wider">[FORCED DECISION]</span>
            </div>

            {/* Sonar Image Crop Container */}
            <div className="relative h-44 bg-panel border border-rule/40 overflow-hidden mb-3">
              <img
                src="/figures/unknown_sample_images.png"
                alt={current.name}
                className="w-full h-full object-cover"
                style={{ objectPosition: current.objectPosition }}
              />
              <div className="absolute bottom-2 left-2 px-1.5 py-0.5 bg-panel/90 text-[10px] font-mono text-muted text-white">
                FLS Sonar Crop (ARIS 3000)
              </div>
            </div>

            {/* Prediction Output */}
            <div className="space-y-2 text-xs font-mono">
              <div className="flex justify-between items-center py-1 border-b border-rule/30 dark:border-rule-dark/30">
                <span className="text-muted dark:text-muted-dark">Forced Prediction:</span>
                <span className="font-semibold text-red-600 dark:text-red-400 uppercase">
                  {current.closedSetPred}
                </span>
              </div>
              <div className="flex justify-between items-center py-1 border-b border-rule/30 dark:border-rule-dark/30">
                <span className="text-muted dark:text-muted-dark">Output Confidence:</span>
                <span className="font-semibold text-red-600 dark:text-red-400">
                  {current.closedSetConf}%*
                </span>
              </div>
              <p className="text-[11px] text-muted dark:text-muted-dark font-sans pt-1 leading-normal">
                {current.closedSetReason}
              </p>
            </div>
          </div>

          <div className="mt-4 pt-3 border-t border-rule dark:border-rule-dark text-[11px] font-mono text-red-700 dark:text-red-400 flex items-center justify-between">
            <span>Result: False Positive</span>
            <span className="text-[10px] text-muted">*Illustrative confidence</span>
          </div>
        </div>

        {/* Column 2: Open-Set Recognition System */}
        <div className="border border-accent/60 dark:border-accent/60 p-4 bg-accent/5 dark:bg-panel flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between border-b border-accent/30 pb-2 mb-3 font-mono text-xs">
              <span className="font-semibold text-accent">Open-Set Classifier (OSR)</span>
              <span className="text-accent font-bold tracking-wider">[REJECTION GATE]</span>
            </div>

            {/* Same Sonar Image Crop Container */}
            <div className="relative h-44 bg-panel border border-accent/40 overflow-hidden mb-3">
              <img
                src="/figures/unknown_sample_images.png"
                alt={current.name}
                className="w-full h-full object-cover"
                style={{ objectPosition: current.objectPosition }}
              />
              <div className="absolute top-2 right-2 px-1.5 py-0.5 bg-accent text-[10px] font-mono text-black font-semibold">
                Boundary Evaluated
              </div>
            </div>

            {/* Decision Logic Output */}
            <div className="space-y-2 text-xs font-mono">
              <div className="flex justify-between items-center py-1 border-b border-rule/30 dark:border-rule-dark/30">
                <span className="text-muted dark:text-muted-dark">Familiarity Score:</span>
                <span className="font-semibold text-accent">
                  {current.osrScore} (below threshold {current.osrThreshold})
                </span>
              </div>
              <div className="flex justify-between items-center py-1 border-b border-rule/30 dark:border-rule-dark/30">
                <span className="text-muted dark:text-muted-dark">System Action:</span>
                <span className="font-semibold text-accent uppercase">
                  {current.osrDecision}
                </span>
              </div>
              <p className="text-[11px] text-muted dark:text-muted-dark font-sans pt-1 leading-normal">
                Score falls into open space risk zone. Instead of forcing a false label, the target is safely sequestered for anomaly handling.
              </p>
            </div>
          </div>

          <div className="mt-4 pt-3 border-t border-accent/30 text-[11px] font-mono text-accent flex items-center justify-between">
            <span>Result: Correct Unknown Rejection</span>
            <span className="text-[10px] text-muted">Risk bounded</span>
          </div>
        </div>
      </div>
      <p className="mt-3 text-[11px] text-muted dark:text-muted-dark font-sans italic">
        *Note: The 96–99% confidence values in the closed-set column illustrate the classic overconfidence problem in softmax networks when novel inputs lie far from known distributions; the open-set rejection threshold is derived from empirical benchmark evaluation.
      </p>
    </div>
  );
}
