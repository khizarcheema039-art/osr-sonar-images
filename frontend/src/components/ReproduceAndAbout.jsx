import React from 'react';
import { REFERENCES, PROJECT_METADATA } from '../data/results';

export default function ReproduceAndAbout() {
  const folderTree = `OSR of SONAR images/
├── configs/
│   └── config.py                 # Dataset paths, known/unknown partitions, hyperparameters
├── src/
│   ├── dataset.py                # MD-FLS loaders, CLAHE, median filter, train/val/test splits
│   ├── models/                   # ResNet-50 backbone, 128-D projection head EmbeddingNet
│   ├── methods/
│   │   ├── msp.py                # Maximum Softmax Probability thresholding
│   │   ├── svdd.py               # Deep SVDD nearest centroid Euclidean distance
│   │   ├── dml.py                # Deep Metric Learning multi-similarity miner
│   │   └── openmax.py            # OpenMax EVT Weibull calibration (implemented)
│   └── evaluation/
│       ├── metrics.py            # AUROC, AP, TPR, FPR, ECE calculation
│       └── visualizations.py     # Master benchmark dashboard and curve generators
├── scripts/
│   ├── train_baseline.py         # Trains closed-set ResNet-50 (25 epochs, SGD)
│   ├── train_metric.py           # Trains 128-D metric embedding head (triplet loss)
│   ├── evaluate_all.py           # Runs complete 3-method benchmark and plots figures
│   └── export_scores.py          # Exports raw per-sample scores to frontend/public/data/scores.json
├── outputs/
│   ├── models/                   # Checkpoints: resnet50_baseline.pth, metric_embedding_model.pth
│   ├── results/                  # all_results.json, benchmark_summary.csv
│   └── figures/                  # 22 publication-grade diagnostic plots
└── frontend/                     # Interactive research companion website`;

  return (
    <div className="space-y-8">
      {/* Reproduction Commands */}
      <div>
        <h4 className="font-serif text-base font-semibold text-ink dark:text-ink-dark mb-2">
          Execution Commands
        </h4>
        <p className="text-xs text-muted dark:text-muted-dark font-sans mb-3">
          All models and benchmark tables can be reproduced deterministically from the repository root:
        </p>

        <div className="bg-panel text-paper p-4 font-mono text-xs border border-rule/30 space-y-2 overflow-x-auto">
          <div className="text-muted text-[11px]"># 1. Install PyTorch, torchvision, and metric-learning libraries</div>
          <div className="text-accent">pip install -r requirements.txt</div>
          
          <div className="text-muted text-[11px] pt-2"># 2. Train baseline closed-set ResNet-50 classifier</div>
          <div className="text-accent">python scripts/train_baseline.py</div>

          <div className="text-muted text-[11px] pt-2"># 3. Train deep metric learning 128-D embedding space</div>
          <div className="text-accent">python scripts/train_metric.py</div>

          <div className="text-muted text-[11px] pt-2"># 4. Run full comparative open-set benchmark & generate figures</div>
          <div className="text-accent">python scripts/evaluate_all.py</div>

          <div className="text-muted text-[11px] pt-2"># 5. Export per-sample scores to enable the frontend live slider</div>
          <div className="text-accent">python scripts/export_scores.py</div>
        </div>
      </div>

      {/* Directory Structure */}
      <div>
        <h4 className="font-serif text-base font-semibold text-ink dark:text-ink-dark mb-2">
          Repository Structure
        </h4>
        <div className="bg-panel text-paper/90 p-4 font-mono text-xs border border-rule/30 overflow-x-auto whitespace-pre leading-relaxed">
          {folderTree}
        </div>
      </div>

      {/* Tools & Frameworks */}
      <div>
        <h4 className="font-serif text-base font-semibold text-ink dark:text-ink-dark mb-2">
          Tools & Hardware Specifications
        </h4>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 font-mono text-xs">
          <div className="p-3 border border-rule dark:border-rule-dark bg-paper dark:bg-panel">
            <span className="text-muted dark:text-muted-dark block text-[10px] uppercase">Deep Learning Framework</span>
            <span className="text-ink dark:text-ink-dark font-medium">PyTorch 2.x • torchvision • CUDA</span>
          </div>
          <div className="p-3 border border-rule dark:border-rule-dark bg-paper dark:bg-panel">
            <span className="text-muted dark:text-muted-dark block text-[10px] uppercase">Metric Learning Head</span>
            <span className="text-ink dark:text-ink-dark font-medium">pytorch-metric-learning (MS Miner)</span>
          </div>
          <div className="p-3 border border-rule dark:border-rule-dark bg-paper dark:bg-panel">
            <span className="text-muted dark:text-muted-dark block text-[10px] uppercase">Computer Vision / Signal</span>
            <span className="text-ink dark:text-ink-dark font-medium">OpenCV (CLAHE) • scikit-learn</span>
          </div>
          <div className="p-3 border border-rule dark:border-rule-dark bg-paper dark:bg-panel">
            <span className="text-muted dark:text-muted-dark block text-[10px] uppercase">Acoustic Sensor Hardware</span>
            <span className="text-ink dark:text-ink-dark font-medium">Sound Metrics ARIS Explorer 3000 FLS</span>
          </div>
        </div>
      </div>

      {/* References Cited */}
      <div>
        <h4 className="font-serif text-base font-semibold text-ink dark:text-ink-dark mb-2">
          References
        </h4>
        <ol className="divide-y divide-rule/50 dark:divide-rule-dark/50 border-t border-b border-rule dark:border-rule-dark">
          {REFERENCES.map((ref, idx) => (
            <li key={idx} className="py-3 text-xs font-sans text-ink/90 dark:text-ink-dark/90 leading-relaxed">
              <div className="font-mono text-accent font-semibold mb-0.5">
                [{idx + 1}] {ref.cite}
              </div>
              <div>
                "{ref.title}." <span className="italic">{ref.venue}</span>.
              </div>
              <div className="text-muted dark:text-muted-dark text-[11px] mt-0.5">
                {ref.note}
              </div>
            </li>
          ))}
        </ol>
      </div>

      {/* Academic Credits */}
      <div className="p-4 border border-rule dark:border-rule-dark bg-paper dark:bg-panel font-mono text-xs space-y-1">
        <div className="font-semibold text-accent uppercase text-[11px]">Final Year Project Attribution</div>
        <div className="text-ink dark:text-ink-dark">
          Department of Computer Science, Bahria University, Islamabad
        </div>
        <div className="text-muted dark:text-muted-dark">
          Authors: Mohammad Khizar Hayat Cheema (01-134231-039), Muhammad Haroon (01-134231-052)
        </div>
        <div className="text-muted dark:text-muted-dark">
          Supervisor: Dr. Arif Ur Rahman
        </div>
      </div>
    </div>
  );
}
