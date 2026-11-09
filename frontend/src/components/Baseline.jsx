import React from 'react';
import Figure from './Figure';
import { BASELINE_METRICS } from '../data/results';

export default function Baseline({ onOpenLightbox }) {
  return (
    <div className="space-y-6">
      <p className="text-sm text-ink/90 dark:text-ink-dark/90 leading-relaxed font-sans">
        Before evaluating open-set rejection boundaries, we established an empirical closed-set baseline. 
        We fine-tuned a standard ResNet-50 architecture across the six known classes using categorical cross-entropy 
        loss and SGD with momentum for 25 epochs. Under closed-world assumptions where the test set contains zero unmodeled classes, 
        the classifier achieves near-perfect discrimination.
      </p>

      {/* 3 Metrics Callout Boxes */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 my-6 font-mono text-center">
        <div className="p-3 border border-rule dark:border-rule-dark bg-paper dark:bg-panel">
          <div className="text-[11px] text-muted dark:text-muted-dark uppercase tracking-wider mb-1">
            Validation Accuracy
          </div>
          <div className="text-2xl font-bold text-ink dark:text-ink-dark">
            {(BASELINE_METRICS.validation_accuracy * 100).toFixed(2)}%
          </div>
          <div className="text-[10px] text-muted dark:text-muted-dark mt-1">
            0.9962 on known val split
          </div>
        </div>

        <div className="p-3 border border-rule dark:border-rule-dark bg-paper dark:bg-panel">
          <div className="text-[11px] text-muted dark:text-muted-dark uppercase tracking-wider mb-1">
            Test Known Accuracy
          </div>
          <div className="text-2xl font-bold text-teal-sonar">
            {(BASELINE_METRICS.test_known_accuracy * 100).toFixed(2)}%
          </div>
          <div className="text-[10px] text-muted dark:text-muted-dark mt-1">
            0.9924 on known test set
          </div>
        </div>

        <div className="p-3 border border-rule dark:border-rule-dark bg-paper dark:bg-panel">
          <div className="text-[11px] text-muted dark:text-muted-dark uppercase tracking-wider mb-1">
            Macro F1-Score
          </div>
          <div className="text-2xl font-bold text-accent">
            {BASELINE_METRICS.f1_score.toFixed(3)}
          </div>
          <div className="text-[10px] text-muted dark:text-muted-dark mt-1">
            balanced across all 6 classes
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <Figure
          figure={{
            file: "/figures/confusion_matrix_baseline.png",
            number: "Fig. 5A",
            title: "Closed-Set Baseline Confusion Matrix",
            caption: "ResNet-50 test set confusion matrix on the 6 known classes. Almost all diagonal entries exceed 0.99."
          }}
          onOpenLightbox={onOpenLightbox}
        />

        <Figure
          figure={{
            file: "/figures/resnet50_baseline_curves.png",
            number: "Fig. 5B",
            title: "Training & Validation Convergence Curves",
            caption: "Cross-entropy loss and top-1 accuracy across 25 epochs. Learning rate: 1e-4 with step decay."
          }}
          onOpenLightbox={onOpenLightbox}
        />
      </div>

      <p className="text-xs text-muted dark:text-muted-dark font-sans leading-relaxed border-l-2 border-rule dark:border-rule-dark pl-3 italic">
        Caveat: This exceptional closed-set performance (99.24% test accuracy) is deceptive in real marine operations. 
        Because the softmax function normalizes exponents to sum to 1.0, the model is architecturally incapable of outputting 
        an "I do not know" state. When exposed to novel seabed debris, it assigns confident labels from the known set.
      </p>
    </div>
  );
}
