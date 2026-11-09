import React, { useState } from 'react';
import Figure from './Figure';

export default function EmbeddingSpace({ onOpenLightbox }) {
  const [activeTab, setActiveTab] = useState('dml');

  const tabs = [
    {
      id: 'dml',
      label: 'DML 128-D Hypersphere t-SNE',
      figure: {
        file: '/figures/dml_tsne_visualization.png',
        number: 'Fig. 9A',
        title: 'Deep Metric Learning 128-D Embedding Space (t-SNE)',
        caption: 'Two-dimensional t-SNE projection of the 128-dimensional L2 unit-norm embedding space trained with Triplet Margin Loss and Multi-Similarity mining.'
      },
      separationNotes: 'What separation to look for: Observe how the six known classes contract into dense, well-isolated clusters with significant inter-class angular margins. In contrast, the novel unknown instances (withheld during training) do not collapse into known clusters; rather, they disperse across the ambient embedding space or peripheral voids, allowing simple Euclidean thresholding to reject them.'
    },
    {
      id: 'metric',
      label: 'Metric Space vs Unknowns',
      figure: {
        file: '/figures/metric_tsne.png',
        number: 'Fig. 9B',
        title: 'Metric Space Distribution Highlighting Novel Objects',
        caption: 'Detailed t-SNE manifold mapping known class centroids against unknown acoustic scatter.'
      },
      separationNotes: 'What separation to look for: Note the boundary tightness around known targets. Unknown acoustic artifacts that produce low-density echoes or diffuse specular returns occupy the outer regions. While occasional hard unknown instances fall near peripheral clusters (explaining the 25% FPR), the majority are separated by wide geometric margins.'
    }
  ];

  const current = tabs.find((t) => t.id === activeTab) || tabs[0];

  return (
    <div className="space-y-6">
      <p className="text-sm text-ink/90 dark:text-ink-dark/90 leading-relaxed font-sans">
        To explain why Deep Metric Learning achieves an AUROC of 0.9255, we projected the high-dimensional representations 
        into 2D space using t-Distributed Stochastic Neighbor Embedding (t-SNE). The visual geometry confirms that metric loss 
        imposes strict intra-class compactness and forces novel acoustic objects into the exterior open-space region.
      </p>

      {/* Tab bar */}
      <div className="border border-rule dark:border-rule-dark bg-paper dark:bg-panel">
        <div className="flex border-b border-rule dark:border-rule-dark bg-paper-dark text-paper text-xs font-mono">
          {tabs.map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`px-4 py-2.5 transition-colors text-left border-r border-rule/30 ${
                activeTab === tab.id
                  ? 'bg-paper dark:bg-panel text-accent font-semibold border-b-2 border-b-accent'
                  : 'text-muted hover:text-paper'
              }`}
            >
              {tab.label}
            </button>
          ))}
        </div>

        <div className="p-4">
          <Figure
            figure={current.figure}
            onOpenLightbox={onOpenLightbox}
          />

          <div className="mt-3 p-3 bg-paper-dark/5 dark:bg-paper-dark border-l-2 border-accent text-xs font-sans text-ink/90 dark:text-ink-dark/90 leading-relaxed">
            <strong className="text-accent font-mono text-[11px] block uppercase mb-1">
              Geometric Separation Analysis:
            </strong>
            {current.separationNotes}
          </div>
        </div>
      </div>
    </div>
  );
}
