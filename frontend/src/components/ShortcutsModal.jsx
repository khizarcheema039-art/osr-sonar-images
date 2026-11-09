import React, { useEffect } from 'react';

export default function ShortcutsModal({ isOpen, onClose }) {
  useEffect(() => {
    if (!isOpen) return;
    const handleKeyDown = (e) => {
      if (e.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isOpen, onClose]);

  if (!isOpen) return null;

  const shortcuts = [
    { key: '1', action: 'Select Maximum Softmax Probability (MSP)' },
    { key: '2', action: 'Select Deep SVDD (nearest centroid)' },
    { key: '3', action: 'Select Deep Metric Learning (triplet loss)' },
    { key: '[', action: 'Nudge threshold slider down (more conservative rejection)' },
    { key: ']', action: 'Nudge threshold slider up (stricter rejection boundary)' },
    { key: 'Esc', action: 'Close active Lightbox or modal dialog' },
    { key: '?', action: 'Toggle this keyboard shortcut helper' },
  ];

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4"
      onClick={onClose}
      role="dialog"
      aria-modal="true"
      aria-label="Keyboard Shortcuts"
    >
      <div
        className="w-full max-w-md bg-paper dark:bg-panel border border-rule dark:border-rule-dark p-6 shadow-sm"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center justify-between pb-3 border-b border-rule dark:border-rule-dark font-mono text-xs">
          <span className="font-semibold text-accent uppercase tracking-wider">
            Keyboard Shortcuts
          </span>
          <button
            onClick={onClose}
            className="text-muted hover:text-ink dark:hover:text-ink-dark"
          >
            [esc]
          </button>
        </div>

        <ul className="divide-y divide-rule/40 dark:divide-rule-dark/40 my-4 font-mono text-xs">
          {shortcuts.map((s) => (
            <li key={s.key} className="py-2 flex items-center justify-between">
              <span className="text-muted dark:text-muted-dark font-sans text-xs">
                {s.action}
              </span>
              <kbd className="px-2 py-0.5 border border-rule dark:border-rule-dark bg-paper dark:bg-panel text-accent font-mono text-[11px] font-semibold">
                {s.key}
              </kbd>
            </li>
          ))}
        </ul>

        <div className="pt-2 border-t border-rule dark:border-rule-dark text-[11px] font-sans text-muted dark:text-muted-dark flex justify-between items-center">
          <span>Final Year Project • OSR Companion</span>
          <button
            onClick={onClose}
            className="text-xs font-mono underline hover:text-accent"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
}
