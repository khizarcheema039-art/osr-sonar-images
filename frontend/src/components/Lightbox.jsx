import React, { useEffect } from 'react';

export default function Lightbox({ isOpen, onClose, figure, onNext, onPrev, currentIndex, totalCount }) {
  useEffect(() => {
    if (!isOpen) return;
    const handleKeyDown = (e) => {
      if (e.key === 'Escape') onClose();
      if (e.key === 'ArrowRight' && onNext) onNext();
      if (e.key === 'ArrowLeft' && onPrev) onPrev();
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isOpen, onClose, onNext, onPrev]);

  if (!isOpen || !figure) return null;

  return (
    <div 
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/85 p-4 md:p-8 backdrop-blur-none"
      onClick={onClose}
      role="dialog"
      aria-modal="true"
      aria-label={figure.title}
    >
      <div 
        className="relative max-w-5xl w-full max-h-[92vh] flex flex-col bg-panel border border-rule/40 text-paper overflow-hidden"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header bar */}
        <div className="flex items-center justify-between px-4 py-3 border-b border-rule/20 bg-panel text-xs font-mono">
          <div className="flex items-center gap-3">
            <span className="text-accent font-semibold">{figure.number}</span>
            <span className="text-paper/90 truncate max-w-md">{figure.title}</span>
          </div>
          <div className="flex items-center gap-4">
            <span className="text-muted">
              {currentIndex + 1} / {totalCount}
            </span>
            <button
              onClick={onClose}
              className="text-muted hover:text-paper font-mono px-2 py-0.5 border border-rule/40 hover:border-accent text-xs transition-colors"
              aria-label="Close Lightbox (Esc)"
            >
              [esc] close
            </button>
          </div>
        </div>

        {/* Image Display */}
        <div className="relative flex-1 flex items-center justify-center p-3 bg-black/60 overflow-hidden min-h-[300px]">
          <img
            src={figure.file}
            alt={figure.caption || figure.title}
            className="max-h-[68vh] max-w-full object-contain"
          />

          {/* Prev/Next buttons */}
          <button
            onClick={onPrev}
            className="absolute left-3 top-1/2 -translate-y-1/2 p-2 bg-panel/90 border border-rule/40 text-paper hover:text-accent font-mono text-sm transition-colors"
            aria-label="Previous Figure"
          >
            &larr;
          </button>
          <button
            onClick={onNext}
            className="absolute right-3 top-1/2 -translate-y-1/2 p-2 bg-panel/90 border border-rule/40 text-paper hover:text-accent font-mono text-sm transition-colors"
            aria-label="Next Figure"
          >
            &rarr;
          </button>
        </div>

        {/* Caption */}
        <div className="p-4 bg-panel border-t border-rule/20 text-xs text-muted-dark font-sans leading-relaxed">
          <p className="text-paper/85">
            <strong className="text-accent font-mono mr-2">{figure.number}:</strong>
            {figure.caption}
          </p>
        </div>
      </div>
    </div>
  );
}
