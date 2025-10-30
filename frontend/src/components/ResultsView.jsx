import React, { useState } from 'react'
import ScoreBreakdown from './ScoreBreakdown'
import './ResultsView.css'

function ResultsView({ results, loading }) {
  const [expandedId, setExpandedId] = useState(null)

  if (loading) {
    return (
      <div className="results-view">
        <div className="loading">Searching...</div>
      </div>
    )
  }

  if (!results || results.length === 0) {
    return (
      <div className="results-view">
        <div className="no-results">No results found</div>
      </div>
    )
  }

  const toggleExpand = (rank) => {
    setExpandedId(expandedId === rank ? null : rank)
  }

  return (
    <div className="results-view">
      <div className="results-header">
        <h2>Results ({results.length})</h2>
      </div>

      <div className="results-list">
        {results.map((result) => {
          const isExpanded = expandedId === result.rank

          return (
            <div key={result.rank} className="result-card">
              <div className="result-header" onClick={() => toggleExpand(result.rank)}>
                <div className="result-rank">#{result.rank}</div>
                <div className="result-main">
                  <div className="result-title-row">
                    <span className="result-sku">{result.product.sku}</span>
                    <span className="result-score">
                      Score: {result.score.toFixed(4)}
                    </span>
                  </div>
                  <h3 className="result-name">{result.product.name}</h3>
                  {result.product.brand && (
                    <div className="result-brand">{result.product.brand}</div>
                  )}
                </div>
                <button className="expand-button">
                  {isExpanded ? '▼' : '▶'}
                </button>
              </div>

              {!isExpanded && (
                <div className="result-description">
                  {result.product.description}
                </div>
              )}

              {isExpanded && (
                <div className="result-expanded">
                  <div className="result-details">
                    <div className="detail-section">
                      <h4>Description</h4>
                      <p>{result.product.description}</p>
                    </div>

                    {result.product.specs && (
                      <div className="detail-section">
                        <h4>Specifications</h4>
                        <pre className="specs-text">{result.product.specs}</pre>
                      </div>
                    )}

                    <div className="detail-section">
                      <h4>Product Info</h4>
                      <div className="info-grid">
                        <div className="info-item">
                          <span className="info-label">SKU:</span>
                          <span className="info-value">{result.product.sku}</span>
                        </div>
                        {result.product.mpn && (
                          <div className="info-item">
                            <span className="info-label">MPN:</span>
                            <span className="info-value">{result.product.mpn}</span>
                          </div>
                        )}
                        {result.product.brand && (
                          <div className="info-item">
                            <span className="info-label">Brand:</span>
                            <span className="info-value">{result.product.brand}</span>
                          </div>
                        )}
                        {result.product.url && (
                          <div className="info-item">
                            <span className="info-label">URL:</span>
                            <a
                              href={result.product.url}
                              target="_blank"
                              rel="noopener noreferrer"
                              className="info-link"
                            >
                              View Product →
                            </a>
                          </div>
                        )}
                      </div>
                    </div>
                  </div>

                  <div className="result-scoring">
                    <h4>Score Breakdown</h4>
                    <ScoreBreakdown result={result} />
                  </div>
                </div>
              )}
            </div>
          )
        })}
      </div>
    </div>
  )
}

export default ResultsView
