import React, { useState } from 'react'
import './DebugPanel.css'

function DebugPanel({ debugInfo, loading }) {
  const [expandedSections, setExpandedSections] = useState({
    analysis: true,
    timing: true,
    weights: false,
    raw: false
  })

  if (loading) {
    return (
      <div className="debug-panel">
        <div className="debug-loading">Loading debug info...</div>
      </div>
    )
  }

  if (!debugInfo) {
    return (
      <div className="debug-panel">
        <div className="debug-empty">No debug information available</div>
      </div>
    )
  }

  const toggleSection = (section) => {
    setExpandedSections({
      ...expandedSections,
      [section]: !expandedSections[section]
    })
  }

  const copyToClipboard = (text) => {
    navigator.clipboard.writeText(text)
  }

  return (
    <div className="debug-panel">
      <div className="debug-header">
        <h3>Debug Information</h3>
      </div>

      {/* Query Analysis */}
      <div className="debug-section">
        <button
          className="section-toggle"
          onClick={() => toggleSection('analysis')}
        >
          {expandedSections.analysis ? '▼' : '▶'} Query Analysis
        </button>
        {expandedSections.analysis && (
          <div className="section-content">
            <div className="info-row">
              <span className="info-key">Original Query:</span>
              <code className="info-value">{debugInfo.query}</code>
            </div>
            <div className="info-row">
              <span className="info-key">Object Query:</span>
              <code className="info-value">{debugInfo.object_query}</code>
            </div>
            <div className="info-row">
              <span className="info-key">Shortlist Size:</span>
              <span className="info-value">{debugInfo.shortlist_size} products</span>
            </div>
            <div className="info-row">
              <span className="info-key">Candidates:</span>
              <span className="info-value">{debugInfo.candidate_count} products</span>
            </div>

            {debugInfo.alias_hits && debugInfo.alias_hits.length > 0 && (
              <div className="alias-hits">
                <span className="info-key">Detected Aliases:</span>
                <div className="alias-list">
                  {debugInfo.alias_hits.map((hit, idx) => (
                    <div key={idx} className="alias-item">
                      <code className="alias-text">{hit.alias}</code>
                      <span className="alias-arrow">→</span>
                      <span className="alias-field">{hit.field}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {debugInfo.constraints && debugInfo.constraints.length > 0 && (
              <div className="constraints-info">
                <span className="info-key">Constraints:</span>
                <div className="constraints-list">
                  {debugInfo.constraints.map((constraint, idx) => (
                    <div key={idx} className="constraint-info-item">
                      <div className="constraint-info-header">
                        <code className="constraint-field">{constraint.field}</code>
                        {constraint.matched ? (
                          <span className="constraint-status success">✓ Matched</span>
                        ) : (
                          <span className="constraint-status failed">✗ Failed</span>
                        )}
                      </div>
                      <div className="constraint-details">
                        <span>Value: {constraint.value}</span>
                        {constraint.tolerance && <span>Tolerance: ±{constraint.tolerance}</span>}
                        {constraint.count !== undefined && (
                          <span>Products: {constraint.count}</span>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Timing Breakdown */}
      <div className="debug-section">
        <button
          className="section-toggle"
          onClick={() => toggleSection('timing')}
        >
          {expandedSections.timing ? '▼' : '▶'} Timing Breakdown
        </button>
        {expandedSections.timing && debugInfo.timings && (
          <div className="section-content">
            <div className="timing-table">
              {Object.entries(debugInfo.timings).map(([key, value]) => (
                <div key={key} className="timing-row">
                  <span className="timing-label">{key.replace(/_/g, ' ')}</span>
                  <div className="timing-bar-container">
                    <div
                      className="timing-bar"
                      style={{
                        width: `${(value / debugInfo.timings.total) * 100}%`
                      }}
                    />
                    <span className="timing-value">
                      {(value * 1000).toFixed(2)}ms
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Weights */}
      <div className="debug-section">
        <button
          className="section-toggle"
          onClick={() => toggleSection('weights')}
        >
          {expandedSections.weights ? '▼' : '▶'} Active Weights
        </button>
        {expandedSections.weights && debugInfo.weights && (
          <div className="section-content">
            <div className="weights-grid">
              {Object.entries(debugInfo.weights).map(([key, value]) => (
                <div key={key} className="weight-item">
                  <span className="weight-label">{key}</span>
                  <code className="weight-value">{value.toFixed(3)}</code>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Raw JSON */}
      <div className="debug-section">
        <button
          className="section-toggle"
          onClick={() => toggleSection('raw')}
        >
          {expandedSections.raw ? '▼' : '▶'} Raw JSON
        </button>
        {expandedSections.raw && (
          <div className="section-content">
            <div className="raw-json-header">
              <button
                className="copy-button"
                onClick={() => copyToClipboard(JSON.stringify(debugInfo, null, 2))}
              >
                Copy JSON
              </button>
            </div>
            <pre className="raw-json">
              {JSON.stringify(debugInfo, null, 2)}
            </pre>
          </div>
        )}
      </div>
    </div>
  )
}

export default DebugPanel
