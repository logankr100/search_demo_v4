import React, { useState } from 'react'
import './SearchPanel.css'

function SearchPanel({ onSearch, loading, metadata }) {
  const [query, setQuery] = useState('')
  const [showAdvanced, setShowAdvanced] = useState(false)

  // Weight parameters
  const [wTitle, setWTitle] = useState(0.35)
  const [wDesc, setWDesc] = useState(0.50)
  const [wSpecs, setWSpecs] = useState(0.15)
  const [wFull, setWFull] = useState(0.95)
  const [wObj, setWObj] = useState(0.05)
  const [wNum, setWNum] = useState(0.25)
  const [k, setK] = useState(20)

  const handleSubmit = (e) => {
    e.preventDefault()
    if (!query.trim()) return

    onSearch({
      query: query.trim(),
      w_title: wTitle,
      w_desc: wDesc,
      w_specs: wSpecs,
      w_full: wFull,
      w_obj: wObj,
      w_num: wNum,
      k: k
    })
  }

  const highlightAliases = (text) => {
    if (!metadata?.aliases) return text

    let highlighted = text
    const aliases = Object.keys(metadata.aliases)

    // Sort by length (longest first) to avoid partial matches
    aliases.sort((a, b) => b.length - a.length)

    // Simple highlighting for display
    for (const alias of aliases) {
      const regex = new RegExp(`\\b${alias}\\b`, 'gi')
      if (regex.test(text.toLowerCase())) {
        return true // Has alias
      }
    }
    return false
  }

  const hasAlias = highlightAliases(query)

  // Ensure weights sum to ~1.0 for semantic channels
  const semanticSum = wTitle + wDesc + wSpecs
  const isSemanticNormalized = Math.abs(semanticSum - 1.0) < 0.01

  return (
    <div className="search-panel">
      <form onSubmit={handleSubmit}>
        <div className="search-input-row">
          <input
            type="text"
            className={`search-input ${hasAlias ? 'has-alias' : ''}`}
            placeholder="Enter search query (e.g., M10 x 1.5 thread 2 in wheel dia)"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            disabled={loading}
          />
          <button
            type="submit"
            className="search-button"
            disabled={loading || !query.trim()}
          >
            {loading ? 'Searching...' : 'Search'}
          </button>
          <button
            type="button"
            className="toggle-advanced"
            onClick={() => setShowAdvanced(!showAdvanced)}
          >
            {showAdvanced ? '▼' : '▶'} Weights
          </button>
        </div>

        {hasAlias && (
          <div className="alias-indicator">
            ✓ Field aliases detected in query
          </div>
        )}

        {showAdvanced && (
          <div className="advanced-controls">
            <div className="controls-grid">
              <div className="control-section">
                <h4>Semantic Channel Weights {!isSemanticNormalized && <span className="warning">⚠️ Should sum to 1.0</span>}</h4>
                <div className="control-group">
                  <label>
                    Title: {wTitle.toFixed(2)}
                    <input
                      type="range"
                      min="0"
                      max="1"
                      step="0.05"
                      value={wTitle}
                      onChange={(e) => setWTitle(parseFloat(e.target.value))}
                    />
                  </label>
                  <label>
                    Description: {wDesc.toFixed(2)}
                    <input
                      type="range"
                      min="0"
                      max="1"
                      step="0.05"
                      value={wDesc}
                      onChange={(e) => setWDesc(parseFloat(e.target.value))}
                    />
                  </label>
                  <label>
                    Specs: {wSpecs.toFixed(2)}
                    <input
                      type="range"
                      min="0"
                      max="1"
                      step="0.05"
                      value={wSpecs}
                      onChange={(e) => setWSpecs(parseFloat(e.target.value))}
                    />
                  </label>
                </div>
              </div>

              <div className="control-section">
                <h4>Query Type Weights</h4>
                <div className="control-group">
                  <label>
                    Full Query: {wFull.toFixed(2)}
                    <input
                      type="range"
                      min="0"
                      max="1"
                      step="0.05"
                      value={wFull}
                      onChange={(e) => setWFull(parseFloat(e.target.value))}
                    />
                  </label>
                  <label>
                    Object Only: {wObj.toFixed(2)}
                    <input
                      type="range"
                      min="0"
                      max="1"
                      step="0.05"
                      value={wObj}
                      onChange={(e) => setWObj(parseFloat(e.target.value))}
                    />
                  </label>
                </div>
              </div>

              <div className="control-section">
                <h4>Boost & Results</h4>
                <div className="control-group">
                  <label>
                    Numeric Boost: {wNum.toFixed(2)}
                    <input
                      type="range"
                      min="0"
                      max="1"
                      step="0.05"
                      value={wNum}
                      onChange={(e) => setWNum(parseFloat(e.target.value))}
                    />
                  </label>
                  <label>
                    Top K Results: {k}
                    <input
                      type="range"
                      min="5"
                      max="50"
                      step="5"
                      value={k}
                      onChange={(e) => setK(parseInt(e.target.value))}
                    />
                  </label>
                </div>
              </div>
            </div>

            <div className="presets">
              <button
                type="button"
                onClick={() => {
                  setWTitle(0.35); setWDesc(0.50); setWSpecs(0.15);
                  setWFull(0.95); setWObj(0.05); setWNum(0.25);
                }}
              >
                Reset to Defaults
              </button>
              <button
                type="button"
                onClick={() => {
                  setWTitle(0.50); setWDesc(0.30); setWSpecs(0.20);
                }}
              >
                Title-Heavy
              </button>
              <button
                type="button"
                onClick={() => {
                  setWTitle(0.20); setWDesc(0.20); setWSpecs(0.60);
                }}
              >
                Specs-Heavy
              </button>
            </div>
          </div>
        )}
      </form>
    </div>
  )
}

export default SearchPanel
