import { useState, useEffect } from 'react'
import SearchPanel from './components/SearchPanel'
import ConstraintEditor from './components/ConstraintEditor'
import ResultsView from './components/ResultsView'
import DebugPanel from './components/DebugPanel'
import './App.css'

function App() {
  const [metadata, setMetadata] = useState(null)
  const [searchResults, setSearchResults] = useState(null)
  const [debugInfo, setDebugInfo] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [constraints, setConstraints] = useState([])
  const [showConstraintEditor, setShowConstraintEditor] = useState(false)

  // Load metadata on mount
  useEffect(() => {
    fetch('/api/metadata')
      .then(res => res.json())
      .then(data => setMetadata(data))
      .catch(err => console.error('Failed to load metadata:', err))
  }, [])

  const handleSearch = async (searchParams) => {
    setLoading(true)
    setError(null)

    try {
      const response = await fetch('/api/search', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          ...searchParams,
          constraints: constraints
        })
      })

      // Check if response is ok before parsing
      if (!response.ok) {
        const text = await response.text()
        setError(`Server error (${response.status}): ${text}`)
        setLoading(false)
        return
      }

      // Try to parse JSON
      let data
      try {
        data = await response.json()
      } catch (parseErr) {
        const text = await response.text()
        setError(`Invalid JSON response: ${text.substring(0, 100)}`)
        setLoading(false)
        return
      }

      setSearchResults(data.results)
      setDebugInfo(data.debug)
    } catch (err) {
      setError(`Network error: ${err.message}`)
    } finally {
      setLoading(false)
    }
  }

  const handleConstraintsChange = (newConstraints) => {
    setConstraints(newConstraints)
  }

  return (
    <div className="app">
      <header className="app-header">
        <h1>🔍 Semantic Search Engine</h1>
        <span className="subtitle">Developer Tool</span>
      </header>

      <div className="app-body">
        <div className="top-section">
          <SearchPanel
            onSearch={handleSearch}
            loading={loading}
            metadata={metadata}
          />

          <div className="constraint-section">
            <button
              className="toggle-constraints"
              onClick={() => setShowConstraintEditor(!showConstraintEditor)}
            >
              {showConstraintEditor ? '▼' : '▶'} Constraint Editor
              {constraints.length > 0 && (
                <span className="constraint-badge">{constraints.length}</span>
              )}
            </button>

            {showConstraintEditor && metadata && (
              <ConstraintEditor
                constraints={constraints}
                onConstraintsChange={handleConstraintsChange}
                fields={metadata.fields}
                numericSchema={metadata.numeric_schema}
              />
            )}
          </div>
        </div>

        {error && (
          <div className="error-message">
            <strong>Error:</strong> {error}
          </div>
        )}

        {(searchResults || debugInfo) && (
          <div className="results-container">
            <div className="results-main">
              <ResultsView
                results={searchResults}
                loading={loading}
              />
            </div>
            <div className="results-debug">
              <DebugPanel
                debugInfo={debugInfo}
                loading={loading}
              />
            </div>
          </div>
        )}

        {!searchResults && !loading && !error && (
          <div className="empty-state">
            <p>Enter a search query to get started</p>
            <p className="hint">Try: "M10 x 1.5 thread 2 in wheel dia"</p>
          </div>
        )}
      </div>
    </div>
  )
}

export default App
