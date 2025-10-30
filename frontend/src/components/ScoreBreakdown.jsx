import React from 'react'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, Cell } from 'recharts'
import './ScoreBreakdown.css'

function ScoreBreakdown({ result }) {
  const { score, semantic_score, numeric_boost, channel_scores } = result

  // Prepare data for chart
  const semanticData = [
    { name: 'Title (Full)', value: channel_scores.title_full, color: '#58a6ff' },
    { name: 'Desc (Full)', value: channel_scores.desc_full, color: '#79c0ff' },
    { name: 'Specs (Full)', value: channel_scores.specs_full, color: '#a5d6ff' },
    { name: 'Title (Obj)', value: channel_scores.title_obj, color: '#f778ba' },
    { name: 'Desc (Obj)', value: channel_scores.desc_obj, color: '#faa6d8' },
    { name: 'Specs (Obj)', value: channel_scores.specs_obj, color: '#fcc9e8' },
  ]

  const compositionData = [
    { name: 'Semantic', value: semantic_score, color: '#58a6ff' },
    { name: 'Numeric Boost', value: numeric_boost, color: '#3fb950' },
  ]

  return (
    <div className="score-breakdown">
      {/* Final Score */}
      <div className="score-summary">
        <div className="score-item score-final">
          <span className="score-label">Final Score</span>
          <span className="score-value">{score.toFixed(4)}</span>
        </div>
        <div className="score-equation">
          <div className="score-item">
            <span className="score-label">Semantic</span>
            <span className="score-value semantic">{semantic_score.toFixed(4)}</span>
          </div>
          <span className="plus">+</span>
          <div className="score-item">
            <span className="score-label">Numeric Boost</span>
            <span className="score-value numeric">{numeric_boost.toFixed(4)}</span>
          </div>
        </div>
      </div>

      {/* Composition Chart */}
      <div className="chart-section">
        <h5>Score Composition</h5>
        <div className="bar-container">
          {compositionData.map((item) => (
            <div key={item.name} className="bar-item">
              <span className="bar-label">{item.name}</span>
              <div className="bar-wrapper">
                <div
                  className="bar-fill"
                  style={{
                    width: `${(item.value / score) * 100}%`,
                    backgroundColor: item.color
                  }}
                />
                <span className="bar-value">{item.value.toFixed(4)}</span>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Channel Breakdown */}
      <div className="chart-section">
        <h5>Channel Scores (Weighted)</h5>
        <div className="channel-grid">
          {semanticData.map((item) => (
            <div key={item.name} className="channel-item">
              <div className="channel-header">
                <span className="channel-name">{item.name}</span>
                <span className="channel-value">{item.value.toFixed(4)}</span>
              </div>
              <div className="channel-bar">
                <div
                  className="channel-bar-fill"
                  style={{
                    width: `${Math.min((item.value / semantic_score) * 100, 100)}%`,
                    backgroundColor: item.color
                  }}
                />
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Raw Values */}
      <div className="raw-values">
        <h5>Raw Values</h5>
        <div className="raw-grid">
          <div className="raw-item">
            <span className="raw-label">Final Score:</span>
            <code className="raw-code">{score.toFixed(6)}</code>
          </div>
          <div className="raw-item">
            <span className="raw-label">Semantic Score:</span>
            <code className="raw-code">{semantic_score.toFixed(6)}</code>
          </div>
          <div className="raw-item">
            <span className="raw-label">Numeric Boost:</span>
            <code className="raw-code">{numeric_boost.toFixed(6)}</code>
          </div>
          <div className="raw-item">
            <span className="raw-label">Title (Full):</span>
            <code className="raw-code">{channel_scores.title_full.toFixed(6)}</code>
          </div>
          <div className="raw-item">
            <span className="raw-label">Desc (Full):</span>
            <code className="raw-code">{channel_scores.desc_full.toFixed(6)}</code>
          </div>
          <div className="raw-item">
            <span className="raw-label">Specs (Full):</span>
            <code className="raw-code">{channel_scores.specs_full.toFixed(6)}</code>
          </div>
          <div className="raw-item">
            <span className="raw-label">Title (Obj):</span>
            <code className="raw-code">{channel_scores.title_obj.toFixed(6)}</code>
          </div>
          <div className="raw-item">
            <span className="raw-label">Desc (Obj):</span>
            <code className="raw-code">{channel_scores.desc_obj.toFixed(6)}</code>
          </div>
          <div className="raw-item">
            <span className="raw-label">Specs (Obj):</span>
            <code className="raw-code">{channel_scores.specs_obj.toFixed(6)}</code>
          </div>
        </div>
      </div>
    </div>
  )
}

export default ScoreBreakdown
