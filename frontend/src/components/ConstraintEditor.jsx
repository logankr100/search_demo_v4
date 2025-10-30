import React, { useState } from 'react'
import './ConstraintEditor.css'

function ConstraintEditor({ constraints, onConstraintsChange, fields, numericSchema }) {
  const [newConstraint, setNewConstraint] = useState({
    field: '',
    value: '',
    tolerance: ''
  })

  // Get numeric fields only
  const numericFields = fields?.filter(f => f.type === 'numeric') || []

  const handleAddConstraint = () => {
    if (!newConstraint.field || !newConstraint.value) {
      return
    }

    const value = parseFloat(newConstraint.value)
    if (isNaN(value)) {
      alert('Please enter a valid numeric value')
      return
    }

    const constraint = {
      field: newConstraint.field,
      value: value,
      tolerance: newConstraint.tolerance ? parseFloat(newConstraint.tolerance) : undefined
    }

    onConstraintsChange([...constraints, constraint])

    // Reset form
    setNewConstraint({ field: '', value: '', tolerance: '' })
  }

  const handleRemoveConstraint = (index) => {
    const updated = constraints.filter((_, i) => i !== index)
    onConstraintsChange(updated)
  }

  const handleUpdateConstraint = (index, updates) => {
    const updated = [...constraints]
    updated[index] = { ...updated[index], ...updates }
    onConstraintsChange(updated)
  }

  const getFieldInfo = (fieldId) => {
    return numericFields.find(f => f.id === fieldId)
  }

  return (
    <div className="constraint-editor">
      <div className="constraint-list">
        {constraints.length === 0 && (
          <div className="no-constraints">
            No constraints defined. Add constraints to filter products by numeric specifications.
          </div>
        )}

        {constraints.map((constraint, index) => {
          const fieldInfo = getFieldInfo(constraint.field)
          return (
            <div key={index} className="constraint-item">
              <div className="constraint-field">
                <span className="field-label">{fieldInfo?.label || constraint.field}</span>
                {fieldInfo?.unit_family && (
                  <span className="field-unit">({fieldInfo.unit_family})</span>
                )}
              </div>
              <div className="constraint-controls">
                <input
                  type="number"
                  step="any"
                  className="constraint-input"
                  value={constraint.value}
                  onChange={(e) =>
                    handleUpdateConstraint(index, { value: parseFloat(e.target.value) })
                  }
                />
                <span className="constraint-separator">±</span>
                <input
                  type="number"
                  step="any"
                  className="constraint-input tolerance"
                  placeholder="tol"
                  value={constraint.tolerance || ''}
                  onChange={(e) =>
                    handleUpdateConstraint(index, {
                      tolerance: e.target.value ? parseFloat(e.target.value) : undefined
                    })
                  }
                />
                <button
                  className="constraint-remove"
                  onClick={() => handleRemoveConstraint(index)}
                  title="Remove constraint"
                >
                  ×
                </button>
              </div>
            </div>
          )
        })}
      </div>

      <div className="constraint-add">
        <select
          className="constraint-select"
          value={newConstraint.field}
          onChange={(e) => setNewConstraint({ ...newConstraint, field: e.target.value })}
        >
          <option value="">Select field...</option>
          {numericFields.map((field) => (
            <option key={field.id} value={field.id}>
              {field.label} {field.unit_family && `(${field.unit_family})`}
            </option>
          ))}
        </select>

        <input
          type="number"
          step="any"
          className="constraint-input"
          placeholder="Value"
          value={newConstraint.value}
          onChange={(e) => setNewConstraint({ ...newConstraint, value: e.target.value })}
          onKeyPress={(e) => {
            if (e.key === 'Enter') {
              handleAddConstraint()
            }
          }}
        />

        <input
          type="number"
          step="any"
          className="constraint-input tolerance"
          placeholder="Tolerance"
          value={newConstraint.tolerance}
          onChange={(e) => setNewConstraint({ ...newConstraint, tolerance: e.target.value })}
          onKeyPress={(e) => {
            if (e.key === 'Enter') {
              handleAddConstraint()
            }
          }}
        />

        <button
          className="constraint-add-button"
          onClick={handleAddConstraint}
          disabled={!newConstraint.field || !newConstraint.value}
        >
          + Add
        </button>
      </div>

      <div className="constraint-help">
        <p>
          <strong>Tip:</strong> Constraints filter products before semantic search. Only products
          matching ALL constraints will be included in results.
        </p>
        <p>
          Tolerance is optional. If not specified, a small default tolerance will be used based on
          the field type.
        </p>
      </div>
    </div>
  )
}

export default ConstraintEditor
