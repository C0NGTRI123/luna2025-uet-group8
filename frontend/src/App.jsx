import React, { useState } from 'react'
import axios from 'axios'

function App() {
  const [imageFile, setImageFile] = useState(null)
  const [noduleFile, setNoduleFile] = useState(null)
  const [clinicalFile, setClinicalFile] = useState(null)
  const [mode, setMode] = useState('2D')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [results, setResults] = useState(null)

  const handleSubmit = async (e) => {
    e.preventDefault()
    
    if (!imageFile || !noduleFile) {
      setError('Please upload CT image and nodule locations files')
      return
    }

    setLoading(true)
    setError(null)
    setResults(null)

    const formData = new FormData()
    formData.append('image', imageFile)
    formData.append('nodule_locations', noduleFile)
    if (clinicalFile) {
      formData.append('clinical_information', clinicalFile)
    }
    formData.append('mode', mode)

    try {
      const response = await axios.post('/api/v1/predict/', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      })
      setResults(response.data)
    } catch (err) {
      setError(err.response?.data?.detail || 'Prediction failed. Please try again.')
    } finally {
      setLoading(false)
    }
  }

  const getMalignancyClass = (probability) => {
    if (probability >= 0.7) return 'score-high'
    if (probability >= 0.4) return 'score-medium'
    return 'score-low'
  }

  const formatProbability = (prob) => {
    return (prob * 100).toFixed(2) + '%'
  }

  return (
    <div className="app">
      <div className="header">
        <h1>LUNA25 Nodule Malignancy Prediction</h1>
        <p>Upload CT images and nodule locations to predict malignancy risk</p>
      </div>

      <div className="container">
        <form onSubmit={handleSubmit}>
          <div className="form-group">
            <label>CT Image File (.mha) *</label>
            <input
              type="file"
              accept=".mha"
              onChange={(e) => setImageFile(e.target.files[0])}
              required
            />
            {imageFile && <div className="file-info">✓ {imageFile.name}</div>}
          </div>

          <div className="form-group">
            <label>Nodule Locations (JSON) *</label>
            <input
              type="file"
              accept=".json"
              onChange={(e) => setNoduleFile(e.target.files[0])}
              required
            />
            {noduleFile && <div className="file-info">✓ {noduleFile.name}</div>}
          </div>

          <div className="form-group">
            <label>Clinical Information (JSON) <span className="optional">(Optional)</span></label>
            <input
              type="file"
              accept=".json"
              onChange={(e) => setClinicalFile(e.target.files[0])}
            />
            {clinicalFile && <div className="file-info">✓ {clinicalFile.name}</div>}
          </div>

          <div className="form-group">
            <label>Prediction Mode</label>
            <div className="mode-selector">
              <label>
                <input
                  type="radio"
                  value="2D"
                  checked={mode === '2D'}
                  onChange={(e) => setMode(e.target.value)}
                />
                2D Model
              </label>
              <label>
                <input
                  type="radio"
                  value="3D"
                  checked={mode === '3D'}
                  onChange={(e) => setMode(e.target.value)}
                />
                3D Model
              </label>
            </div>
          </div>

          <button type="submit" className="submit-btn" disabled={loading}>
            {loading ? 'Predicting...' : 'Predict Malignancy'}
          </button>
        </form>

        {loading && (
          <div className="loading">
            <div className="spinner"></div>
            <p>Analyzing nodules... This may take a moment.</p>
          </div>
        )}

        {error && (
          <div className="error">
            <strong>Error:</strong> {error}
          </div>
        )}

        {results && (
          <div className="results">
            <h2>Prediction Results</h2>
            <p style={{ marginBottom: '20px', color: '#666' }}>
              Found {results.points.length} nodule(s) - Mode: {mode}
            </p>
            
            {results.points.map((nodule, index) => (
              <div key={index} className="nodule-card">
                <div className="nodule-header">
                  <div className="nodule-id">
                    {nodule.name || `Nodule ${index + 1}`}
                  </div>
                  <div className={`malignancy-score ${getMalignancyClass(nodule.probability)}`}>
                    {formatProbability(nodule.probability)}
                  </div>
                </div>
                
                <div className="nodule-details">
                  <div className="detail-item">
                    <span className="detail-label">X Coordinate:</span>
                    <span>{nodule.point[0].toFixed(2)}</span>
                  </div>
                  <div className="detail-item">
                    <span className="detail-label">Y Coordinate:</span>
                    <span>{nodule.point[1].toFixed(2)}</span>
                  </div>
                  <div className="detail-item">
                    <span className="detail-label">Z Coordinate:</span>
                    <span>{nodule.point[2].toFixed(2)}</span>
                  </div>
                  <div className="detail-item">
                    <span className="detail-label">Risk Level:</span>
                    <span>
                      {nodule.probability >= 0.7 ? 'High' : 
                       nodule.probability >= 0.4 ? 'Medium' : 'Low'}
                    </span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}

export default App
