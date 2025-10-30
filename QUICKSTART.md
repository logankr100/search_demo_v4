# Quick Start Guide

## 🚀 Get Started in 3 Steps

### 1. Install & Setup
```bash
# Install Python dependencies
pip install -r api/requirements.txt

# Install Node dependencies
cd frontend && npm install && cd ..

# Create .env file with your OpenAI API key
echo "OPENAI_API_KEY=your_key_here" > .env
```

### 2. Start the Servers
```bash
# Option A: Use the start script (recommended)
./start_frontend.sh

# Option B: Start manually in separate terminals
# Terminal 1 - Backend
cd api && python3 server.py

# Terminal 2 - Frontend
cd frontend && npm run dev
```

### 3. Open Browser
Navigate to **http://localhost:3000**

---

## 📝 Example Queries

Try these sample queries to see the search engine in action:

1. **"M10 x 1.5 thread 2 in wheel dia"**
   - Tests alias detection (thread, wheel dia)
   - Shows numeric extraction
   - Demonstrates multi-channel scoring

2. **"stainless steel bolt"**
   - Material + product type query
   - Title and description focused

3. **"3000 psi pressure capacity"**
   - Numeric specification query
   - Tests numeric proximity boost

---

## 🎛️ Key Features

### Search Panel
- **Query Input**: Type your search query
- **Weights Toggle**: Click "▶ Weights" to adjust parameters
- **Presets**: Quick buttons for common weight configurations

### Constraint Editor
- **Add Constraints**: Filter by numeric specs (e.g., diameter, length, pressure)
- **Set Tolerances**: Control how strict the matching should be
- **Live Filtering**: See shortlist size update in real-time

### Results View
- **Product Cards**: Click to expand for full details
- **Score Display**: See final score for each result
- **Score Breakdown**: Expand to see semantic vs numeric contribution

### Debug Panel
- **Query Analysis**: Object extraction and alias detection
- **Timing**: Performance metrics for each search stage
- **Weights**: Active parameter values
- **Raw JSON**: Copy full response for debugging

---

## 🔧 Common Tasks

### Tune Channel Weights
1. Click "▶ Weights" to expand controls
2. Adjust sliders:
   - **Title**: Product name importance
   - **Description**: Description text importance
   - **Specs**: Technical specs importance
3. Ensure they sum to 1.0
4. Run search to see effect

### Add Numeric Constraint
1. Click "▶ Constraint Editor"
2. Select field from dropdown (e.g., "Wheel Diameter")
3. Enter target value (e.g., 2)
4. Optionally set tolerance (e.g., 0.125 for ±1/8")
5. Click "+ Add"
6. Run search

### Compare Score Breakdowns
1. Run a search
2. Click on multiple product cards to expand them
3. Compare the score breakdown visualizations:
   - Which channel contributed most?
   - How much did numeric boost help?
   - Are full vs object scores different?

### Debug Performance
1. Run a search
2. Open "Timing Breakdown" in debug panel
3. Identify slow stages:
   - **Embedding**: OpenAI API call
   - **Semantic Scoring**: Vector similarity
   - **Shortlist Build**: Numeric filtering

---

## ⚙️ Default Parameters

```
Channel Weights (Semantic):
- Title:       0.35
- Description: 0.50
- Specs:       0.15

Query Type Weights:
- Full Query:  0.95
- Object Only: 0.05

Boost:
- Numeric:     0.25

Results:
- Top K:       20
```

---

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| Backend won't start | Check `.env` has valid OpenAI API key |
| Frontend shows "Failed to load metadata" | Ensure backend is running on port 5000 |
| No results returned | Check shortlist size in debug panel - may need to relax constraints |
| Weights warning | Adjust title/desc/specs sliders to sum to 1.0 |
| Slow searches | Check timing panel - embedding API calls can be slow |

---

## 📚 More Information

See **README_FRONTEND.md** for complete documentation including:
- Architecture details
- API reference
- Customization guide
- Performance tips
- Future enhancements
