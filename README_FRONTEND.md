# Semantic Search Engine - Frontend Developer Tool

A clean, developer-focused frontend for iterating on the semantic search engine design. This tool helps visualize search results, analyze scoring breakdowns, and tune search parameters in real-time.

## Features

- **Interactive Search Interface**: Test queries with real-time alias detection
- **Weight Tuning**: Adjust channel weights, query type weights, and numeric boosts with live sliders
- **Score Breakdown Visualization**: Detailed breakdown showing:
  - Semantic vs numeric boost contribution
  - Per-channel scores (title, description, specs)
  - Dual query scores (full vs object-only)
  - Raw score values for debugging
- **Constraint Editor**: Manually add/edit numeric constraints to filter products
- **Debug Panel**: Comprehensive debugging information including:
  - Query analysis and object extraction
  - Detected field aliases
  - Timing breakdowns for performance analysis
  - Active weights and parameters
  - Raw JSON response viewer
- **Expandable Product Cards**: Click to see full specs, descriptions, and product details

## Architecture

### Backend (Flask API)
- `api/server.py` - REST API wrapping the search functionality
- Endpoints:
  - `GET /api/health` - Health check
  - `GET /api/metadata` - Get field schema, aliases, and metadata
  - `POST /api/search` - Execute search with parameters

### Frontend (React)
- **SearchPanel** - Query input and weight controls
- **ConstraintEditor** - Numeric constraint management
- **ResultsView** - Product cards with expandable details
- **ScoreBreakdown** - Detailed scoring visualization
- **DebugPanel** - Search internals and timing information

## Setup Instructions

### Prerequisites
- Python 3.8+
- Node.js 16+ and npm
- OpenAI API key (for embeddings)

### 1. Backend Setup

```bash
# Install Python dependencies
pip install -r api/requirements.txt

# Ensure you have your .env file with OpenAI API key
echo "OPENAI_API_KEY=your_key_here" > .env

# Start the Flask server
cd api
python server.py
```

The backend will run on http://localhost:5000

### 2. Frontend Setup

```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Start the development server
npm run dev
```

The frontend will run on http://localhost:3000 and proxy API requests to the backend.

### 3. Access the Application

Open your browser to http://localhost:3000

## Usage Guide

### Basic Search Flow

1. **Enter a Query**: Type a search query in the input field
   - Example: "M10 x 1.5 thread 2 in wheel dia"
   - Queries with detected field aliases will highlight in green

2. **Adjust Weights (Optional)**: Click the "▶ Weights" button to expand advanced controls
   - **Semantic Channel Weights**: Control importance of title/desc/specs (should sum to 1.0)
   - **Query Type Weights**: Balance between full query and object-only query
   - **Numeric Boost**: Control the influence of numeric proximity
   - **Presets**: Quick weight configurations (Reset, Title-Heavy, Specs-Heavy)

3. **Add Constraints (Optional)**: Click "▶ Constraint Editor" to add numeric filters
   - Select a field from the dropdown (e.g., "Wheel Diameter")
   - Enter a target value
   - Optionally set a tolerance (default will be used if not specified)
   - Products must match ALL constraints to appear in results

4. **Execute Search**: Click "Search" button

5. **Analyze Results**:
   - View ranked products on the left
   - Check debug information on the right
   - Click any product card to expand and see:
     - Full description and specifications
     - Detailed score breakdown visualization
     - Product metadata (SKU, MPN, brand, URL)

### Weight Tuning Tips

- **Title-Heavy** (0.5, 0.3, 0.2): Good for brand name or product name queries
- **Specs-Heavy** (0.2, 0.2, 0.6): Good for technical specification queries
- **Balanced** (0.35, 0.5, 0.15): Default - works well for general queries

The semantic channel weights should sum to 1.0 for proper normalization. The UI will warn you if they don't.

### Constraint Editor Tips

- Constraints filter products BEFORE semantic search
- Only products matching ALL constraints will be scored
- Check the "Shortlist Size" in the debug panel to see how many products pass your constraints
- If shortlist is too small, relax your tolerance values
- If shortlist is too large, add more constraints or tighten tolerances

### Debug Panel Sections

- **Query Analysis**: See how your query was parsed and which aliases were detected
- **Timing Breakdown**: Performance metrics for each search stage
- **Active Weights**: Current parameter values
- **Raw JSON**: Full API response for detailed inspection (copy to clipboard available)

## Development

### Project Structure

```
gi-search-demo/
├── api/
│   ├── server.py           # Flask API server
│   └── requirements.txt    # Python dependencies
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── SearchPanel.jsx
│   │   │   ├── ConstraintEditor.jsx
│   │   │   ├── ResultsView.jsx
│   │   │   ├── ScoreBreakdown.jsx
│   │   │   └── DebugPanel.jsx
│   │   ├── App.jsx         # Main application component
│   │   └── main.jsx        # Entry point
│   ├── package.json
│   └── vite.config.js      # Vite configuration with proxy
├── 05_search.py            # Original search script
├── index_out_v2/           # Search index data
└── README_FRONTEND.md      # This file
```

### Customization

#### Adding New Fields to Constraint Editor
The constraint editor automatically loads numeric fields from `fields.enriched.json`. No code changes needed.

#### Adjusting Color Scheme
Colors are defined in CSS files using GitHub's dark theme palette. Key color variables:
- `#0d1117` - Background
- `#161b22` - Surface
- `#30363d` - Border
- `#c9d1d9` - Primary text
- `#8b949e` - Secondary text
- `#58a6ff` - Primary accent (blue)
- `#3fb950` - Success (green)
- `#f85149` - Error (red)

#### Adding New Presets
Edit `SearchPanel.jsx` and add buttons in the `.presets` section with your desired weight combinations.

## API Reference

### POST /api/search

Execute a semantic search query.

**Request Body:**
```json
{
  "query": "M10 x 1.5 thread 2 in wheel dia",
  "w_title": 0.35,
  "w_desc": 0.50,
  "w_specs": 0.15,
  "w_full": 0.95,
  "w_obj": 0.05,
  "w_num": 0.25,
  "k": 20,
  "constraints": [
    {
      "field": "wheel_diameter",
      "value": 2.0,
      "tolerance": 0.125
    }
  ]
}
```

**Response:**
```json
{
  "results": [
    {
      "rank": 1,
      "score": 0.5036,
      "semantic_score": 0.4821,
      "numeric_boost": 0.0215,
      "channel_scores": {
        "title_full": 0.1689,
        "desc_full": 0.2411,
        "specs_full": 0.0721,
        "title_obj": 0.0088,
        "desc_obj": 0.0126,
        "specs_obj": 0.0038
      },
      "product": {
        "sku": "B2803564",
        "name": "Product Name",
        "brand": "Brand Name",
        "mpn": "101277",
        "url": "https://...",
        "description": "...",
        "title": "...",
        "specs": "..."
      }
    }
  ],
  "debug": {
    "query": "M10 x 1.5 thread 2 in wheel dia",
    "object_query": "thread wheel",
    "alias_hits": [
      {"alias": "wheel dia", "field": "wheel_diameter", "position": 20}
    ],
    "constraints": [
      {"field": "wheel_diameter", "value": 2.0, "tolerance": 0.125, "matched": true, "count": 45}
    ],
    "shortlist_size": 45,
    "candidate_count": 45,
    "weights": {...},
    "timings": {
      "object_extraction": 0.001,
      "alias_detection": 0.002,
      "shortlist_build": 0.015,
      "embedding": 0.234,
      "semantic_scoring": 0.042,
      "numeric_boost": 0.008,
      "total": 0.302
    }
  }
}
```

## Troubleshooting

### Backend Issues

**Error: "No module named 'flask'"**
```bash
pip install -r api/requirements.txt
```

**Error: "OpenAI API key not found"**
- Ensure `.env` file exists in the project root with `OPENAI_API_KEY=your_key_here`
- Or set environment variable: `export OPENAI_API_KEY=your_key_here`

**Error: "File not found: index_out_v2/meta.jsonl"**
- Ensure you've run the data pipeline scripts to generate the search index
- Check that `index_out_v2/` directory exists with all required files

### Frontend Issues

**Error: "Failed to load metadata"**
- Ensure the Flask backend is running on port 5000
- Check browser console for CORS errors
- Verify the proxy configuration in `vite.config.js`

**Results not updating**
- Check browser console for API errors
- Verify the search request payload in Network tab
- Ensure backend is responding (check terminal logs)

**Weights warning showing**
- Semantic channel weights (title, desc, specs) should sum to 1.0
- Adjust sliders so the sum equals 1.0

## Performance Tips

- **Reduce candidate pool**: Lower `k` value for faster searches
- **Use constraints**: Numeric constraints filter products before semantic search
- **Monitor timing panel**: Identify slow stages (usually embedding or semantic scoring)
- **FAISS indexing**: Consider using FAISS indices for faster similarity search on large datasets

## Future Enhancements

Potential improvements for iteration:
- A/B comparison mode (side-by-side results with different weights)
- Query history and saved presets
- Bulk query testing with CSV import
- Score distribution histograms
- Export results to CSV/JSON
- Real-time search (search as you type)
- Product image thumbnails
- Spec comparison table (side-by-side product specs)

## License

MIT
