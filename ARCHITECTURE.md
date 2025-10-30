# Frontend Architecture

## System Overview

```
┌─────────────────────────────────────────────────────────────┐
│                         Browser                             │
│  ┌───────────────────────────────────────────────────────┐ │
│  │              React Frontend (Port 3000)               │ │
│  │                                                        │ │
│  │  ┌──────────────┐  ┌──────────────┐  ┌────────────┐  │ │
│  │  │ SearchPanel  │  │ Constraint   │  │ ResultsView│  │ │
│  │  │   Component  │  │   Editor     │  │  Component │  │ │
│  │  └──────────────┘  └──────────────┘  └────────────┘  │ │
│  │                                                        │ │
│  │  ┌──────────────┐  ┌──────────────┐                  │ │
│  │  │ DebugPanel   │  │ ScoreBreakdn │                  │ │
│  │  │  Component   │  │  Component   │                  │ │
│  │  └──────────────┘  └──────────────┘                  │ │
│  │                                                        │ │
│  └────────────────────────┬───────────────────────────────┘ │
└────────────────────────────┼──────────────────────────────────┘
                             │ HTTP/JSON
                             │ (Proxied via Vite)
                             ▼
┌─────────────────────────────────────────────────────────────┐
│              Flask Backend (Port 5000)                      │
│  ┌───────────────────────────────────────────────────────┐ │
│  │                    server.py                          │ │
│  │                                                        │ │
│  │  GET  /api/health      → Health check                │ │
│  │  GET  /api/metadata    → Schema, aliases, fields     │ │
│  │  POST /api/search      → Execute search              │ │
│  │                                                        │ │
│  └────────────────────────┬───────────────────────────────┘ │
└────────────────────────────┼──────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                  Search Engine Core                         │
│  ┌───────────────────────────────────────────────────────┐ │
│  │  • spec_patterns.py   - Value parsing                 │ │
│  │  • spec_lexicon.py    - Unit families & tolerances    │ │
│  │  • 05_search.py       - Original search script        │ │
│  └───────────────────────────────────────────────────────┘ │
└────────────────────────────┬──────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                      Data Layer                             │
│  ┌───────────────────────────────────────────────────────┐ │
│  │  index_out_v2/                                        │ │
│  │    ├── meta.jsonl              (Product metadata)     │ │
│  │    ├── vectors_title.npy       (1422 × 1536)         │ │
│  │    ├── vectors_desc.npy        (1422 × 1536)         │ │
│  │    ├── vectors_specs.npy       (1422 × 1536)         │ │
│  │    ├── numeric_specs.npz       (Values + mask)        │ │
│  │    ├── numeric_schema.json     (114 attributes)       │ │
│  │    ├── fields.enriched.json    (Field definitions)    │ │
│  │    └── alias_map.enriched.json (Aliases → fields)     │ │
│  └───────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                  External Services                          │
│  ┌───────────────────────────────────────────────────────┐ │
│  │  OpenAI API (text-embedding-3-small)                  │ │
│  │    - Query embedding generation                       │ │
│  │    - ~150-300ms per query                             │ │
│  └───────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

---

## Component Architecture

### React Frontend

```
App.jsx (Root Component)
├── State Management
│   ├── metadata (from /api/metadata)
│   ├── searchResults (from /api/search)
│   ├── debugInfo (from /api/search)
│   ├── constraints (user-defined)
│   └── loading/error states
│
├── SearchPanel
│   ├── Query input
│   ├── Weight controls (collapsible)
│   │   ├── Semantic channel weights (title/desc/specs)
│   │   ├── Query type weights (full/obj)
│   │   ├── Numeric boost weight
│   │   └── Top K selector
│   └── Preset buttons
│
├── ConstraintEditor (collapsible)
│   ├── Constraint list
│   │   └── Remove/edit constraints
│   ├── Add constraint form
│   │   ├── Field selector (dropdown)
│   │   ├── Value input
│   │   └── Tolerance input
│   └── Help text
│
├── ResultsView
│   ├── Results header (count)
│   └── Result cards (expandable)
│       ├── Header (rank, SKU, score, name)
│       ├── Description (collapsed)
│       └── Expanded view
│           ├── Details (full desc, specs, metadata)
│           └── ScoreBreakdown
│
├── ScoreBreakdown
│   ├── Final score summary
│   ├── Composition bars (semantic + numeric)
│   ├── Channel scores (6 channels)
│   └── Raw values grid
│
└── DebugPanel
    ├── Query Analysis (collapsible)
    │   ├── Original vs object query
    │   ├── Alias hits
    │   ├── Constraints status
    │   └── Shortlist/candidate size
    ├── Timing Breakdown (collapsible)
    │   └── Visual timing bars
    ├── Active Weights (collapsible)
    └── Raw JSON (collapsible + copy)
```

---

## Data Flow

### Search Request Flow

```
1. User enters query
   ↓
2. User clicks "Search"
   ↓
3. App.jsx collects:
   - Query text
   - Weight parameters
   - Constraints array
   ↓
4. POST /api/search
   {
     query: "...",
     w_title: 0.35,
     w_desc: 0.50,
     ...
     constraints: [...]
   }
   ↓
5. Flask server.py:
   a. Extract object query
   b. Detect aliases in query
   c. Build numeric shortlist (filter by constraints)
   d. Get embeddings from OpenAI
   e. Compute semantic scores (3 channels × 2 query types)
   f. Compute numeric proximity boost
   g. Rank by final score
   ↓
6. Return JSON response:
   {
     results: [...],
     debug: {
       query: "...",
       object_query: "...",
       alias_hits: [...],
       constraints: [...],
       shortlist_size: N,
       timings: {...}
     }
   }
   ↓
7. App.jsx updates state:
   - setSearchResults(results)
   - setDebugInfo(debug)
   ↓
8. Components re-render:
   - ResultsView shows products
   - DebugPanel shows debug info
```

### Metadata Loading Flow

```
1. App.jsx mounts
   ↓
2. useEffect fires
   ↓
3. GET /api/metadata
   ↓
4. Flask returns:
   {
     fields: [...],           // 114+ field definitions
     aliases: {...},          // Alias → field mappings
     numeric_schema: {...},   // Numeric field schema
     product_count: 1422
   }
   ↓
5. App.jsx sets metadata state
   ↓
6. ConstraintEditor receives fields
   ↓
7. Dropdown populates with numeric fields
```

---

## Backend API Architecture

### Flask Server Structure

```python
server.py
├── Global Cache (_cache)
│   └── Loaded on first search:
│       ├── meta (JSONL)
│       ├── vectors (3 × NPY)
│       ├── numeric data (NPZ)
│       ├── schema (JSON)
│       └── aliases (JSON)
│
├── Helper Functions
│   ├── load_search_data()        - Lazy load & cache
│   ├── get_embedding()           - OpenAI API call
│   ├── extract_object_query()    - Simple noun extraction
│   ├── detect_aliases()          - Find aliases in query
│   ├── build_numeric_shortlist() - Hard-spec filtering
│   ├── compute_semantic_scores() - Cosine similarity
│   └── compute_numeric_boost()   - Exponential decay
│
└── Endpoints
    ├── /api/health              - Health check
    ├── /api/metadata            - Schema info
    └── /api/search              - Main search
```

### Search Algorithm in Backend

```
1. Extract object query (remove numbers/units)
   "M10 x 1.5 thread 2 in wheel dia" → "thread wheel"

2. Detect aliases
   "wheel dia" → wheel_diameter
   "thread" → [various thread fields]

3. Build numeric shortlist
   For each constraint:
     - Find field index in schema
     - Filter products by value ± tolerance
   Intersection (AND) of all constraints
   → shortlist_indices

4. Get embeddings
   - Full query → query_vec_full
   - Object query → query_vec_obj

5. Compute semantic scores (for shortlist only)
   For each channel (title/desc/specs):
     - scores_full = cosine(query_vec_full, product_vecs) * weight
     - scores_obj = cosine(query_vec_obj, product_vecs) * weight
   semantic_scores = w_full * scores_full + w_obj * scores_obj

6. Compute numeric boost
   For each constraint:
     - distance = |product_value - target_value|
     - boost = exp(-distance / scale)
   Average boost across constraints

7. Final ranking
   final_scores = semantic_scores + w_num * numeric_boosts
   Sort descending, take top K

8. Build response
   - results: product metadata + scores
   - debug: timing, aliases, constraints, weights
```

---

## File Structure

```
gi-search-demo/
├── api/
│   ├── __init__.py             # Package init
│   ├── server.py               # Flask REST API
│   └── requirements.txt        # Python dependencies
│
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── SearchPanel.jsx         # Query + weights
│   │   │   ├── SearchPanel.css
│   │   │   ├── ConstraintEditor.jsx    # Numeric filters
│   │   │   ├── ConstraintEditor.css
│   │   │   ├── ResultsView.jsx         # Product cards
│   │   │   ├── ResultsView.css
│   │   │   ├── ScoreBreakdown.jsx      # Score viz
│   │   │   ├── ScoreBreakdown.css
│   │   │   ├── DebugPanel.jsx          # Debug info
│   │   │   └── DebugPanel.css
│   │   ├── App.jsx                     # Root component
│   │   ├── App.css
│   │   ├── main.jsx                    # Entry point
│   │   └── index.css                   # Global styles
│   ├── index.html                      # HTML template
│   ├── package.json                    # Node dependencies
│   ├── vite.config.js                  # Build config
│   └── .gitignore                      # Git ignore
│
├── index_out_v2/                       # Search index (existing)
│   ├── meta.jsonl
│   ├── vectors_*.npy
│   ├── numeric_specs.npz
│   ├── numeric_schema.json
│   ├── fields.enriched.json
│   └── alias_map.enriched.json
│
├── spec_patterns.py                    # Value parsing (existing)
├── spec_lexicon.py                     # Units/tolerances (existing)
├── 05_search.py                        # Original search (existing)
│
├── start_frontend.sh                   # Startup script
├── README_FRONTEND.md                  # Complete docs
├── QUICKSTART.md                       # Quick start
├── FRONTEND_SUMMARY.md                 # Implementation summary
└── ARCHITECTURE.md                     # This file
```

---

## Technology Decisions

### Why Flask?
- **Lightweight**: Minimal overhead for simple API
- **Python-native**: Direct imports of search modules
- **CORS built-in**: Easy cross-origin setup
- **Development friendly**: Auto-reload, simple debugging

### Why React?
- **Component-based**: Clean separation of concerns
- **State management**: Simple useState for this scale
- **Ecosystem**: Rich tooling and libraries
- **Developer familiar**: Widely known

### Why Vite?
- **Fast**: HMR in <50ms
- **Simple**: Zero config for React
- **Proxy**: Easy backend integration
- **Modern**: ES modules, optimized builds

### Why Not...?

**Next.js?**
- Overkill for this use case
- No need for SSR or routing
- Single-page app is sufficient

**FastAPI?**
- Would work equally well
- Flask is simpler for this scale
- No async benefits here (sequential search)

**GraphQL?**
- REST is sufficient for 3 endpoints
- No need for flexible queries
- Simpler for frontend devs

**Redux?**
- State is simple enough for useState
- No complex state interactions
- Avoid boilerplate overhead

---

## Performance Considerations

### Backend Bottlenecks
1. **OpenAI API** (150-300ms)
   - Network latency
   - Embedding generation
   - Mitigation: Cache embeddings, use FAISS

2. **Semantic Scoring** (30-80ms)
   - Cosine similarity on 1422 vectors
   - Mitigation: Use FAISS indices

3. **Data Loading** (2s cold start)
   - Loading NPY files
   - Mitigation: Keep server running, pre-load

### Frontend Bottlenecks
1. **Initial Bundle** (~500KB dev)
   - React + dependencies
   - Mitigation: Lazy load components

2. **Result Rendering** (<100ms)
   - 20 product cards
   - Mitigation: Virtualize if >100 results

3. **Expand Animation** (<50ms)
   - CSS transitions
   - No mitigation needed

---

## Security Considerations

### Current State (Development)
- **No authentication**: Open API
- **No rate limiting**: Unlimited requests
- **No input validation**: Trust user input
- **CORS**: Wide open (`*`)
- **API key exposed**: In backend .env

### Production Recommendations
- [ ] Add API authentication (JWT tokens)
- [ ] Rate limit search endpoint (e.g., 10 req/min)
- [ ] Validate/sanitize query input
- [ ] Restrict CORS to frontend domain
- [ ] Use secrets manager for API key
- [ ] Add request logging
- [ ] Implement HTTPS

---

## Deployment Considerations

### Current (Development)
```
Backend:  python3 server.py (Flask dev server)
Frontend: npm run dev (Vite dev server)
```

### Production Options

**Option 1: Docker Compose**
```yaml
services:
  backend:
    build: ./api
    ports: ["5000:5000"]
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}

  frontend:
    build: ./frontend
    ports: ["80:80"]
    depends_on: [backend]
```

**Option 2: Separate Deployments**
- Backend: Heroku, AWS Lambda, Google Cloud Run
- Frontend: Vercel, Netlify, S3 + CloudFront

**Option 3: All-in-One**
- Flask serves built React app
- Single deployment target
- Simpler but less scalable

---

## Extension Points

### Adding New Features

**New Search Parameter**
1. Add slider to SearchPanel.jsx
2. Pass in search request
3. Use in server.py scoring
4. Display in DebugPanel

**New Visualization**
1. Create component in components/
2. Import in ResultsView.jsx or DebugPanel.jsx
3. Pass result/debug data as props
4. Style with matching CSS

**New API Endpoint**
1. Add route in server.py
2. Add fetch call in App.jsx
3. Use response in components

**New Constraint Type**
1. Extend ConstraintEditor.jsx UI
2. Modify server.py shortlist logic
3. Update constraint data structure

---

## Summary

This architecture provides:
- **Clean separation**: React ↔ Flask ↔ Search Core
- **Modular design**: Easy to extend/modify
- **Developer-friendly**: Clear data flow, good DX
- **Production-ready foundation**: Easy to deploy with minor changes

The focus on transparency and debugging makes it ideal for iterating on search algorithm design.
