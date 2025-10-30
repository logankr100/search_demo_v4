# Setup Instructions for Your Partner

## Prerequisites
- Python 3.8+
- Node.js 16+ and npm
- Git
- OpenAI API key

## 1. Clone the Repository

```bash
git clone <your-repo-url>
cd gi-search-demo
git checkout frontend  # Switch to frontend branch
```

## 2. Set Up Environment Variables

Create a `.env` file in the project root with your OpenAI API key:

```bash
echo "OPENAI_API_KEY=your_openai_api_key_here" > .env
```

**Important**: Never commit the `.env` file! It's already in `.gitignore`.

## 3. Install Backend Dependencies

```bash
pip install -r api/requirements.txt
```

Or if using pip3:
```bash
pip3 install -r api/requirements.txt
```

## 4. Install Frontend Dependencies

```bash
cd frontend
npm install
cd ..
```

This will install ~500MB of node modules (takes 1-2 minutes).

## 5. Verify Data Files Exist

Make sure the `index_out_v2/` directory exists with these files:
- `meta.jsonl`
- `vectors_title.npy`
- `vectors_desc.npy`
- `vectors_specs.npy`
- `numeric_specs.npz`
- `numeric_schema.json`
- `fields.enriched.json`
- `alias_map.enriched.json`

If these files are missing, you'll need to run the data pipeline scripts:
```bash
python 00_harvest_spec_keys.py
python 01_categorize_strict.py
# ... etc
```

## 6. Start the Servers

### Option A: Using the Start Script (Recommended)
```bash
./start_frontend.sh
```

This starts both backend and frontend automatically.

### Option B: Manual Start (Two Terminals)

**Terminal 1 - Backend:**
```bash
cd api
python3 server.py
```
Backend runs on: http://localhost:5001

**Terminal 2 - Frontend:**
```bash
cd frontend
npm run dev
```
Frontend runs on: http://localhost:3000

## 7. Open the Application

Navigate to: **http://localhost:3000**

## 8. Test the Search

Try this query to verify everything works:
```
heavy duty caster 3 inch wheel diameter
```

You should see:
- Green highlight on "wheel diameter" (alias detected)
- 20 ranked results
- Debug information on the right panel

---

## Troubleshooting

### "Port 5000 in use"
**Issue**: macOS AirPlay Receiver uses port 5000
**Solution**: The backend now runs on port 5001 (already configured)

### "Module not found: flask"
**Issue**: Python dependencies not installed
**Solution**: Run `pip3 install -r api/requirements.txt`

### "Cannot find module 'react'"
**Issue**: Node modules not installed
**Solution**: Run `cd frontend && npm install`

### "Failed to load metadata"
**Issue**: Backend not running or wrong port
**Solution**:
- Check backend is running: `curl http://localhost:5001/api/health`
- Should return: `{"status":"ok"}`
- If not, restart backend: `cd api && python3 server.py`

### "No such file: index_out_v2/meta.jsonl"
**Issue**: Search index files missing
**Solution**:
- Verify files exist: `ls index_out_v2/`
- If missing, ask your partner to share the index files
- They're too large for git (1.4GB total), use Google Drive/Dropbox

### Frontend not updating after code changes
**Issue**: Vite cache
**Solution**:
- Stop frontend (Ctrl+C)
- Delete cache: `rm -rf frontend/.vite`
- Restart: `cd frontend && npm run dev`

---

## Project Structure

```
gi-search-demo/
├── api/
│   ├── server.py          # Flask backend (port 5001)
│   └── requirements.txt   # Python dependencies
├── frontend/
│   ├── src/               # React components
│   ├── package.json       # Node dependencies
│   └── vite.config.js     # Proxy config (5001 → 5001)
├── index_out_v2/          # Search index (1.4GB, not in git)
├── .env                   # OpenAI API key (not in git)
├── start_frontend.sh      # Startup script
└── README_FRONTEND.md     # Full documentation
```

---

## What's Already Configured

✅ Backend runs on port 5001 (avoids macOS AirPlay conflict)
✅ Frontend proxies `/api` requests to backend
✅ `.gitignore` excludes `node_modules/`, `.env`, etc.
✅ CORS enabled on backend
✅ Error handling for better debugging

---

## Development Workflow

1. **Make changes** to React components in `frontend/src/`
2. **Vite auto-reloads** - see changes instantly in browser
3. **Backend changes** require manual restart (Ctrl+C, then restart)

---

## Notes

- **OpenAI API costs**: Each search costs ~$0.0001 (embedding generation)
- **Search speed**: ~250-350ms per query (mostly OpenAI API time)
- **Data size**: 1422 products, 3 vector channels (1536 dims each)
- **Browser**: Chrome/Firefox recommended for best dev tools

---

## Getting Help

- Full docs: `README_FRONTEND.md`
- Quick start: `QUICKSTART.md`
- Architecture: `ARCHITECTURE.md`
- Frontend summary: `FRONTEND_SUMMARY.md`

---

## Success Checklist

- [ ] `.env` file created with OpenAI API key
- [ ] Backend dependencies installed (`pip3 install -r api/requirements.txt`)
- [ ] Frontend dependencies installed (`cd frontend && npm install`)
- [ ] Backend running on http://localhost:5001 (`curl http://localhost:5001/api/health` returns `{"status":"ok"}`)
- [ ] Frontend running on http://localhost:3000
- [ ] Can perform search and see results
- [ ] Debug panel shows timing and query analysis

If all checkboxes are ✅, you're good to go!
