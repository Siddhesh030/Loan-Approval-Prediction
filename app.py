import os
import pickle
import numpy as np
from flask import Flask, render_template_string, request, jsonify

app = Flask(__name__)

# Model loading logic
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'Xg_model.pkl')

def load_model():
    if os.path.exists(MODEL_PATH):
        try:
            with open(MODEL_PATH, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"[CrediPulse Engine] Model load failure: {e}")
            return None
    return None

model = load_model()

# Feature order mapping for XGBoost
FEATURE_KEYS = [
    'no_of_dependents', 'education', 'self_employed', 'income_annum',
    'loan_amount', 'loan_term', 'cibil_score', 'residential_assets_value',
    'commercial_assets_value', 'luxury_assets_value', 'bank_asset_value'
]

HTML_LAYOUT = """
<!DOCTYPE html>
<html lang="en" data-theme="light">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CrediPulse Enterprise | Automated Credit Underwriting</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.0/css/all.min.css">
    
    <style>
        :root[data-theme="light"] {
            --bg-page: #f8fafc;
            --surface-primary: #ffffff;
            --surface-secondary: #f1f5f9;
            --border: #e2e8f0;
            --border-focus: #3b82f6;
            --text-heading: #0f172a;
            --text-body: #334155;
            --text-muted: #64748b;
            --brand-primary: #2563eb;
            --brand-accent: #1d4ed8;
            --status-approved-bg: #ecfdf5;
            --status-approved-text: #059669;
            --status-approved-border: #a7f3d0;
            --status-rejected-bg: #fef2f2;
            --status-rejected-text: #dc2626;
            --status-rejected-border: #fecaca;
            --card-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -2px rgba(0, 0, 0, 0.05);
        }

        :root[data-theme="dark"] {
            --bg-page: #090d16;
            --surface-primary: #111827;
            --surface-secondary: #1e293b;
            --border: #334155;
            --border-focus: #60a5fa;
            --text-heading: #f8fafc;
            --text-body: #cbd5e1;
            --text-muted: #94a3b8;
            --brand-primary: #3b82f6;
            --brand-accent: #60a5fa;
            --status-approved-bg: rgba(5, 150, 105, 0.15);
            --status-approved-text: #34d399;
            --status-approved-border: rgba(52, 211, 153, 0.3);
            --status-rejected-bg: rgba(220, 38, 38, 0.15);
            --status-rejected-text: #f87171;
            --status-rejected-border: rgba(248, 113, 113, 0.3);
            --card-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.5);
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            transition: background 0.25s ease, border-color 0.25s ease, color 0.25s ease;
        }

        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background-color: var(--bg-page);
            color: var(--text-body);
            min-height: 100vh;
        }

        .header {
            background: var(--surface-primary);
            border-bottom: 1px solid var(--border);
            padding: 1rem 2.5rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
            position: sticky;
            top: 0;
            z-index: 50;
        }

        .brand-group {
            display: flex;
            align-items: center;
            gap: 0.85rem;
        }

        .brand-icon {
            width: 40px;
            height: 40px;
            border-radius: 10px;
            background: linear-gradient(135deg, var(--brand-primary), #0284c7);
            display: flex;
            align-items: center;
            justify-content: center;
            color: #fff;
            font-size: 1.2rem;
        }

        .brand-name {
            font-size: 1.25rem;
            font-weight: 800;
            letter-spacing: -0.5px;
            color: var(--text-heading);
        }

        .brand-sub {
            font-size: 0.72rem;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: var(--text-muted);
            font-weight: 600;
        }

        .toolbar {
            display: flex;
            align-items: center;
            gap: 1rem;
        }

        .control-pill {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            background: var(--surface-secondary);
            border: 1px solid var(--border);
            padding: 0.4rem 0.8rem;
            border-radius: 8px;
            font-size: 0.85rem;
            color: var(--text-heading);
            font-weight: 600;
        }

        .control-pill select {
            background: transparent;
            border: none;
            color: inherit;
            font-size: inherit;
            font-weight: inherit;
            cursor: pointer;
            outline: none;
        }

        .theme-toggle-btn {
            background: var(--surface-secondary);
            border: 1px solid var(--border);
            color: var(--text-heading);
            width: 36px;
            height: 36px;
            border-radius: 8px;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
        }

        .layout {
            max-width: 1440px;
            margin: 2rem auto;
            padding: 0 2rem;
            display: grid;
            grid-template-columns: 1.1fr 0.9fr;
            gap: 2rem;
        }

        @media (max-width: 1080px) {
            .layout { grid-template-columns: 1fr; }
            .header { padding: 1rem 1.5rem; }
        }

        .card {
            background: var(--surface-primary);
            border: 1px solid var(--border);
            border-radius: 14px;
            box-shadow: var(--card-shadow);
            padding: 1.75rem;
        }

        .card-header {
            margin-bottom: 1.5rem;
            display: flex;
            align-items: center;
            justify-content: space-between;
            border-bottom: 1px solid var(--border);
            padding-bottom: 1rem;
        }

        .card-title {
            font-size: 1.1rem;
            font-weight: 700;
            color: var(--text-heading);
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .form-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 1.25rem;
        }

        @media (max-width: 640px) {
            .form-grid { grid-template-columns: 1fr; }
        }

        .col-span-2 { grid-column: span 2; }
        @media (max-width: 640px) { .col-span-2 { grid-column: span 1; } }

        .field {
            display: flex;
            flex-direction: column;
            gap: 0.4rem;
        }

        .field label {
            font-size: 0.78rem;
            font-weight: 700;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.4px;
        }

        .field input, .field select {
            background: var(--surface-secondary);
            border: 1px solid var(--border);
            padding: 0.75rem 0.9rem;
            border-radius: 8px;
            color: var(--text-heading);
            font-size: 0.95rem;
            font-weight: 500;
            outline: none;
            width: 100%;
        }

        .field input:focus, .field select:focus {
            border-color: var(--border-focus);
            box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.15);
        }

        .submit-btn {
            background: var(--brand-primary);
            color: #ffffff;
            border: none;
            padding: 0.9rem;
            border-radius: 8px;
            font-size: 0.95rem;
            font-weight: 700;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 0.5rem;
            margin-top: 1rem;
        }

        .submit-btn:hover { background: var(--brand-accent); }

        .decision-banner {
            border-radius: 12px;
            padding: 1.5rem;
            text-align: center;
            margin-bottom: 1.5rem;
            border: 1px dashed var(--border);
            background: var(--surface-secondary);
        }

        .decision-banner.approved {
            background: var(--status-approved-bg);
            border: 1px solid var(--status-approved-border);
            color: var(--status-approved-text);
        }

        .decision-banner.rejected {
            background: var(--status-rejected-bg);
            border: 1px solid var(--status-rejected-border);
            color: var(--status-rejected-text);
        }

        .decision-title {
            font-size: 1.4rem;
            font-weight: 800;
            letter-spacing: -0.3px;
        }

        .decision-sub {
            font-size: 0.85rem;
            margin-top: 0.25rem;
            font-weight: 500;
        }

        .kpi-row {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 1rem;
            margin-bottom: 1.5rem;
        }

        @media (max-width: 640px) {
            .kpi-row { grid-template-columns: 1fr; }
        }

        .kpi-box {
            background: var(--surface-secondary);
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 1rem;
        }

        .kpi-label {
            font-size: 0.72rem;
            font-weight: 700;
            text-transform: uppercase;
            color: var(--text-muted);
        }

        .kpi-value {
            font-size: 1.25rem;
            font-weight: 800;
            color: var(--text-heading);
            margin-top: 0.25rem;
            font-family: 'JetBrains Mono', monospace;
        }

        .breakdown-row {
            display: flex;
            flex-direction: column;
            gap: 0.85rem;
        }

        .asset-meter {
            display: flex;
            flex-direction: column;
            gap: 0.3rem;
        }

        .asset-meter-meta {
            display: flex;
            justify-content: space-between;
            font-size: 0.8rem;
            font-weight: 600;
        }

        .asset-bar-bg {
            height: 8px;
            background: var(--surface-secondary);
            border-radius: 999px;
            overflow: hidden;
            border: 1px solid var(--border);
        }

        .asset-bar-fill {
            height: 100%;
            background: var(--brand-primary);
            border-radius: 999px;
            width: 0%;
            transition: width 0.8s cubic-bezier(0.4, 0, 0.2, 1);
        }
    </style>
</head>
<body>

    <header class="header">
        <div class="brand-group">
            <div class="brand-icon">
                <i class="fa-solid fa-shield-halved"></i>
            </div>
            <div>
                <div class="brand-name">CrediPulse Enterprise</div>
                <div class="brand-sub">ML Underwriting System</div>
            </div>
        </div>

        <div class="toolbar">
            <div class="control-pill">
                <i class="fa-solid fa-coins" style="color: var(--text-muted);"></i>
                <select id="currencySelect" onchange="switchCurrency()">
                    <option value="$" data-rate="1">USD ($)</option>
                    <option value="₹" data-rate="83.5">INR (₹)</option>
                    <option value="€" data-rate="0.92">EUR (€)</option>
                    <option value="£" data-rate="0.78">GBP (£)</option>
                </select>
            </div>
            <button class="theme-toggle-btn" onclick="toggleTheme()" title="Toggle appearance">
                <i id="themeIcon" class="fa-solid fa-moon"></i>
            </button>
        </div>
    </header>

    <main class="layout">
        
        <!-- Applicant Profile Card -->
        <section class="card">
            <div class="card-header">
                <div class="card-title">
                    <i class="fa-solid fa-user-tie" style="color: var(--brand-primary);"></i>
                    Borrower Financial Profile
                </div>
                <span style="font-size: 0.75rem; font-weight: 700; color: var(--text-muted); text-transform: uppercase;">11 Feature Vector</span>
            </div>

            <form id="loanForm" class="form-grid">
                
                <div class="field">
                    <label>Dependents</label>
                    <input type="number" name="no_of_dependents" min="0" max="10" value="2" required>
                </div>

                <div class="field">
                    <label>Education</label>
                    <select name="education">
                        <option value="1">Graduate</option>
                        <option value="0">Undergraduate</option>
                    </select>
                </div>

                <div class="field">
                    <label>Employment</label>
                    <select name="self_employed">
                        <option value="0">Salaried / Employed</option>
                        <option value="1">Self Employed</option>
                    </select>
                </div>

                <div class="field">
                    <label>Loan Duration (Years)</label>
                    <input type="number" name="loan_term" min="1" max="30" value="10" required>
                </div>

                <div class="field">
                    <label>Annual Gross Income (<span class="curr-mark">$</span>)</label>
                    <input type="number" name="income_annum" step="1000" value="6500000" required>
                </div>

                <div class="field">
                    <label>Principal Loan Amount (<span class="curr-mark">$</span>)</label>
                    <input type="number" name="loan_amount" step="1000" value="15000000" required>
                </div>

                <div class="field col-span-2">
                    <label>Bureau CIBIL Score (300 - 900)</label>
                    <input type="number" name="cibil_score" min="300" max="900" value="750" required>
                </div>

                <div class="field">
                    <label>Residential Assets (<span class="curr-mark">$</span>)</label>
                    <input type="number" name="residential_assets_value" value="4000000" required>
                </div>

                <div class="field">
                    <label>Commercial Assets (<span class="curr-mark">$</span>)</label>
                    <input type="number" name="commercial_assets_value" value="2500000" required>
                </div>

                <div class="field">
                    <label>Luxury Assets (<span class="curr-mark">$</span>)</label>
                    <input type="number" name="luxury_assets_value" value="8000000" required>
                </div>

                <div class="field">
                    <label>Liquid Bank Deposits (<span class="curr-mark">$</span>)</label>
                    <input type="number" name="bank_asset_value" value="5000000" required>
                </div>

                <button type="submit" class="submit-btn col-span-2">
                    <i class="fa-solid fa-bolt"></i> Run Instant Evaluation
                </button>
            </form>
        </section>

        <!-- Evaluation Results & Diagnostics -->
        <section class="card">
            <div class="card-header">
                <div class="card-title">
                    <i class="fa-solid fa-chart-pie" style="color: var(--brand-primary);"></i>
                    Underwriting Assessment
                </div>
                <span id="decisionTimestamp" style="font-size: 0.75rem; color: var(--text-muted); font-family: 'JetBrains Mono', monospace;">Ready</span>
            </div>

            <!-- Underwriting Decision Banner -->
            <div id="statusCard" class="decision-banner">
                <div class="decision-title" id="decisionTitle">System Ready</div>
                <div class="decision-sub" id="decisionSub">Input customer parameters and execute assessment</div>
            </div>

            <!-- Core Financial Metrics -->
            <div class="kpi-row">
                <div class="kpi-box">
                    <div class="kpi-label">Approval Confidence</div>
                    <div class="kpi-value" id="kpiConfidence">--</div>
                </div>
                <div class="kpi-box">
                    <div class="kpi-label">Debt-to-Income (DTI)</div>
                    <div class="kpi-value" id="kpiDti">--</div>
                </div>
                <div class="kpi-box">
                    <div class="kpi-label">Cumulative Assets</div>
                    <div class="kpi-value" id="kpiAssets">--</div>
                </div>
            </div>

            <!-- Asset Collateral Breakdown -->
            <div class="card-header" style="margin-top: 1rem; padding-bottom: 0.5rem; margin-bottom: 1rem;">
                <div class="card-title" style="font-size: 0.95rem;">
                    <i class="fa-solid fa-layer-group" style="font-size: 0.85rem; color: var(--text-muted);"></i>
                    Collateral Portfolio Breakdown
                </div>
            </div>

            <div class="breakdown-row">
                <div class="asset-meter">
                    <div class="asset-meter-meta">
                        <span>Residential Assets</span>
                        <span id="txtRes">--</span>
                    </div>
                    <div class="asset-bar-bg"><div id="barRes" class="asset-bar-fill"></div></div>
                </div>

                <div class="asset-meter">
                    <div class="asset-meter-meta">
                        <span>Commercial Real Estate</span>
                        <span id="txtCom">--</span>
                    </div>
                    <div class="asset-bar-bg"><div id="barCom" class="asset-bar-fill"></div></div>
                </div>

                <div class="asset-meter">
                    <div class="asset-meter-meta">
                        <span>Luxury & Tangible Assets</span>
                        <span id="txtLux">--</span>
                    </div>
                    <div class="asset-bar-bg"><div id="barLux" class="asset-bar-fill"></div></div>
                </div>

                <div class="asset-meter">
                    <div class="asset-meter-meta">
                        <span>Liquid Bank Balances</span>
                        <span id="txtBnk">--</span>
                    </div>
                    <div class="asset-bar-bg"><div id="barBnk" class="asset-bar-fill"></div></div>
                </div>
            </div>
        </section>

    </main>

    <script>
        let cachedResult = null;

        function toggleTheme() {
            const html = document.documentElement;
            const current = html.getAttribute('data-theme');
            const target = current === 'dark' ? 'light' : 'dark';
            html.setAttribute('data-theme', target);
            document.getElementById('themeIcon').className = target === 'dark' ? 'fa-solid fa-sun' : 'fa-solid fa-moon';
        }

        function switchCurrency() {
            const sym = document.getElementById('currencySelect').value;
            document.querySelectorAll('.curr-mark').forEach(el => el.innerText = sym);
            if (cachedResult) renderResults(cachedResult);
        }

        function formatFinancial(val, sym) {
            if (val >= 10000000) return sym + (val / 10000000).toFixed(2) + ' Cr';
            if (val >= 100000) return sym + (val / 100000).toFixed(2) + ' L';
            if (val >= 1000) return sym + (val / 1000).toFixed(1) + ' K';
            return sym + val.toLocaleString();
        }

        function renderResults(data) {
            const sym = document.getElementById('currencySelect').value;
            const banner = document.getElementById('statusCard');
            const title = document.getElementById('decisionTitle');
            const sub = document.getElementById('decisionSub');

            if (data.status === 'Approved') {
                banner.className = 'decision-banner approved';
                title.innerHTML = '<i class="fa-solid fa-circle-check"></i> Credit Facility Approved';
                sub.innerText = 'Application fulfills institutional risk threshold criteria.';
            } else {
                banner.className = 'decision-banner rejected';
                title.innerHTML = '<i class="fa-solid fa-triangle-exclamation"></i> Facility Request Declined';
                sub.innerText = 'High credit exposure or asset-to-debt shortfall detected.';
            }

            document.getElementById('decisionTimestamp').innerText = 'Evaluated: ' + new Date().toLocaleTimeString();
            document.getElementById('kpiConfidence').innerText = (data.probability * 100).toFixed(1) + '%';
            document.getElementById('kpiDti').innerText = data.dti + 'x';
            document.getElementById('kpiAssets').innerText = formatFinancial(data.total_assets, sym);

            // Populate Breakdown Bars
            const assets = data.assets_breakdown;
            const total = Math.max(data.total_assets, 1);

            const map = [
                { k: 'residential', bar: 'barRes', txt: 'txtRes' },
                { k: 'commercial', bar: 'barCom', txt: 'txtCom' },
                { k: 'luxury', bar: 'barLux', txt: 'txtLux' },
                { k: 'bank', bar: 'barBnk', txt: 'txtBnk' }
            ];

            map.forEach(item => {
                const amount = assets[item.k];
                const pct = Math.round((amount / total) * 100);
                document.getElementById(item.bar).style.width = pct + '%';
                document.getElementById(item.txt).innerText = formatFinancial(amount, sym) + ' (' + pct + '%)';
            });
        }

        document.getElementById('loanForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            const submitBtn = e.target.querySelector('button[type="submit"]');
            submitBtn.disabled = true;
            submitBtn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Processing Model Weights...';

            try {
                const res = await fetch('/predict', {
                    method: 'POST',
                    body: new FormData(e.target)
                });
                const data = await res.json();
                if (data.error) throw new Error(data.error);

                cachedResult = data;
                renderResults(data);
            } catch (err) {
                alert('Evaluation Error: ' + err.message);
            } finally {
                submitBtn.disabled = false;
                submitBtn.innerHTML = '<i class="fa-solid fa-bolt"></i> Run Instant Evaluation';
            }
        });
    </script>
</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(HTML_LAYOUT)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        features = [float(request.form.get(k, 0)) for k in FEATURE_KEYS]
        
        income = float(request.form.get('income_annum', 1))
        loan_amount = float(request.form.get('loan_amount', 0))
        res_asset = float(request.form.get('residential_assets_value', 0))
        com_asset = float(request.form.get('commercial_assets_value', 0))
        lux_asset = float(request.form.get('luxury_assets_value', 0))
        bnk_asset = float(request.form.get('bank_asset_value', 0))
        
        total_assets = res_asset + com_asset + lux_asset + bnk_asset
        dti = round(loan_amount / income, 2) if income > 0 else 0.0

        if model is not None:
            input_array = np.array([features])
            raw_pred = int(model.predict(input_array)[0])
            
            # The XGBoost model sets class 0 as Approved and 1 as Rejected
            is_approved = (raw_pred == 0)
            
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(input_array)[0]
                probability = float(probs[0]) if is_approved else float(probs[1])
            else:
                probability = 0.94 if is_approved else 0.15
        else:
            # Fallback underwriting rules if pickle is unmounted
            cibil = float(request.form.get('cibil_score', 300))
            is_approved = (cibil >= 650 and dti < 3.5)
            probability = 0.85 if is_approved else 0.20

        return jsonify({
            'status': "Approved" if is_approved else "Rejected",
            'probability': round(probability, 4),
            'total_assets': total_assets,
            'dti': dti,
            'assets_breakdown': {
                'residential': res_asset,
                'commercial': com_asset,
                'luxury': lux_asset,
                'bank': bnk_asset
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# Vercel WSGI entrypoint export
app = app

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
