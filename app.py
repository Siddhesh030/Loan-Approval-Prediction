import os
import pickle
import numpy as np
from flask import Flask, render_template_string, request, jsonify

app = Flask(__name__)

MODEL_PATH = os.environ.get('MODEL_PATH', 'Xg_model.pkl')

def load_model():
    if os.path.exists(MODEL_PATH):
        try:
            with open(MODEL_PATH, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"[ERROR] Failed to load model from {MODEL_PATH}: {e}")
            return None
    print(f"[WARN] Model file '{MODEL_PATH}' not found. Using fallback engine.")
    return None

model = load_model()

# Exact feature sequence required by XGBoost booster
FEATURE_KEYS = [
    'no_of_dependents', 'education', 'self_employed', 'income_annum',
    'loan_amount', 'loan_term', 'cibil_score', 'residential_assets_value',
    'commercial_assets_value', 'luxury_assets_value', 'bank_asset_value'
]

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en" data-theme="emerald">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Loan Approval Prediction System</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.2/css/all.min.css">
    
    <style>
        /* Theme Palettes */
        :root[data-theme="emerald"] {
            --bg-base: #061412;
            --bg-surface: rgba(10, 31, 28, 0.8);
            --bg-input: #04100e;
            --accent-primary: #10b981;
            --accent-secondary: #06b6d4;
            --accent-glow: rgba(16, 185, 129, 0.35);
            --border-subtle: rgba(16, 185, 129, 0.18);
            --text-main: #f0fdf4;
            --text-muted: #6ee7b7;
        }

        :root[data-theme="obsidian"] {
            --bg-base: #020617;
            --bg-surface: rgba(15, 23, 42, 0.8);
            --bg-input: #090d16;
            --accent-primary: #38bdf8;
            --accent-secondary: #6366f1;
            --accent-glow: rgba(56, 189, 248, 0.35);
            --border-subtle: rgba(255, 255, 255, 0.08);
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
        }

        :root[data-theme="purple"] {
            --bg-base: #0d061a;
            --bg-surface: rgba(26, 14, 48, 0.8);
            --bg-input: #090312;
            --accent-primary: #a855f7;
            --accent-secondary: #ec4899;
            --accent-glow: rgba(168, 85, 247, 0.35);
            --border-subtle: rgba(168, 85, 247, 0.2);
            --text-main: #faf5ff;
            --text-muted: #d8b4fe;
        }

        :root[data-theme="sunset"] {
            --bg-base: #180907;
            --bg-surface: rgba(38, 17, 13, 0.8);
            --bg-input: #100503;
            --accent-primary: #f97316;
            --accent-secondary: #eab308;
            --accent-glow: rgba(249, 115, 22, 0.35);
            --border-subtle: rgba(249, 115, 22, 0.2);
            --text-main: #fff7ed;
            --text-muted: #fdba74;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Plus Jakarta Sans', sans-serif;
            transition: background-color 0.35s ease, border-color 0.35s ease, color 0.35s ease, box-shadow 0.35s ease;
        }

        body {
            background-color: var(--bg-base);
            color: var(--text-main);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            background-image: 
                radial-gradient(circle at 10% 10%, var(--accent-glow), transparent 35%),
                radial-gradient(circle at 90% 90%, rgba(99, 102, 241, 0.08), transparent 35%);
        }

        /* Cockpit Header */
        .navbar {
            height: 72px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0 2rem;
            border-bottom: 1px solid var(--border-subtle);
            background: rgba(2, 6, 23, 0.6);
            backdrop-filter: blur(14px);
            position: sticky;
            top: 0;
            z-index: 50;
        }

        .brand-cluster {
            display: flex;
            align-items: center;
            gap: 0.9rem;
        }

        .brand-badge {
            width: 40px;
            height: 40px;
            border-radius: 10px;
            background: linear-gradient(135deg, var(--accent-primary), var(--accent-secondary));
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.2rem;
            color: #000;
            font-weight: 800;
            box-shadow: 0 0 20px var(--accent-glow);
        }

        .brand-title {
            font-size: 1.15rem;
            font-weight: 800;
            letter-spacing: -0.01em;
            color: var(--text-main);
        }

        .brand-subtitle {
            font-size: 0.7rem;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            color: var(--text-muted);
        }

        .nav-meta {
            display: flex;
            align-items: center;
            gap: 1rem;
        }

        .theme-selector {
            display: flex;
            align-items: center;
            gap: 0.4rem;
            background: rgba(0, 0, 0, 0.3);
            border: 1px solid var(--border-subtle);
            padding: 0.35rem 0.65rem;
            border-radius: 30px;
        }

        .theme-dot {
            width: 18px;
            height: 18px;
            border-radius: 50%;
            cursor: pointer;
            border: 2px solid transparent;
            transition: transform 0.2s;
        }

        .theme-dot:hover { transform: scale(1.2); }
        .theme-dot.active { border-color: #fff; transform: scale(1.1); }
        .dot-emerald { background: linear-gradient(135deg, #10b981, #06b6d4); }
        .dot-obsidian { background: linear-gradient(135deg, #38bdf8, #6366f1); }
        .dot-purple { background: linear-gradient(135deg, #a855f7, #ec4899); }
        .dot-sunset { background: linear-gradient(135deg, #f97316, #eab308); }

        .currency-switch {
            background: var(--bg-input);
            border: 1px solid var(--border-subtle);
            border-radius: 8px;
            color: var(--text-main);
            padding: 0.4rem 0.75rem;
            font-size: 0.8rem;
            font-weight: 600;
            outline: none;
            cursor: pointer;
        }

        /* Content Grid */
        .workspace {
            max-width: 1480px;
            width: 100%;
            margin: 0 auto;
            padding: 2rem;
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 2rem;
            flex: 1;
        }

        @media (max-width: 1100px) {
            .workspace {
                grid-template-columns: 1fr;
                padding: 1.25rem;
            }
        }

        .surface-card {
            background: var(--bg-surface);
            border: 1px solid var(--border-subtle);
            border-radius: 18px;
            padding: 1.85rem;
            backdrop-filter: blur(16px);
            display: flex;
            flex-direction: column;
            gap: 1.5rem;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.4);
        }

        .card-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            border-bottom: 1px solid var(--border-subtle);
            padding-bottom: 1rem;
        }

        .card-heading {
            font-size: 1.05rem;
            font-weight: 700;
            display: flex;
            align-items: center;
            gap: 0.65rem;
            color: var(--text-main);
        }

        .card-heading i { color: var(--accent-primary); }

        .field-group-title {
            font-size: 0.75rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: var(--accent-primary);
            margin-bottom: 0.85rem;
            display: flex;
            align-items: center;
            gap: 0.45rem;
        }

        .grid-2 {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1rem;
        }

        @media (max-width: 600px) {
            .grid-2 { grid-template-columns: 1fr; }
        }

        .input-block {
            display: flex;
            flex-direction: column;
            gap: 0.4rem;
        }

        .input-block label {
            font-size: 0.75rem;
            font-weight: 600;
            color: var(--text-muted);
        }

        .input-shell {
            position: relative;
            display: flex;
            align-items: center;
        }

        .input-shell i {
            position: absolute;
            left: 0.85rem;
            color: var(--text-muted);
            font-size: 0.85rem;
            pointer-events: none;
        }

        .input-shell input, .input-shell select {
            width: 100%;
            background: var(--bg-input);
            border: 1px solid var(--border-subtle);
            border-radius: 10px;
            color: var(--text-main);
            font-size: 0.85rem;
            padding: 0.65rem 0.85rem 0.65rem 2.4rem;
            outline: none;
            font-family: 'JetBrains Mono', monospace;
        }

        .input-shell input:focus, .input-shell select:focus {
            border-color: var(--accent-primary);
            box-shadow: 0 0 0 3px var(--accent-glow);
        }

        .btn-evaluate {
            background: linear-gradient(135deg, var(--accent-primary), var(--accent-secondary));
            color: #000;
            font-size: 0.95rem;
            font-weight: 800;
            padding: 1rem;
            border-radius: 12px;
            border: none;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 0.6rem;
            box-shadow: 0 4px 20px var(--accent-glow);
            margin-top: 0.5rem;
        }

        .btn-evaluate:hover {
            transform: translateY(-2px);
            filter: brightness(1.1);
        }

        /* Verdict Box */
        .verdict-banner {
            border-radius: 14px;
            padding: 1.5rem;
            display: flex;
            align-items: center;
            justify-content: space-between;
            background: rgba(0, 0, 0, 0.25);
            border: 1px solid var(--border-subtle);
        }

        .verdict-banner.approved {
            background: linear-gradient(135deg, rgba(16, 185, 129, 0.15), rgba(0, 0, 0, 0.4));
            border-color: #10b981;
        }

        .verdict-banner.rejected {
            background: linear-gradient(135deg, rgba(239, 68, 68, 0.15), rgba(0, 0, 0, 0.4));
            border-color: #ef4444;
        }

        .verdict-title-sub {
            font-size: 0.7rem;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            color: var(--text-muted);
        }

        .verdict-badge {
            font-size: 1.5rem;
            font-weight: 800;
            color: var(--text-main);
            margin-top: 0.25rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .tier-chip {
            font-size: 0.7rem;
            font-family: 'JetBrains Mono', monospace;
            padding: 0.35rem 0.65rem;
            border-radius: 6px;
            background: rgba(0, 0, 0, 0.3);
            border: 1px solid var(--border-subtle);
            color: var(--text-muted);
        }

        /* Scoreboard */
        .scoreboard {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 1rem;
        }

        @media (max-width: 600px) {
            .scoreboard { grid-template-columns: 1fr; }
        }

        .score-tile {
            background: rgba(0, 0, 0, 0.25);
            border: 1px solid var(--border-subtle);
            border-radius: 12px;
            padding: 1rem;
            display: flex;
            flex-direction: column;
            gap: 0.3rem;
        }

        .score-label {
            font-size: 0.7rem;
            text-transform: uppercase;
            color: var(--text-muted);
            font-weight: 600;
        }

        .score-val {
            font-size: 1.25rem;
            font-weight: 700;
            font-family: 'JetBrains Mono', monospace;
            color: var(--accent-primary);
        }

        /* Visual Stack Bar */
        .asset-stack-container {
            display: flex;
            flex-direction: column;
            gap: 0.8rem;
        }

        .stack-bar {
            height: 12px;
            width: 100%;
            background: var(--bg-input);
            border-radius: 6px;
            overflow: hidden;
            display: flex;
        }

        .stack-segment {
            height: 100%;
            transition: width 0.8s cubic-bezier(0.16, 1, 0.3, 1);
        }

        .seg-res { background: #38bdf8; }
        .seg-com { background: #818cf8; }
        .seg-lux { background: #f472b6; }
        .seg-bnk { background: #34d399; }

        .legend-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 0.65rem;
            font-size: 0.75rem;
        }

        .legend-item {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0.5rem 0.75rem;
            border-radius: 8px;
            background: rgba(0, 0, 0, 0.25);
            border: 1px solid var(--border-subtle);
        }

        .legend-tag {
            display: flex;
            align-items: center;
            gap: 0.45rem;
            color: var(--text-muted);
        }

        .dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
        }

        .legend-amount {
            font-family: 'JetBrains Mono', monospace;
            font-weight: 600;
            color: var(--text-main);
        }

        .audit-box {
            background: rgba(0, 0, 0, 0.25);
            border: 1px solid var(--border-subtle);
            border-radius: 12px;
            padding: 1rem;
            font-size: 0.75rem;
            color: var(--text-muted);
            line-height: 1.6;
        }

        .audit-header {
            font-weight: 700;
            color: var(--text-main);
            margin-bottom: 0.35rem;
            display: flex;
            align-items: center;
            gap: 0.45rem;
        }
    </style>
</head>
<body>

    <header class="navbar">
        <div class="brand-cluster">
            <div class="brand-badge"><i class="fa-solid fa-scale-balanced"></i></div>
            <div>
                <div class="brand-title">Loan Approval Prediction System</div>
                <div class="brand-subtitle">Automated Risk & Credit Evaluation</div>
            </div>
        </div>

        <div class="nav-meta">
            <!-- Dynamic Theme Selector -->
            <div class="theme-selector">
                <span style="font-size: 0.7rem; color: var(--text-muted); margin-right: 4px; font-weight: 700;">THEME</span>
                <div class="theme-dot dot-emerald active" onclick="setTheme('emerald')" title="Emerald Neon"></div>
                <div class="theme-dot dot-obsidian" onclick="setTheme('obsidian')" title="Obsidian Cyan"></div>
                <div class="theme-dot dot-purple" onclick="setTheme('purple')" title="Royal Purple"></div>
                <div class="theme-dot dot-sunset" onclick="setTheme('sunset')" title="Sunset Gold"></div>
            </div>

            <!-- Currency Switcher -->
            <select id="currencySelect" class="currency-switch" onchange="applyCurrencyChange()">
                <option value="$" data-rate="1">USD ($)</option>
                <option value="₹" data-rate="83.5">INR (₹)</option>
                <option value="€" data-rate="0.92">EUR (€)</option>
                <option value="£" data-rate="0.78">GBP (£)</option>
            </select>
        </div>
    </header>

    <main class="workspace">
        
        <!-- Input Terminal -->
        <section class="surface-card">
            <div class="card-header">
                <span class="card-heading">
                    <i class="fa-solid fa-sliders"></i> Applicant Risk Parameters
                </span>
                <span style="font-size: 0.7rem; color: var(--text-muted);">Institutional Verification</span>
            </div>

            <form id="loanForm" style="display: flex; flex-direction: column; gap: 1.25rem;">
                
                <div>
                    <div class="field-group-title">
                        <i class="fa-solid fa-user-check"></i> Demographic & Credit Score
                    </div>
                    <div class="grid-2">
                        <div class="input-block">
                            <label>Dependents Count</label>
                            <div class="input-shell">
                                <i class="fa-solid fa-users"></i>
                                <input type="number" name="no_of_dependents" min="0" max="8" value="2" required>
                            </div>
                        </div>

                        <div class="input-block">
                            <label>Education Status</label>
                            <div class="input-shell">
                                <i class="fa-solid fa-graduation-cap"></i>
                                <select name="education">
                                    <option value="1">Graduate</option>
                                    <option value="0">Not Graduate</option>
                                </select>
                            </div>
                        </div>

                        <div class="input-block">
                            <label>Self Employed</label>
                            <div class="input-shell">
                                <i class="fa-solid fa-briefcase"></i>
                                <select name="self_employed">
                                    <option value="0">No (Salaried)</option>
                                    <option value="1">Yes</option>
                                </select>
                            </div>
                        </div>

                        <div class="input-block">
                            <label>CIBIL Bureau Score (300 - 900)</label>
                            <div class="input-shell">
                                <i class="fa-solid fa-gauge-high"></i>
                                <input type="number" name="cibil_score" min="300" max="900" value="750" required>
                            </div>
                        </div>
                    </div>
                </div>

                <div>
                    <div class="field-group-title">
                        <i class="fa-solid fa-hand-holding-dollar"></i> Loan Terms & Income
                    </div>
                    <div class="grid-2">
                        <div class="input-block">
                            <label>Annual Gross Income (<span class="curr-sym">$</span>)</label>
                            <div class="input-shell">
                                <i class="fa-solid fa-money-bill-wave"></i>
                                <input type="number" name="income_annum" step="1000" value="6500000" required>
                            </div>
                        </div>

                        <div class="input-block">
                            <label>Requested Loan Principal (<span class="curr-sym">$</span>)</label>
                            <div class="input-shell">
                                <i class="fa-solid fa-file-invoice-dollar"></i>
                                <input type="number" name="loan_amount" step="1000" value="12000000" required>
                            </div>
                        </div>

                        <div class="input-block" style="grid-column: span 2;">
                            <label>Loan Duration (Years)</label>
                            <div class="input-shell">
                                <i class="fa-solid fa-clock"></i>
                                <input type="number" name="loan_term" min="1" max="30" value="10" required>
                            </div>
                        </div>
                    </div>
                </div>

                <div>
                    <div class="field-group-title">
                        <i class="fa-solid fa-building-columns"></i> Collateral Asset Base (<span class="curr-sym">$</span>)
                    </div>
                    <div class="grid-2">
                        <div class="input-block">
                            <label>Residential Real Estate</label>
                            <div class="input-shell">
                                <i class="fa-solid fa-house"></i>
                                <input type="number" name="residential_assets_value" value="4000000" required>
                            </div>
                        </div>

                        <div class="input-block">
                            <label>Commercial Real Estate</label>
                            <div class="input-shell">
                                <i class="fa-solid fa-city"></i>
                                <input type="number" name="commercial_assets_value" value="2500000" required>
                            </div>
                        </div>

                        <div class="input-block">
                            <label>Luxury Holdings</label>
                            <div class="input-shell">
                                <i class="fa-solid fa-gem"></i>
                                <input type="number" name="luxury_assets_value" value="8000000" required>
                            </div>
                        </div>

                        <div class="input-block">
                            <label>Bank Liquidity & Reserves</label>
                            <div class="input-shell">
                                <i class="fa-solid fa-piggy-bank"></i>
                                <input type="number" name="bank_asset_value" value="5000000" required>
                            </div>
                        </div>
                    </div>
                </div>

                <button type="submit" class="btn-evaluate" id="btnSubmit">
                    <i class="fa-solid fa-bolt"></i> Evaluate Loan Eligibility
                </button>
            </form>
        </section>

        <!-- Evaluation Results Terminal -->
        <section class="surface-card">
            <div class="card-header">
                <span class="card-heading">
                    <i class="fa-solid fa-chart-pie"></i> Risk Evaluation Dossier
                </span>
                <span class="tier-chip" id="tierChip">TIER: PENDING</span>
            </div>

            <div class="verdict-banner" id="verdictBanner">
                <div>
                    <div class="verdict-title-sub">Underwriting Status</div>
                    <div class="verdict-badge" id="verdictText">
                        <i class="fa-regular fa-hourglass-half"></i> Ready for Analysis
                    </div>
                </div>
                <div style="text-align: right;">
                    <div class="verdict-title-sub">Approval Confidence</div>
                    <div style="font-size: 1.5rem; font-weight: 800; font-family: 'JetBrains Mono', monospace; color: var(--accent-primary);" id="confScore">--%</div>
                </div>
            </div>

            <div class="scoreboard">
                <div class="score-tile">
                    <span class="score-label">Debt-to-Income</span>
                    <span class="score-val" id="metricDTI">0.00x</span>
                </div>
                <div class="score-tile">
                    <span class="score-label">Asset Coverage</span>
                    <span class="score-val" id="metricCoverage">0.00x</span>
                </div>
                <div class="score-tile">
                    <span class="score-label">Net Asset Base</span>
                    <span class="score-val" id="metricNetAssets"><span class="curr-sym">$</span>0</span>
                </div>
            </div>

            <div class="asset-stack-container">
                <div style="display: flex; justify-content: space-between; font-size: 0.75rem; color: var(--text-muted);">
                    <span>Collateral Portfolio Allocation</span>
                    <span id="assetSummaryLabel">0 assets</span>
                </div>

                <div class="stack-bar">
                    <div id="segRes" class="stack-segment seg-res" style="width: 25%;"></div>
                    <div id="segCom" class="stack-segment seg-com" style="width: 25%;"></div>
                    <div id="segLux" class="stack-segment seg-lux" style="width: 25%;"></div>
                    <div id="segBnk" class="stack-segment seg-bnk" style="width: 25%;"></div>
                </div>

                <div class="legend-grid">
                    <div class="legend-item">
                        <div class="legend-tag"><div class="dot seg-res"></div> Residential</div>
                        <span class="legend-amount" id="amtRes">$0</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-tag"><div class="dot seg-com"></div> Commercial</div>
                        <span class="legend-amount" id="amtCom">$0</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-tag"><div class="dot seg-lux"></div> Luxury</div>
                        <span class="legend-amount" id="amtLux">$0</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-tag"><div class="dot seg-bnk"></div> Bank Liquidity</div>
                        <span class="legend-amount" id="amtBnk">$0</span>
                    </div>
                </div>
            </div>

            <div class="audit-box">
                <div class="audit-header">
                    <i class="fa-solid fa-clipboard-check" style="color: var(--accent-primary);"></i> Decision Audit Remarks
                </div>
                <p id="policyNotes">
                    Run the evaluation form to generate the assessment. The underlying XGBoost gradient booster inspects repayment capacity, CIBIL health, and asset solvency ratios.
                </p>
            </div>
        </section>

    </main>

    <script>
        let cachedData = null;

        function setTheme(theme) {
            document.documentElement.setAttribute('data-theme', theme);
            document.querySelectorAll('.theme-dot').forEach(d => d.classList.remove('active'));
            const activeDot = document.querySelector(`.dot-${theme}`);
            if (activeDot) activeDot.classList.add('active');
        }

        function applyCurrencyChange() {
            const sym = document.getElementById('currencySelect').value;
            document.querySelectorAll('.curr-sym').forEach(el => el.innerText = sym);
            if (cachedData) renderResults(cachedData);
        }

        function formatCompact(num, sym) {
            if (num >= 10000000) return sym + (num / 10000000).toFixed(2) + ' Cr';
            if (num >= 1000000) return sym + (num / 1000000).toFixed(2) + ' M';
            if (num >= 1000) return sym + (num / 1000).toFixed(0) + ' K';
            return sym + num.toLocaleString();
        }

        function renderResults(res) {
            const sym = document.getElementById('currencySelect').value;
            const banner = document.getElementById('verdictBanner');
            const vText = document.getElementById('verdictText');
            const tierChip = document.getElementById('tierChip');
            const notes = document.getElementById('policyNotes');

            banner.classList.remove('approved', 'rejected');

            if (res.status === 'Approved') {
                banner.classList.add('approved');
                vText.innerHTML = '<i class="fa-solid fa-circle-check" style="color: #10b981;"></i> Loan Approved';
                tierChip.innerText = res.probability >= 0.85 ? 'TIER A (PRIME)' : 'TIER B (MODERATE)';
                tierChip.style.color = '#10b981';
                notes.innerText = `Facility cleared under standard lending policy rules. Applicant satisfies coverage thresholds with a Debt-to-Income factor of ${res.dti}x.`;
            } else {
                banner.classList.add('rejected');
                vText.innerHTML = '<i class="fa-solid fa-circle-xmark" style="color: #ef4444;"></i> Loan Rejected';
                tierChip.innerText = 'TIER C (HIGH RISK)';
                tierChip.style.color = '#ef4444';
                notes.innerText = `Applicant failed risk acceptance criteria. High leverage ratio (DTI ${res.dti}x) or adverse bureau score detected.`;
            }

            document.getElementById('confScore').innerText = Math.round(res.probability * 100) + '%';
            document.getElementById('metricDTI').innerText = res.dti + 'x';
            document.getElementById('metricCoverage').innerText = res.coverage + 'x';
            document.getElementById('metricNetAssets').innerText = formatCompact(res.total_assets, sym);

            const b = res.assets_breakdown;
            const tot = Math.max(res.total_assets, 1);

            document.getElementById('segRes').style.width = ((b.residential / tot) * 100) + '%';
            document.getElementById('segCom').style.width = ((b.commercial / tot) * 100) + '%';
            document.getElementById('segLux').style.width = ((b.luxury / tot) * 100) + '%';
            document.getElementById('segBnk').style.width = ((b.bank / tot) * 100) + '%';

            document.getElementById('amtRes').innerText = formatCompact(b.residential, sym);
            document.getElementById('amtCom').innerText = formatCompact(b.commercial, sym);
            document.getElementById('amtLux').innerText = formatCompact(b.luxury, sym);
            document.getElementById('amtBnk').innerText = formatCompact(b.bank, sym);
            document.getElementById('assetSummaryLabel').innerText = `Total: ${formatCompact(res.total_assets, sym)}`;
        }

        document.getElementById('loanForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            const btn = document.getElementById('btnSubmit');
            btn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Processing Evaluation...';
            btn.disabled = true;

            const formData = new FormData(e.target);

            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    body: formData
                });
                const data = await response.json();

                if (data.error) {
                    alert('Error: ' + data.error);
                } else {
                    cachedData = data;
                    renderResults(data);
                }
            } catch (err) {
                console.error(err);
                alert('Failed to connect to the prediction API.');
            } finally {
                btn.innerHTML = '<i class="fa-solid fa-bolt"></i> Evaluate Loan Eligibility';
                btn.disabled = false;
            }
        });
    </script>
</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

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
        dti = round(loan_amount / income, 2) if income > 0 else 0
        coverage = round(total_assets / loan_amount, 2) if loan_amount > 0 else 0

        if model is not None:
            input_array = np.array([features])
            raw_pred = int(model.predict(input_array)[0])
            
            # Binary logistic mapping: 0 corresponds to Approved in the model[cite: 1, 2]
            is_approved = (raw_pred == 0)

            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(input_array)[0]
                probability = float(probs[0]) if is_approved else float(probs[1])
            else:
                probability = 0.94 if is_approved else 0.12
        else:
            cibil = float(request.form.get('cibil_score', 300))
            is_approved = (cibil >= 650 and dti < 4.0)
            probability = 0.89 if is_approved else 0.21

        status = "Approved" if is_approved else "Rejected"

        return jsonify({
            'status': status,
            'probability': probability,
            'total_assets': total_assets,
            'dti': dti,
            'coverage': coverage,
            'assets_breakdown': {
                'residential': res_asset,
                'commercial': com_asset,
                'luxury': lux_asset,
                'bank': bnk_asset
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
