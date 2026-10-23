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
    print(f"[WARN] Model file '{MODEL_PATH}' not found. Using underwriting fallback engine.")
    return None

model = load_model()

FEATURE_KEYS = [
    'no_of_dependents', 'education', 'self_employed', 'income_annum',
    'loan_amount', 'loan_term', 'cibil_score', 'residential_assets_value',
    'commercial_assets_value', 'luxury_assets_value', 'bank_asset_value'
]

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Credix AI | Enterprise Credit Decisioning Platform</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.2/css/all.min.css">
    
    <style>
        :root {
            --slate-950: #020617;
            --slate-900: #0f172a;
            --slate-800: #1e293b;
            --slate-700: #334155;
            --slate-600: #475569;
            --slate-400: #94a3b8;
            --slate-200: #e2e8f0;
            --teal-400: #2dd4bf;
            --teal-500: #14b8a6;
            --emerald-400: #34d399;
            --emerald-500: #10b981;
            --rose-400: #fb7185;
            --rose-500: #f43f5e;
            --amber-400: #fbbf24;
            --indigo-500: #6366f1;
            --border-subtle: rgba(255, 255, 255, 0.08);
            --card-surface: rgba(15, 23, 42, 0.75);
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Plus Jakarta Sans', sans-serif;
        }

        body {
            background-color: var(--slate-950);
            color: #ffffff;
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            background-image: 
                radial-gradient(ellipse 80% 50% at 50% -20%, rgba(20, 184, 166, 0.15), transparent),
                radial-gradient(circle at 100% 100%, rgba(99, 102, 241, 0.08), transparent 30%);
        }

        /* Top Navigation Cockpit */
        .navbar {
            height: 70px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0 2rem;
            border-bottom: 1px solid var(--border-subtle);
            background: rgba(2, 6, 23, 0.85);
            backdrop-filter: blur(12px);
            position: sticky;
            top: 0;
            z-index: 50;
        }

        .brand-cluster {
            display: flex;
            align-items: center;
            gap: 0.85rem;
        }

        .brand-badge {
            width: 36px;
            height: 36px;
            border-radius: 10px;
            background: linear-gradient(135deg, var(--teal-400), var(--indigo-500));
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 800;
            color: var(--slate-950);
            box-shadow: 0 0 20px rgba(45, 212, 191, 0.35);
        }

        .brand-title {
            font-size: 1.15rem;
            font-weight: 800;
            letter-spacing: -0.02em;
        }

        .brand-subtitle {
            font-size: 0.7rem;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            color: var(--slate-400);
        }

        .nav-meta {
            display: flex;
            align-items: center;
            gap: 1.5rem;
        }

        .currency-switch {
            background: var(--slate-900);
            border: 1px solid var(--border-subtle);
            border-radius: 8px;
            color: var(--slate-200);
            padding: 0.4rem 0.75rem;
            font-size: 0.8rem;
            font-weight: 600;
            outline: none;
            cursor: pointer;
        }

        .status-pill {
            display: flex;
            align-items: center;
            gap: 0.45rem;
            font-size: 0.75rem;
            padding: 0.3rem 0.75rem;
            border-radius: 9999px;
            background: rgba(45, 212, 191, 0.1);
            color: var(--teal-400);
            border: 1px solid rgba(45, 212, 191, 0.25);
            font-family: 'JetBrains Mono', monospace;
        }

        .pulse-dot {
            width: 6px;
            height: 6px;
            background: var(--teal-400);
            border-radius: 50%;
            animation: pulse 2s infinite;
        }

        @keyframes pulse {
            0% { transform: scale(0.9); opacity: 0.6; }
            50% { transform: scale(1.3); opacity: 1; }
            100% { transform: scale(0.9); opacity: 0.6; }
        }

        /* Workspace Grid */
        .workspace {
            max-width: 1540px;
            width: 100%;
            margin: 0 auto;
            padding: 2rem;
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 2rem;
            flex: 1;
        }

        @media (max-width: 1120px) {
            .workspace {
                grid-template-columns: 1fr;
                padding: 1rem;
            }
        }

        .surface-card {
            background: var(--card-surface);
            border: 1px solid var(--border-subtle);
            border-radius: 16px;
            padding: 1.75rem;
            backdrop-filter: blur(16px);
            display: flex;
            flex-direction: column;
            gap: 1.5rem;
        }

        .card-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            border-bottom: 1px solid var(--border-subtle);
            padding-bottom: 1rem;
        }

        .card-heading {
            font-size: 1rem;
            font-weight: 700;
            display: flex;
            align-items: center;
            gap: 0.65rem;
            color: var(--slate-200);
        }

        .card-heading i {
            color: var(--teal-400);
        }

        /* Form Subsections */
        .field-group-title {
            font-size: 0.75rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: var(--teal-400);
            margin-bottom: 0.85rem;
            display: flex;
            align-items: center;
            gap: 0.4rem;
        }

        .grid-2 {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1rem;
        }

        @media (max-width: 600px) {
            .grid-2 {
                grid-template-columns: 1fr;
            }
        }

        .input-block {
            display: flex;
            flex-direction: column;
            gap: 0.4rem;
        }

        .input-block label {
            font-size: 0.75rem;
            font-weight: 600;
            color: var(--slate-400);
        }

        .input-shell {
            position: relative;
            display: flex;
            align-items: center;
        }

        .input-shell i {
            position: absolute;
            left: 0.85rem;
            color: var(--slate-600);
            font-size: 0.85rem;
            pointer-events: none;
            transition: color 0.2s;
        }

        .input-shell input, .input-shell select {
            width: 100%;
            background: var(--slate-900);
            border: 1px solid var(--border-subtle);
            border-radius: 10px;
            color: #ffffff;
            font-size: 0.85rem;
            padding: 0.65rem 0.85rem 0.65rem 2.4rem;
            outline: none;
            transition: border-color 0.2s, box-shadow 0.2s;
            font-family: 'JetBrains Mono', monospace;
        }

        .input-shell input:focus, .input-shell select:focus {
            border-color: var(--teal-400);
            box-shadow: 0 0 0 3px rgba(45, 212, 191, 0.15);
        }

        .input-shell input:focus + i {
            color: var(--teal-400);
        }

        .btn-evaluate {
            background: linear-gradient(135deg, var(--teal-400), var(--teal-500));
            color: var(--slate-950);
            font-size: 0.95rem;
            font-weight: 800;
            padding: 0.95rem 1.5rem;
            border-radius: 12px;
            border: none;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 0.6rem;
            transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1);
            margin-top: 0.5rem;
            box-shadow: 0 4px 20px rgba(45, 212, 191, 0.25);
        }

        .btn-evaluate:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(45, 212, 191, 0.35);
        }

        /* Decision Panel (Right) */
        .verdict-banner {
            border-radius: 14px;
            padding: 1.5rem;
            display: flex;
            align-items: center;
            justify-content: space-between;
            background: rgba(255, 255, 255, 0.02);
            border: 1px solid var(--border-subtle);
            transition: all 0.4s ease;
        }

        .verdict-banner.approved {
            background: linear-gradient(135deg, rgba(16, 185, 129, 0.12), rgba(15, 23, 42, 0.7));
            border-color: var(--emerald-400);
        }

        .verdict-banner.rejected {
            background: linear-gradient(135deg, rgba(244, 63, 94, 0.12), rgba(15, 23, 42, 0.7));
            border-color: var(--rose-400);
        }

        .verdict-title-sub {
            font-size: 0.7rem;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            color: var(--slate-400);
        }

        .verdict-badge {
            font-size: 1.5rem;
            font-weight: 800;
            letter-spacing: -0.02em;
            color: var(--slate-200);
            margin-top: 0.2rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .tier-chip {
            font-size: 0.7rem;
            font-family: 'JetBrains Mono', monospace;
            padding: 0.35rem 0.65rem;
            border-radius: 6px;
            background: var(--slate-800);
            border: 1px solid var(--border-subtle);
            color: var(--slate-200);
        }

        /* Underwriting Scoreboard */
        .scoreboard {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 1rem;
        }

        @media (max-width: 600px) {
            .scoreboard {
                grid-template-columns: 1fr;
            }
        }

        .score-tile {
            background: rgba(2, 6, 23, 0.6);
            border: 1px solid var(--border-subtle);
            border-radius: 12px;
            padding: 1rem;
            display: flex;
            flex-direction: column;
            gap: 0.25rem;
        }

        .score-label {
            font-size: 0.7rem;
            text-transform: uppercase;
            color: var(--slate-400);
            font-weight: 600;
        }

        .score-val {
            font-size: 1.25rem;
            font-weight: 700;
            font-family: 'JetBrains Mono', monospace;
            color: var(--teal-400);
        }

        /* Asset Breakdown Stacked Visualizer */
        .asset-stack-container {
            display: flex;
            flex-direction: column;
            gap: 0.75rem;
        }

        .stack-bar {
            height: 12px;
            width: 100%;
            background: var(--slate-900);
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
            background: rgba(2, 6, 23, 0.4);
            border: 1px solid var(--border-subtle);
        }

        .legend-tag {
            display: flex;
            align-items: center;
            gap: 0.45rem;
            color: var(--slate-400);
        }

        .dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
        }

        .legend-amount {
            font-family: 'JetBrains Mono', monospace;
            font-weight: 600;
        }

        /* Policy Narrative Log */
        .audit-box {
            background: var(--slate-900);
            border: 1px solid var(--border-subtle);
            border-radius: 12px;
            padding: 1rem;
            font-size: 0.75rem;
            color: var(--slate-400);
            line-height: 1.6;
        }

        .audit-header {
            font-weight: 700;
            color: var(--slate-200);
            margin-bottom: 0.35rem;
            display: flex;
            align-items: center;
            gap: 0.45rem;
        }
    </style>
</head>
<body>

    <!-- Cockpit Navigation -->
    <header class="navbar">
        <div class="brand-cluster">
            <div class="brand-badge">C</div>
            <div>
                <div class="brand-title">CREDIX <span style="color: var(--teal-400);">AI</span></div>
                <div class="brand-subtitle">Risk Underwriting Engine</div>
            </div>
        </div>

        <div class="nav-meta">
            <div class="status-pill">
                <div class="pulse-dot"></div>
                ENGINE ACTIVE v3.2
            </div>

            <select id="currencySelect" class="currency-switch" onchange="applyCurrencyChange()">
                <option value="$" data-rate="1">USD ($)</option>
                <option value="₹" data-rate="83.5">INR (₹)</option>
                <option value="€" data-rate="0.92">EUR (€)</option>
                <option value="£" data-rate="0.78">GBP (£)</option>
            </select>
        </div>
    </header>

    <!-- Operational Grid -->
    <main class="workspace">
        
        <!-- Left: Form Input Terminal -->
        <section class="surface-card">
            <div class="card-header">
                <span class="card-heading">
                    <i class="fa-solid fa-file-invoice-dollar"></i> Applicant Parameters
                </span>
                <span style="font-size: 0.7rem; color: var(--slate-400);">11-Point Verification</span>
            </div>

            <form id="loanForm" style="display: flex; flex-direction: column; gap: 1.25rem;">
                
                <!-- Primary Profile -->
                <div>
                    <div class="field-group-title">
                        <i class="fa-solid fa-user-shield"></i> Demographics & Career
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
                            <label>Education Level</label>
                            <div class="input-shell">
                                <i class="fa-solid fa-graduation-cap"></i>
                                <select name="education">
                                    <option value="1">Graduate Degree</option>
                                    <option value="0">Non-Graduate</option>
                                </select>
                            </div>
                        </div>

                        <div class="input-block">
                            <label>Employment Structure</label>
                            <div class="input-shell">
                                <i class="fa-solid fa-briefcase"></i>
                                <select name="self_employed">
                                    <option value="0">Salaried Professional</option>
                                    <option value="1">Self Employed / Business</option>
                                </select>
                            </div>
                        </div>

                        <div class="input-block">
                            <label>Credit Bureau Score (CIBIL)</label>
                            <div class="input-shell">
                                <i class="fa-solid fa-gauge-high"></i>
                                <input type="number" name="cibil_score" min="300" max="900" value="745" required>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Loan Facility Terms -->
                <div>
                    <div class="field-group-title">
                        <i class="fa-solid fa-landmark"></i> Loan Request Facility
                    </div>
                    <div class="grid-2">
                        <div class="input-block">
                            <label>Annual Declared Income (<span class="curr-sym">$</span>)</label>
                            <div class="input-shell">
                                <i class="fa-solid fa-money-bill-trend-up"></i>
                                <input type="number" name="income_annum" step="1000" value="6500000" required>
                            </div>
                        </div>

                        <div class="input-block">
                            <label>Requested Loan Principal (<span class="curr-sym">$</span>)</label>
                            <div class="input-shell">
                                <i class="fa-solid fa-hand-holding-dollar"></i>
                                <input type="number" name="loan_amount" step="1000" value="12000000" required>
                            </div>
                        </div>

                        <div class="input-block" style="grid-column: span 2;">
                            <label>Tenure Period (Years)</label>
                            <div class="input-shell">
                                <i class="fa-solid fa-clock-rotate-left"></i>
                                <input type="number" name="loan_term" min="1" max="30" value="10" required>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Collateral Portfolio -->
                <div>
                    <div class="field-group-title">
                        <i class="fa-solid fa-vault"></i> Declared Asset Portfolio (<span class="curr-sym">$</span>)
                    </div>
                    <div class="grid-2">
                        <div class="input-block">
                            <label>Residential Real Estate</label>
                            <div class="input-shell">
                                <i class="fa-solid fa-house"></i>
                                <input type="number" name="residential_assets_value" value="4500000" required>
                            </div>
                        </div>

                        <div class="input-block">
                            <label>Commercial Properties</label>
                            <div class="input-shell">
                                <i class="fa-solid fa-city"></i>
                                <input type="number" name="commercial_assets_value" value="3000000" required>
                            </div>
                        </div>

                        <div class="input-block">
                            <label>Luxury Holdings</label>
                            <div class="input-shell">
                                <i class="fa-solid fa-gem"></i>
                                <input type="number" name="luxury_assets_value" value="7000000" required>
                            </div>
                        </div>

                        <div class="input-block">
                            <label>Bank Liquidity & Deposits</label>
                            <div class="input-shell">
                                <i class="fa-solid fa-piggy-bank"></i>
                                <input type="number" name="bank_asset_value" value="5500000" required>
                            </div>
                        </div>
                    </div>
                </div>

                <button type="submit" class="btn-evaluate" id="btnSubmit">
                    <i class="fa-solid fa-shield-halved"></i> Run Instant Underwriting Evaluation
                </button>
            </form>
        </section>

        <!-- Right: Executive Risk Cockpit -->
        <section class="surface-card">
            <div class="card-header">
                <span class="card-heading">
                    <i class="fa-solid fa-chart-pie"></i> Risk Evaluation Dossier
                </span>
                <span class="tier-chip" id="tierChip">TIER: PENDING</span>
            </div>

            <!-- Approval Banner -->
            <div class="verdict-banner" id="verdictBanner">
                <div>
                    <div class="verdict-title-sub">Underwriting Status</div>
                    <div class="verdict-badge" id="verdictText">
                        <i class="fa-regular fa-hourglass-half"></i> Ready to Assess
                    </div>
                </div>
                <div style="text-align: right;">
                    <div class="verdict-title-sub">Approval Confidence</div>
                    <div style="font-size: 1.5rem; font-weight: 800; font-family: 'JetBrains Mono', monospace; color: var(--teal-400);" id="confScore">--%</div>
                </div>
            </div>

            <!-- Scoreboard -->
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

            <!-- Collateral Mix -->
            <div class="asset-stack-container">
                <div style="display: flex; justify-content: space-between; font-size: 0.75rem; color: var(--slate-400);">
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
                        <div class="legend-tag"><div class="dot segCom"></div> Commercial</div>
                        <span class="legend-amount" id="amtCom">$0</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-tag"><div class="dot seg-lux"></div> Luxury</div>
                        <span class="legend-amount" id="amtLux">$0</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-tag"><div class="dot seg-bnk"></div> Bank Reserves</div>
                        <span class="legend-amount" id="amtBnk">$0</span>
                    </div>
                </div>
            </div>

            <!-- System Narrative -->
            <div class="audit-box">
                <div class="audit-header">
                    <i class="fa-solid fa-clipboard-check" style="color: var(--teal-400);"></i> Policy Decision Remarks
                </div>
                <p id="policyNotes">
                    Run an underwriting assessment using the form on the left. The engine analyzes CIBIL parameters, repayment leverage, and collateral stability to render an immediate institutional decision.
                </p>
            </div>
        </section>

    </main>

    <script>
        let cachedData = null;

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
                vText.innerHTML = '<i class="fa-solid fa-circle-check" style="color: var(--emerald-400);"></i> Approved';
                tierChip.innerText = res.probability >= 0.85 ? 'TIER A (PRIME)' : 'TIER B (MODERATE)';
                tierChip.style.color = 'var(--emerald-400)';
                notes.innerText = `Facility cleared under standard policy rules. Collateral covers the credit requirement adequately with a Debt-to-Income factor of ${res.dti}x.`;
            } else {
                banner.classList.add('rejected');
                vText.innerHTML = '<i class="fa-solid fa-circle-xmark" style="color: var(--rose-400);"></i> Declined';
                tierChip.innerText = 'TIER C (HIGH RISK)';
                tierChip.style.color = 'var(--rose-400)';
                notes.innerText = `Credit profile failed policy threshold criteria. Elevated leverage risk (DTI ${res.dti}x) or adverse score parameters detected. Manual supervisor override required.`;
            }

            document.getElementById('confScore').innerText = Math.round(res.probability * 100) + '%';
            document.getElementById('metricDTI').innerText = res.dti + 'x';
            document.getElementById('metricCoverage').innerText = res.coverage + 'x';
            document.getElementById('metricNetAssets').innerText = formatCompact(res.total_assets, sym);

            // Asset distribution segments
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
            btn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Processing Underwriting Query...';
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
                alert('Connection failure contacting prediction endpoint.');
            } finally {
                btn.innerHTML = '<i class="fa-solid fa-shield-halved"></i> Run Instant Underwriting Evaluation';
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
            
            # 0 corresponds to Approved in the model tree schema
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
