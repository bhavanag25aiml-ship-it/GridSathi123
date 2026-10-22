# ⚡ GridSathi: Energy Forecasting & Smart Demand Scheduling

> **An Advanced AI-Driven Platform for Real-Time Load Forecasting, Multi-Appliance Demand-Side Scheduling, and Autonomous Microgrid Resilience.**

---

## 📚 Grounded in Cutting-Edge 2025 Research

GridSathi 2.0 directly operationalizes empirical models, control logic, and optimization algorithms published in top 2025 IEEE and MDPI smart grid literature:

| Research Paper | Key Theoretical Contribution | Implemented in GridSathi 2.0 |
|---|---|---|
| **Mnassri et al. (Energies 2025, MDPI)**<br>*"An Intelligent Framework for Energy Forecasting and Management in Photovoltaic-Integrated Smart Homes with V2H Support Using LSTM Optimized by the Harris Hawks Algorithm"* | • Attention-LSTM tuned via Harris Hawks Optimization ($E = 2E_0(1 - t/T)$), reducing RMSE from 944.9 Wh to 269.5 Wh (MAPE 9.43%, $R^2 = 0.97$).<br>• Hierarchical dual-storage dispatch: Home Battery handles daily cycles, while EV (V2H) covers deep deficits subject to strict mobility reserve ($SOC > SOC_{\text{reserve}}$).<br>• Microgrid Sizing Engine: Connected Demand 73.5 kWh ➔ Diversified Demand 42.5 kWh ($DF = 0.58$), Refrigerator ROF = 0.42, 1-Day Autonomy BESS (59 kWh nominal / 42.5 kWh usable), 10.7 kW PV (36 panels, 75-90 m² roof), and 25-Year Net Economic Benefit ($B_{\text{lifetime}}$). | • **`/v2h-v2g-dispatcher`**: Hierarchical dual-battery dispatch, mobility reserve slider, PV self-consumption (+18.6%), grid reliance drop (-25%), cost savings (-17.3%).<br>• **`/hho-optimization-lab`**: Interactive bio-inspired HHO simulation comparing LSTM vs PSO vs HHO.<br>• **`/smart-sizing-architect`**: Interactive PV + Storage dimensioning, rooftop area allocator, and 25-year life cycle economic analyzer. |
| **Johnsi Rani & Suresh (ICDISS 2025, IEEE)**<br>*"Integration of Electric Vehicles with Smart Grids: Adaptive Control Mechanisms for Suitable and Sustainable EV Charging Governance"* | • Multi-layered adaptive control for dynamic EV charging/discharging.<br>• Quantitative grid resilience standards under EV penetration:<br>  - Voltage Deviation: 7.5% ➔ 2.8% (62.7% improvement)<br>  - Frequency Fluctuation: 0.12 Hz ➔ 0.04 Hz (66.6% improvement)<br>  - Line Power Loss: 130 kW ➔ 75 kW (42.3% reduction)<br>  - Peak Load Reduction: 30.0% (1450 kW ➔ 1015 kW)<br>• Renewable Energy penetration boost: 20% ➔ 42% (Solar + Wind).<br>• Driver queueing wait time cut from 32 min to 10 min (92% satisfaction).<br>• Dynamic V2G Feed-in Remuneration (₹85–₹134 per EV). | • **`/ev-governance-stability`**: 4-Strategy EV governance comparison (Uncontrolled vs TOU vs Centralized vs Adaptive), IEEE stability radar, queueing wait times, and V2G incentive payouts. |
| **Sriamad & Nangtin (iEECON 2025, IEEE)**<br>*"Optimizing Energy Management and Reducing Peak Demand in Buildings using Forecasting Models, Machine Learning, Factorial Design, and Least Squares Method"* | • Full $2 \times 2$ Factorial Design (Chiller Type 1 × Wall-Type AC Type 2).<br>• Empirical Ordinary Least Squares (OLS) model:<br>  $$y = \beta_0 + \beta_1(\text{Type 1}) + \beta_2(\text{Type 2}) + \beta_3(\text{Type 1} \times \text{Type 2}) + \epsilon$$<br>  ($\beta_0=180.52, \beta_1=162.58, \beta_2=12.79, \beta_3=5.12$).<br>• SCADA PLC 6-state load controller (reducing load, shifting, postpone, VFD power reduction, stopping load) with TOU pricing. | • **`/scada-factorial-studio`**: 3D Isometric Response Surface visualization, virtual SCADA / PLC hardware relay digital twin, 4-run factorial experimental table, and TOU shifting savings. |
| **Saroha et al. (Nature Scientific Reports 2025)**<br>*"Dynamic appliance scheduling and energy management in smart homes using adaptive reinforcement learning techniques"* | • Multi-Objective Deep Q-Network (MO-DQN) + Self-Adaptive Puma Optimizer (SAPOA).<br>• Rigorous appliance taxonomy: Rigid (continuous Refrigerator, ROF=0.42), Shiftable (Washing Machine window $[r_{st}, r_{et}]$ with delay penalty $\theta^2_m$), and Controllable (AC modulated $[D_{\min}, D_{\max}]$ with unhappiness $\theta^3_m$).<br>• Slashes Peak-to-Average Ratio (PAR) from 3.4286 down to 1.9765 without RES and 1.0339 with RES (69.8% reduction!). Cuts electricity cost from 92.92 to 6.42 units.<br>• Dual-objective Pareto tradeoff slider: $\text{Reward} = -[w \cdot (\text{Tariff} \cdot \text{Energy}) + (1-w) \cdot \text{Dissatisfaction}]$. | • **`/modqn-puma-scheduler`**: Live MO-DQN + SAPOA scheduler, interactive Cost vs Comfort ($w$) slider, live PAR gauge, 5-algorithm benchmark table, and Puma phase telemetry. |
| **Rao et al. (ICETI4T 2025, IEEE)**<br>*"Automated Energy Management for Sustainable and Cost-Efficient Households: A Forecasting and Optimization Approach"* | • Comprehensive 3-part billing formula:<br>  $$\text{Bill} = (\text{Units} \times \text{Rate}) + \text{Fixed Charge} + \text{Tax (9\%)}$$<br>• Two-stage mid-month budget optimization: Logs Days 1–15, estimates unconstrained Days 16–30; if bill > budget limit (e.g. ₹1600), applies **Greedy Step-Down Algorithm** to calculate exact daily minute cuts on high-power appliances (AC, Washer, Kitchen). | • **`/mid-month-budget-optimizer`**: Two-phase 15-day budget sentinel with greedy appliance minute reduction recommendations ensuring bills never exceed budget limits. |
| **Mishra, Verma & Parikh (IETACS 2025, IEEE)**<br>*"Application of AI in the Load Forecasting for Smart Grid"* | • Multi-horizon forecasting taxonomy: STLF (Short-Term, 24h), MTLF (Medium-Term, 4 weeks), and LTLF (Long-Term, 12 months).<br>• Quantified financial penalty of forecasting errors: **"A 1% load forecasting deviation incurs a financial loss of $10 million for large utility grids"** due to spinning reserve over-procurement and imbalance penalties. | • **`/multi-horizon-forecasting`**: Multi-horizon engine (STLF/MTLF/LTLF) + Interactive 1% Error Financial Risk & Avoidable Reserve Penalty Meter. |
| **Goswami, Roy & Mandal (ICPEE 2025, IEEE)**<br>*"A Comprehensive Review of Machine Learning Techniques for Enhancing Energy Efficiency in Renewable Integrated Smart Grids"* | • Survey of ML across forecasting, Demand Response (DR), and cybersecurity.<br>• Unsupervised clustering/autoencoders for electricity theft and phantom loads.<br>• Dynamic 50Hz frequency droop, marginal carbon intensity, and Virtual Power Plant (VPP) negawatt aggregation. | • **`/carbon-frequency`**: 50Hz droop & marginal emissions ($gCO_2/kWh$).<br>• **`/anomaly-detector`**: Residual Z-score sentinel for vampire standby power and meter tampering.<br>• **`/grid-stress-sim`**: VPP blackout avoidance and negawatt reward calculator. |

---

## 🌟 Complete Feature Matrix (17 Interactive Dashboard Tabs)

1. 📊 **AI Forecasting & Peaks**: Single-hour prediction with $\pm 1.96\sigma$ confidence bounds, 24-hr load curve, ToD tariff tiers, and Explainable AI (XAI) Gini feature weights.
2. 🏭 **Industrial SCADA & 2×2 Factorial PLC Studio**: Full 2×2 Factorial Design (Chiller × AC), 3D Isometric Response Surface, OLS regression equation, and 6-state PLC relay hardware digital twin (IEEE iEECON 2025).
3. 🐾 **Nature MO-DQN & Puma Optimizer (SAPOA)**: Multi-objective reinforcement learning balancing cost vs comfort ($), live PAR reduction gauge (3.43 ➔ 1.03), 5-algorithm benchmark, and Puma ambush/sprint telemetry (Nature Scientific Reports 2025).
4. ⚡ **EV Charging Governance & Grid Stability Matrix**: 4-Strategy comparison (Uncontrolled, TOU, Centralized, Adaptive), 62.7% voltage deviation cut, 66.6% frequency jitter cut, 42% renewable integration, and V2G incentive payouts (IEEE ICDISS 2025).
5. 📐 **Smart Home PV & Storage Autonomy Sizing Architect**: Dimensioning calculator applying Diversity Factor ( = 0.58$), Refrigerator ROF (0.42), 1-day autonomy battery sizing (59 kWh nominal / 42.5 kWh usable), and 25-year life cycle economic NPV ({\text{lifetime}}$) (MDPI Energies 2025).
6. 📅 **Multi-Horizon Forecasting & 1% Error Cost Meter**: STLF (24h), MTLF (4 weeks), and LTLF (12 months) forecasting with utility financial risk calculation.
7. 🦅 **Harris Hawks Optimization (HHO) Bio-Inspired AI Lab**: Simulates cooperative hunting search dynamics ( = 2E_0(1 - t/T)$), comparing Baseline LSTM vs LSTM+PSO vs LSTM+HHO (71.5% error reduction!).
8. ⏰ **Multi-Appliance Pareto Fleet Optimizer**: Concurrently schedules EV, Heat Pump, Water Heater, and Washer under a hard Sanctioned Maximum Demand breaker limit with interactive Gantt timeline.
9. 💰 **15-Day Mid-Month Budget Sentinel**: Evaluates Days 1–15 actuals, projects Days 16–30, and executes Greedy Step-Down optimization to trim high-power appliance minutes to hit target budgets (e.g., ₹1600).
10. 🕒 **Grid Time Machine**: Before vs after appliance timing shift peak reduction comparison with interactive timeline graphs.
11. 🚗 **Bidirectional V2H & V2G Dispatcher**: Hierarchical dual-battery dispatch, non-negotiable mobility reserve slider ( > 40\%$), V2G peak grid feed remuneration, and IEEE grid stability metrics.
12. ☀️ **Solar Rooftop + BESS Storage Digital Twin**: Irradiance curve with temperature derating, battery charge/discharge SOC bounds, and Net Metering export credits.
13. 🌿 **Autonomous Grid Carbon & 50Hz Droop Engine**: Dynamic grid frequency droop and hourly marginal emissions (/kWh$), identifying clean renewable absorption windows.
14. 🛡️ **AI Anomaly & Phantom Vampire Load Sentinel**: Residual Z-score audit detecting nighttime standby leaks (01:00–05:00) and meter bypass with annual loss in ₹.
15. 🌪️ **Virtual Power Plant (VPP) Blackout Stress-Tester**: Simulates extreme heatwaves (+6°C surge) and transformer overloads; demonstrates how AI Demand Response negawatts avert blackouts.
16. 📋 **Certified Household Energy Health Audit**: Professional scorecard calculating IEEE Load Factor (LF), Peak-to-Average Ratio (PAR), and printable ESG certificate.
17. 💬 **Conversational Sathi Voice & Text AI Copilot**: Natural language and hands-free voice control powered by Web Speech Recognition and Speech Synthesis.

---

## 🛠️ Mathematical Formulations from the Research Papers

### 1. Thermal & Humidity PV Generation Model (Mnassri et al. 2025, Eq. 30)
$$P_{PV}(t) = P_{\max} \cdot \left[1 - \alpha (T(t) - 25) - \beta \cdot RH(t)\right] \cdot \sin\left(\pi \frac{t - 6}{12}\right)$$
- $\alpha = 0.004$ (temperature derating coefficient)
- $\beta = 0.001$ (relative humidity atmospheric attenuation)

### 2. Microclimate Household Cooling/Heating Load (Mnassri et al. 2025, Eq. 31)
$$P_{\text{load}}(t) = \gamma_0 + \gamma_1 \max(0, T(t) - T_{\text{base}}) + \gamma_2 \max(0, T_{\text{base}} - T(t)) + \gamma_3 \cdot RH(t)$$

### 3. Harris Hawks Optimization Escaping Energy (Mnassri et al. 2025, Eq. 20–23)
$$E = 2E_0 \left(1 - \frac{t}{T}\right)$$
- $|E| \ge 1$: **Exploration Phase** (global search)
- $|E| < 1, r \ge 0.5$: **Soft Besiege** ($X(t+1) = \Delta X(t) - E |J X_{\text{best}} - X(t)|$)
- $|E| < 1, r < 0.5$: **Hard Besiege** ($X(t+1) = X_{\text{best}} - E |\Delta X(t)|$)

### 4. Electricity Bill with Fixed Charges & Taxes (Rao et al. 2025, Eq. 1–3)
$$\text{Bill} = \Big(\text{Total Units} \times \text{Cost per Unit}\Big) + \text{Fixed Charges} + \text{Tax (9\%)}$$

### 5. Utility Financial Risk per 1% Forecasting Deviation (Mishra et al. 2025)
$$\text{Loss}_{\text{annual}} = \left(\frac{\text{Grid Capacity (MW)}}{5000\text{ MW}}\right) \times \$10,000,000 \times \text{Error}_{\%}$$

---

## 🚀 Running the Application

### 1. Start the Server
```bash
.\.venv\Scripts\python.exe backend/main.py
```
*(Or `.\.venv\Scripts\python.exe -m uvicorn backend.app:app --host 127.0.0.1 --port 8000`)*

### 2. Access the Interactive Dashboard
Open your browser at:
```
http://127.0.0.1:8000/
```

---

## 📡 Complete REST API Endpoints

| Endpoint | Method | Research Citation | Description |
|---|---|---|---|
| `/` | `GET` | — | Serves interactive UI (browser) or JSON health check (API) |
| `/api/status` | `GET` | — | System health, XGBoost model status & uncertainty metrics |
| `/predict` | `POST` | Mishra et al. 2025 | Single-hour prediction with 95% Confidence Bounds ($\pm 1.96\sigma$) |
| `/peak-hours` | `POST` | Goswami et al. 2025 | 24-hr chronological forecast with peak/off-peak classification |
| `/v2h-v2g-dispatcher` | `POST` | Mnassri et al. & Johnsi Rani et al. 2025 | Hierarchical dual-battery V2H/V2G adaptive dispatch & IEEE stability telemetry |
| `/mid-month-budget-optimizer` | `POST` | Rao et al. 2025 | 15-day actuals logging, Days 16–30 projection & Greedy Step-Down trimmer |
| `/hho-optimization-lab` | `POST` | Mnassri et al. 2025 | Harris Hawks Optimization simulation with LSTM vs PSO vs HHO benchmarks |
| `/multi-horizon-forecasting` | `POST` | Mishra et al. 2025 | STLF, MTLF, LTLF load forecasts and 1% error financial loss calculator |
| `/schedule` | `POST` | Goswami et al. 2025 | Single appliance scheduler with sanctioned load limit guard |
| `/multi-schedule` | `POST` | Goswami et al. 2025 | Multi-appliance concurrent fleet optimizer with Gantt dispatch |
| `/solar-battery-twin` | `POST` | Mnassri et al. 2025 | Solar PV generation, battery SOC curves, and Net Metering export credits |
| `/anomaly-detector` | `POST` | Goswami et al. 2025 | Residual Z-score detection for vampire leaks and meter tampering |
| `/grid-stress-sim` | `POST` | Goswami et al. 2025 | Virtual Power Plant (VPP) extreme weather and blackout resilience simulator |
| `/energy-audit` | `GET` | — | Household Load Factor, PAR, and A+ to F energy health scorecard |
| `/copilot` | `POST` | — | Natural language & voice assistant intent parser |
| `/explain-ai` | `GET` | — | XGBoost Gini feature importances and domain explanations |
| `/time-machine` | `POST` | — | Before vs after appliance timing shift peak reduction comparison |
| `/scada-factorial-studio` | `POST` | Sriamad & Nangtin 2025 | SCADA 2×2 Factorial DOE, 3D Response Surface & 6-State PLC Relay Digital Twin |
| `/modqn-puma-scheduler` | `POST` | Saroha et al. Nature 2025 | Multi-Objective DQN + Puma Optimizer (SAPOA), PAR Slashed (3.43 ➔ 1.03) |
| `/ev-governance-stability` | `POST` | Johnsi Rani & Suresh 2025 | 4-Strategy EV Governance Matrix, Grid Jitter Suppression & V2G Incentives |
| `/smart-sizing-architect` | `POST` | Mnassri et al. 2025 | Smart Home PV & BESS Autonomy Dimensioning (DF=0.58, ROF=0.42, 25-yr NPV) |

---

## 🏆 Project Structure
```
Grid Sathi/
│
├── backend/
│   ├── app.py             # FastAPI backend with 16 research-backed endpoints
│   └── main.py            # Resilient entrypoint launcher script
│
├── frontend/
│   └── index.html         # Modern cyber-physical glassmorphic UI + Canvas particle visualizer
│
├── models/
│   └── energy_model.pkl   # Trained XGBoost regression model
│
├── data/
│   ├── training_electricity.csv
│   ├── processed_electricity.csv
│   └── electricity.csv
│
├── train_model.py         # XGBoost training pipeline
├── generate_dataset.py    # Synthetic weather & consumption generator
├── prepare_data.py        # Data preprocessing pipeline
└── README.md              # Complete project documentation with research paper citations
```
