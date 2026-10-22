# 🎓 GridSathi: Mini-Project Viva & Technical Defense Guide

**Project Title:** GridSathi: Energy Forecasting & Smart Demand Scheduling  
**Scope:** AI-Powered Load Forecasting, Demand-Side Management (DSM), Bio-Inspired Metaheuristics, Industrial SCADA PLC Control & Autonomous Microgrid Resilience  
**Frameworks:** FastAPI, Python, XGBoost, Attention-LSTM + HHO, MO-DQN + SAPOA, HTML5 Glassmorphism UI  

---

## 📚 1. Grounded in 4 Top-Tier 2025 Research Papers

GridSathi is grounded in empirical research and published literature from IEEE and Nature Portfolio:

| # | Research Paper & Citation | Key Method | Mathematical Contribution | Exact Benchmarks Replicated |
|---|---|---|---|---|
| **1** | **Sriamad & Nangtin (IEEE iEECON 2025)**<br>*"Optimizing Energy Management and Reducing Peak Demand in Buildings using Forecasting Models, Machine Learning, Factorial Design, and Least Squares Method"* | $2 \times 2$ Full Factorial Design + Ordinary Least Squares (OLS) + SCADA PLC Relays | $$y = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \beta_3(X_1 X_2) + \epsilon$$ | **Run 1 (+1,+1):** 361.01 kWh<br>**Run 2 (+1,-1):** 325.19 kWh<br>**Run 3 (-1,+1):** 25.61 kWh<br>**Run 4 (-1,-1):** 10.26 kWh<br>$\beta_0 = 180.52, \beta_1 = 162.58, \beta_2 = 12.79, \beta_3 = 5.12$ |
| **2** | **Saroha et al. (Nature Scientific Reports 2025)**<br>*"Dynamic appliance scheduling and energy management in smart homes using adaptive reinforcement learning techniques"* | Multi-Objective Deep Q-Network (MO-DQN) + Self-Adaptive Puma Optimizer (SAPOA) | $$\text{Reward} = - [ w \cdot (\lambda_r \cdot E) + (1-w) \cdot \theta_m ]$$ | **PAR without RES:** 3.4286 ➔ 1.9765 (42.5% cut)<br>**PAR with RES:** 3.4286 ➔ 1.0339 (69.8% cut)<br>**Cost:** 92.92 ➔ 6.42 units (93.1% bill reduction) |
| **3** | **Johnsi Rani & Suresh (IEEE ICDISS 2025)**<br>*"Integration of Electric Vehicles with Smart Grids: Adaptive Control Mechanisms for Suitable and Sustainable EV Charging Governance"* | 4-Strategy Multi-Layer Adaptive EV Governance & Grid Stability Standards | Voltage Deviation, Frequency Jitter, Power Loss Reduction | **Peak Demand:** 1450 kW ➔ 1015 kW (30% reduction)<br>**Voltage Dev:** 7.5% ➔ 2.8% (62.7% boost)<br>**Frequency Jitter:** 0.12 Hz ➔ 0.04 Hz (66.6% boost)<br>**Power Loss:** 130 kW ➔ 75 kW (42.3% cut)<br>**Renewables:** 20% ➔ 42% (26% Solar + 16% Wind) |
| **4** | **Mnassri et al. (MDPI Energies 2025)**<br>*"An Intelligent Framework for Energy Forecasting and Management in Photovoltaic-Integrated Smart Homes in Tunisia with V2H Support Using LSTM Optimized by the Harris Hawks Algorithm"* | Microgrid Autonomy Dimensioning & 25-Year Life Cycle Economic NPV | $$DF = \frac{E_{\text{actual}}}{E_{\text{connected}}}, \quad B_{\text{lifetime}} = \sum_{y=1}^{25} \frac{E_y \cdot C_{el} - C_{O\&M} - C_{\text{rep}}}{(1+r)^y}$$ | **Connected Load:** 73.5 kWh ➔ **Diversified:** 42.5 kWh ($DF=0.58$)<br>**ROF:** 0.42 (Fridge: 1.5 kWh vs 3.6 kWh)<br>**BESS:** 59 kWh nominal / 42.5 kWh usable (80% DoD)<br>**Solar PV:** 10.7 kW (36 panels, 61.2 m² array, 75–90 m² roof) |

---

## 🌟 2. Detailed Technical Features & Demonstration Guide

### 🏭 Feature 1: Industrial SCADA & 2×2 Factorial PLC Load Control Studio (`/scada-factorial-studio`)
* **Located in Tab 2:** `🏭 2. SCADA 2×2 Factorial Studio (IEEE)`
* **Theoretical Foundation:** Sriamad & Nangtin (IEEE iEECON 2025).
* **What it Demonstrates:**
  1. Full $2 \times 2$ Factorial Design with 2 binary control factors:
     * **Type 1 ($X_1$):** Industrial Air Cool Chiller (ACH, 137,260 Btu)
     * **Type 2 ($X_2$):** Wall-Type Air Conditioners across classrooms
  2. Live Ordinary Least Squares (OLS) engine calculates regression coefficients:
     * $\beta_0 = 180.52$: Baseline intercept
     * $\beta_1 = 162.58$: Chiller main effect
     * $\beta_2 = 12.79$: Wall AC main effect
     * $\beta_3 = 5.12$: Compound interaction effect (super-additive when both run simultaneously!)
  3. Interactive 3D Isometric Response Surface visualization with glowing coordinate indicator.
  4. Virtual SCADA / PLC Relay Control Board with live LED indicator lights:
     * **Relay 1:** Air Cool Chiller Main Contactor (OPEN / CLOSED)
     * **Relay 2:** Air Handling Unit (AHU) VFD Inverter Speed %
     * **Relay 3:** Wall AC Relay Group (ENERGIZED / DE-ENERGIZED)
     * **Relay 4:** Chilled Water Circulation Pump (CIRCULATING / STANDBY)
  5. 6 PLC Operational States (Figure 4 of paper): Reducing load, Shifting load, Notification, Postpone hours, Power reduction (50% VFD limit), Emergency load tripping.

---

### 🐾 Feature 2: Nature MO-DQN & Self-Adaptive Puma Optimizer (SAPOA) (`/modqn-puma-scheduler`)
* **Located in Tab 3:** `🐾 3. Nature MO-DQN & Puma Optimizer`
* **Theoretical Foundation:** Saroha et al. (Nature Scientific Reports 2025).
* **What it Demonstrates:**
  1. Multi-Objective Reinforcement Learning (MO-DQN) balancing two conflicting objectives:
     * Minimizing Electricity Cost ($\lambda_r^{RTP} \cdot \text{Energy}$)
     * Minimizing User Dissatisfaction / Unhappiness ($\theta_m$)
  2. Interactive Pareto Trade-Off Slider ($w$):
     * Move $w \to 0.0$: Prioritizes User Comfort (zero appliance delay, full AC cooling).
     * Move $w \to 1.0$: Prioritizes Cost Savings (drastic peak clipping, load shifting).
  3. Appliance Taxonomy:
     * **Rigid:** Refrigerator (continuous operation with ROF = 0.42 duty cycle).
     * **Shiftable:** Washing Machine (shifts between $[r_{st}, r_{et}]$ according to RTP price).
     * **Controllable:** Air Conditioner (modulated dynamically within $[D_{\min}, D_{\max}]$).
  4. Live Peak-to-Average Ratio (PAR) Gauge:
     * Displays actual reduction from 3.4286 down to 1.9765 (grid-only) and 1.0339 (with Solar RES).
  5. Puma Hunting Phase Telemetry:
     * Explains transitions between Wide-Area Territory Sprint (Exploration) and Localized Ambush Leap (Exploitation) governed by a Bernoulli distribution.

---

### ⚡ Feature 3: Smart EV Charging Governance & Grid Stability Matrix (`/ev-governance-stability`)
* **Located in Tab 4:** `⚡ 4. EV Governance & Stability Matrix`
* **Theoretical Foundation:** Johnsi Rani & Suresh (IEEE ICDISS 2025).
* **What it Demonstrates:**
  1. Side-by-side comparison of 4 charging governance strategies:
     * **Uncontrolled:** Random charging, 1450 kW peak load, 7.5% voltage deviation, 0.12 Hz jitter.
     * **Time-of-Use (TOU):** Static off-peak incentives, 1220 kW peak load (15.9% reduction).
     * **Centralized:** Utility-managed dispatch, 1080 kW peak load (25.5% reduction).
     * **Adaptive Control (Proposed):** Multi-layered dynamic dispatch, 1015 kW peak load (30.0% reduction).
  2. Grid Stability Standards Radar:
     * **Voltage Deviation:** 7.5% ➔ 2.8% (62.7% improvement)
     * **Frequency Fluctuation:** 0.12 Hz ➔ 0.04 Hz (66.6% improvement)
     * **Power Losses:** 130 kW ➔ 75 kW (42.3% reduction)
  3. Renewable Energy Integration Breakdown:
     * Rises from 20% to 42% (26% Solar + 16% Wind).
  4. Driver Queueing & Quality of Experience:
     * Wait time drops from 32 min to 10 min; satisfaction climbs from 62% to 92%.
  5. V2G Peak Demand & Remuneration:
     * Injects 10,200 kWh into grid during peak hours, yielding ₹134 incentive per EV.

---

### 📐 Feature 4: Smart Home PV & Storage Autonomy Sizing Architect (`/smart-sizing-architect`)
* **Located in Tab 5:** `📐 5. PV & Storage Autonomy Sizing`
* **Theoretical Foundation:** Mnassri et al. (MDPI Energies 2025, Section 6.1 & 6.2).
* **What it Demonstrates:**
  1. **Diversity Factor ($DF = 0.58$):**
     * Converts theoretical maximum connected load ($73.5\text{ kWh/day}$) into effective diversified load ($42.5\text{ kWh/day}$).
  2. **Refrigerator Operating Factor ($ROF = 0.42$):**
     * Replaces unrealistic 24h continuous compressor operation ($3.6\text{ kWh}$) with actual duty cycle ($1.5\text{ kWh}$).
  3. **1-Day Autonomy BESS Sizing:**
     * Requires $42.5\text{ kWh}$ usable.
     * Nominal capacity dimensioned at: $\frac{42.5}{0.80 \times 0.90} = 59.0\text{ kWh}$ (accounting for 80% Depth of Discharge and 90% round-trip efficiency).
  4. **Solar PV Rooftop Array:**
     * $10.7\text{ kW}$ peak array ($36\text{ panels} \times 300\text{W}$) with $1.08$ overcast safety factor.
     * Solar module area: $61.2\text{ m}^2$; recommended usable rooftop space: $75 - 90\text{ m}^2$.
  5. **25-Year Life Cycle Feasibility ($B_{\text{lifetime}}$):**
     * Solves discounted NPV formula accounting for $0.5\%/\text{yr}$ PV degradation, yearly O&M, and full battery replacement at Year 12.

---

## 🎯 3. Top 10 Viva Examination Questions & Model Answers

### Q1: Why did you use Factorial Design of Experiments instead of testing one device at a time?
**Answer:**
> "Testing one factor at a time (OFAT) completely misses interaction effects. In smart buildings, when a 137k Btu Chiller and wall ACs operate simultaneously, the peak load is not merely additive—it creates super-additive voltage drop and thermal stress. The IEEE iEECON 2025 paper proved that the interaction term $\beta_3 = 5.12\text{ kWh}$ is statistically significant. Factorial design accurately captures main and interaction effects in only 4 experimental runs."

### Q2: What is Peak-to-Average Ratio (PAR) and why does reducing it from 3.43 to 1.03 matter?
**Answer:**
> "PAR is the ratio of instantaneous peak electrical demand to average consumption over a 24-hour cycle. When PAR is high (e.g. 3.43), utility grids must maintain expensive standby peaker generators (coal/gas) that sit idle most of the day, driving up tariffs and carbon intensity. Flattening PAR down to 1.03 creates an almost flat baseload profile, eliminating distribution transformer overload and slashing electricity generation costs by up to 93%."

### Q3: How does the Self-Adaptive Puma Optimizer (SAPOA) outperform Particle Swarm Optimization (PSO)?
**Answer:**
> "PSO often suffers from premature convergence in high-dimensional non-linear spaces because particles cluster around local minima. SAPOA introduces bio-inspired hunting mechanisms: it evaluates exploration scores ($S_{cexp}$) versus exploitation scores ($S_{cexl}$) with a Bernoulli distribution. It executes wide-area sprints when escaping energy is high ($E \ge 1$), and transitions to localized ambush leaps when $E < 1$, guaranteeing superior global convergence."

### Q4: Why is an EV charging load different from a typical residential load?
**Answer:**
> "An electric vehicle is a large, mobile, and transient load—a single Level 2 charger draws 3.3 to 7.4 kW, which equals or exceeds an entire home's peak demand. If uncoordinated, hundreds of EVs charging at 19:00 will trip local distribution transformers. However, EVs possess bidirectional onboard storage. By implementing V2H and V2G governance, EVs transform into mobile distributed energy banks that relieve grid strain."

### Q5: What is the significance of the Diversity Factor ($DF = 0.58$) and Refrigerator Operating Factor ($ROF = 0.42$)?
**Answer:**
> "If an engineer designs a residential microgrid by simply summing appliance nameplate ratings, the connected load is 73.5 kWh/day, necessitating a 140+ kWh battery which is economically prohibitive. In reality, appliances never run simultaneously at 100% capacity. Refrigerator compressors cycle on and off ($ROF = 0.42$). Applying $DF = 0.58$ dimensions the system realistically to 42.5 kWh/day with a 59 kWh battery, saving lakhs of rupees in unnecessary capital expenditure."

### Q6: How does your system protect the EV owner from waking up with a dead vehicle battery?
**Answer:**
> "Our hierarchical V2H/V2G dispatcher enforces a non-negotiable Mobility Reserve Threshold ($SOC > SOC_{\text{reserve}}$, typically 40%–50%). The EMS will never discharge the EV battery below this threshold, ensuring the vehicle's primary transportation function is unconditionally safeguarded."

### Q7: What machine learning algorithms are utilized in GridSathi?
**Answer:**
> "GridSathi utilizes a hybrid AI architecture:
> 1. **XGBoost Regressor:** Fast, high-accuracy baseline hourly load forecasting with 95% statistical confidence bounds ($\pm 1.96\sigma$).
> 2. **Attention-LSTM + HHO:** Deep recurrent sequence modeling optimized by Harris Hawks metaheuristic for one-hour-ahead forecasting (RMSE 269 Wh, $R^2 = 0.97$).
> 3. **Multi-Objective DQN (MO-DQN) + SAPOA:** Deep reinforcement learning for dynamic appliance scheduling.
> 4. **OLS Regression:** Factorial response surface modeling for industrial SCADA load control."

### Q8: How does GridSathi calculate marginal grid carbon emissions and 50Hz frequency droop?
**Answer:**
> "Under nominal conditions, the grid operates at 50.00 Hz with base emission intensity of ~480 gCO₂/kWh. During peak demand stress, dirty fossil peaker plants are dispatched, raising marginal carbon intensity up to 800 gCO₂/kWh and causing frequency to droop toward 49.78 Hz. When solar generation is high, clean renewable power displaces thermal generation, lowering carbon intensity to ~350 gCO₂/kWh."

### Q9: How does the 15-day Mid-Month Budget Sentinel help users avoid bill shocks?
**Answer:**
> "Based on Rao et al. (IEEE 2025), it logs actual consumption for Days 1–15 and computes the pro-rated bill with taxes and fixed meter charges. It then projects unconstrained usage for Days 16–30. If the total bill exceeds the user's budget (e.g. ₹1600), a Greedy Step-Down algorithm trims high-power flexible loads (AC, washing machine) by exact minutes per day to strictly hit the budget limit."

### Q10: How do you demonstrate that your system works in real time?
**Answer:**
> "We provide a live Cyber-Physical Energy Flow Visualizer powered by an HTML5 Canvas particle engine that animates real-time power packets flowing between the Sun, Solar PV, BESS Battery, Smart Home, Grid, and EV. Furthermore, our virtual SCADA PLC board exhibits live relay contactors opening and closing in response to simulated load conditions."

---

## 🚀 4. How to Launch and Demo GridSathi

1. **One-Click Startup:**
   Double-click `START_GRIDSATHI.bat` on the Desktop.
2. **Access URL:**
   Open Google Chrome or any browser at `http://127.0.0.1:8000/`.
3. **Trigger Presentation Mode:**
   Click the emerald button **🎓 Mini-Project Viva & Presentation Mode** at the top right of the dashboard.
4. **Use Quick Demo Presets:**
   Click any pill in the Quick Demos bar (e.g., `🏭 SCADA 2×2 Factorial PLC`, `🐾 MO-DQN Puma Optimizer`, `⚡ EV Grid Stability Matrix`, `📐 Smart Home PV+BESS Sizing`) to show live calculations instantly to the examiner.
