import streamlit as st
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Draw
from sklearn.ensemble import RandomForestRegressor

# --- PAGE CONFIG ---
st.set_page_config(page_title="PolySights | Polymer Analytical Model", layout="wide")

# --- MODERN LABORATORY CSS with VIOLET-WHITE GRADIENT ---
st.markdown("""
    <style>
    /* Global Styles with Gradient Background */
    .stApp {
        background: linear-gradient(135deg, #7c3aed 0%, #c4b5fd 50%, #ffffff 100%);
        min-height: 100vh;
    }
    
    /* Global Text Color for Visibility */
    p, span, label, .stMarkdown {
        color: #1e293b !important;
        font-weight: 500;
    }

    /* Header Banner */
    .header-box {
        background: linear-gradient(135deg, #7c3aed 0%, #a78bfa 100%);
        padding: 50px;
        border-radius: 15px;
        color: #ffffff !important;
        text-align: center;
        margin-bottom: 40px;
        box-shadow: 0 20px 40px rgba(124, 58, 237, 0.3);
    }
    .header-box h1 {
        color: #ffffff !important;
        font-size: 3.5rem !important;
        margin-bottom: 10px;
        font-weight: 900 !important;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
    }
    .header-box p {
        color: #e0e7ff !important;
        font-size: 1.2rem;
    }

    /* Remove Sidebar */
    [data-testid="stSidebar"] {
        display: none;
    }

    /* Section Cards */
    .info-card {
        background: linear-gradient(135deg, rgba(255,255,255,0.95), rgba(250,245,255,0.95));
        border: 2px solid rgba(124, 58, 237, 0.2);
        border-radius: 12px;
        padding: 25px;
        height: 100%;
        box-shadow: 0 8px 16px rgba(124, 58, 237, 0.15);
        transition: all 0.3s ease;
    }
    
    .info-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 24px rgba(124, 58, 237, 0.25);
        border-color: rgba(124, 58, 237, 0.4);
    }

    /* Results Highlight */
    .results-container {
        background: linear-gradient(135deg, rgba(250,245,255,0.8), rgba(240,230,255,0.8));
        border-left: 6px solid #7c3aed;
        padding: 25px;
        margin: 20px 0;
        border-radius: 10px;
        box-shadow: 0 8px 16px rgba(124, 58, 237, 0.1);
    }

    /* Interpretation Box */
    .interpretation-box {
        background: linear-gradient(135deg, rgba(250,245,255,0.9), rgba(240,230,255,0.9));
        border-left: 6px solid #7c3aed;
        padding: 30px;
        margin: 20px 0;
        border-radius: 12px;
        box-shadow: 0 10px 20px rgba(124, 58, 237, 0.15);
    }

    /* Styled Section Titles */
    .section-header {
        color: #7c3aed;
        font-size: 1.5rem;
        font-weight: 700;
        border-bottom: 3px solid #c4b5fd;
        padding-bottom: 10px;
        margin-bottom: 25px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* Quick Library Buttons */
    .quick-btn {
        background: linear-gradient(135deg, #7c3aed, #a78bfa) !important;
        color: white !important;
        font-weight: 700 !important;
        border-radius: 8px !important;
        border: none !important;
        height: 45px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 8px rgba(124, 58, 237, 0.2);
    }

    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #7c3aed, #a78bfa) !important;
        color: white !important;
        font-weight: 700 !important;
        border-radius: 8px !important;
        border: none !important;
        height: 50px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(124, 58, 237, 0.3);
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #6d28d9, #9f7aea) !important;
        transform: translateY(-3px);
        box-shadow: 0 8px 16px rgba(124, 58, 237, 0.4);
    }

    /* Metric Styling */
    [data-testid="stMetricValue"] {
        color: #7c3aed !important;
        font-weight: 800 !important;
        font-size: 2rem !important;
    }
    
    [data-testid="stMetricLabel"] {
        color: #6b21a8 !important;
        font-weight: 700 !important;
    }

    /* Application List Styling */
    .app-item {
        color: #4c1d95;
        font-size: 1.05rem;
        margin: 10px 0;
        padding-left: 25px;
        line-height: 1.8;
        font-weight: 500;
    }
    </style>
    """, unsafe_allow_html=True)

# --- ML MODEL ENGINE ---
@st.cache_resource
def train_analytical_model():
    # Synthetic dataset mimicking Polymer property correlations
    X = np.random.rand(200, 4) * [600, 6, 12, 120]
    y_tg = (X[:, 0] * 0.25) + (X[:, 1] * 15) + 140 + np.random.normal(0, 5, 200)
    y_rho = 0.75 + (X[:, 1] * 0.06) + np.random.normal(0, 0.02, 200)
    
    tg_model = RandomForestRegressor(n_estimators=100, random_state=42).fit(X, y_tg)
    rho_model = RandomForestRegressor(n_estimators=100, random_state=42).fit(X, y_rho)
    return tg_model, rho_model

tg_model, rho_model = train_analytical_model()

# --- 10 EXAMPLE POLYMERS FOR PRESENTATION ---
EXAMPLE_POLYMERS = {
    "Polyethylene (PE)": "CC",
    "Polypropylene (PP)": "CC(C)",
    "Polystyrene (PS)": "C(C)Cc1ccccc1",
    "Poly(Vinyl Chloride) (PVC)": "CC(Cl)",
    "Polyethylene Terephthalate (PET)": "C(=O)c1ccc(cc1)C(=O)OCCO",
    "Polyamide-6 (Nylon-6)": "CCCCCC(=O)N",
    "Polycarbonate (PC)": "C(=O)Oc1ccc(cc1)C(C)(C)c2ccc(OC(=O)C)cc2",
    "Polyurethane (PU)": "CC(=O)NCCCCN",
    "Epoxy Resin": "C1OCCN1C(=O)C",
    "Polyacrylonitrile (PAN)": "CC(C#N)"
}

def get_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        return [Descriptors.MolWt(mol), Descriptors.MolLogP(mol), 
                Descriptors.NumRotatableBonds(mol), Descriptors.TPSA(mol)], mol
    return None, None

def get_interpretation(tg, rho, descriptors):
    """Generate interpretation and applications based on predicted properties."""
    mw, logp, rot, tpsa = descriptors
    
    characteristics = []
    applications = []
    
    # Tg interpretation
    if tg < 100:
        characteristics.append("**Very Flexible Polymer** - Low glass transition temperature")
        applications.extend(["Soft packaging materials", "Elastomers", "Flexible films", "Cushioning applications"])
    elif tg < 200:
        characteristics.append("**Flexible Polymer** - Moderate glass transition temperature")
        applications.extend(["Fibers and textiles", "Impact-resistant plastics", "Consumer products", "Medical devices"])
    else:
        characteristics.append("**Rigid Polymer** - High glass transition temperature")
        applications.extend(["Structural components", "Optical devices", "Heat-resistant applications", "Aerospace components"])
    
    # Density interpretation
    if rho < 0.95:
        characteristics.append("**Lightweight Material** - Low density")
        applications.extend(["Automotive weight reduction", "Foam insulation", "Buoyancy applications", "Lightweight packaging"])
    elif rho < 1.20:
        characteristics.append("**Standard Density** - Normal molecular packing")
        applications.extend(["General-purpose plastics", "Bottles and containers", "Daily consumer products", "Industrial containers"])
    else:
        characteristics.append("**Dense Material** - High density")
        applications.extend(["Heavy-duty applications", "Chemical-resistant containers", "High-strength components", "Structural reinforcement"])
    
    # Flexibility based on rotatable bonds
    if rot < 2:
        characteristics.append("**Rigid Structure** - Minimal chain flexibility")
    elif rot < 5:
        characteristics.append("**Semi-Flexible** - Moderate chain flexibility")
    else:
        characteristics.append("**Flexible Backbone** - High chain flexibility")
    
    return characteristics, list(set(applications))

# --- MAIN INTERFACE ---
st.markdown("""
    <div class="header-box">
        <h1>POLYSIGHTS ANALYTICAL ENGINE</h1>
        <p>Computational Property Prediction for Advanced Polymer Engineering</p>
    </div>
    """, unsafe_allow_html=True)

# Top Info Bar
c1, c2, c3 = st.columns(3)
with c1:
    st.write("**ORGANIZATION**")
    st.write("Chemical Technology Department")
with c2:
    st.write("**PROJECT TEAM**")
    st.write("S. Arun Ulagappan | A. Vishaal | Akash S J")
with c3:
    st.write("**REVIEW DATE**")
    st.write("October 04, 2026")

st.write("---")

# Navigation Tabs
tab_doc, tab_app = st.tabs(["RESEARCH DOCUMENTATION", "PREDICTIVE ANALYSIS"])

with tab_doc:
    st.markdown('<div class="section-header">Project Methodology</div>', unsafe_allow_html=True)
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.markdown("""
        <div class="info-card">
            <h3>Technical Abstract</h3>
            <p>Traditional polymer characterization requires extensive laboratory resources. 
            This analytical engine leverages RDKit-derived molecular descriptors and Random Forest 
            Regression to provide instantaneous estimates of thermophysical properties.</p>
            <p>By mapping the repeating unit's SMILES notation to high-dimensional feature space, 
            we enable rapid material screening for industrial applications.</p>
        </div>
        """, unsafe_allow_html=True)
        
    with col_right:
        st.markdown("""
        <div class="info-card">
            <h3>Key Performance Indicators</h3>
            <ul>
                <li><strong>Glass Transition (Tg):</strong> Critical for determining service temperature ranges.</li>
                <li><strong>Specific Gravity:</strong> Essential for strength-to-weight ratio calculations.</li>
                <li><strong>Molecular Lipophilicity:</strong> Predicts chemical resistance and solubility.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

with tab_app:
    st.markdown('<div class="section-header">Material Configuration</div>', unsafe_allow_html=True)
    
    st.write("### Quick Example Library (Click to Load)")
    # Display 10 examples in 5 columns x 2 rows
    example_names = list(EXAMPLE_POLYMERS.keys())
    
    col1, col2, col3, col4, col5 = st.columns(5)
    cols = [col1, col2, col3, col4, col5]
    
    selected_polymer = None
    for idx, col in enumerate(cols):
        with col:
            if st.button(example_names[idx], key=f"btn_{idx}"):
                selected_polymer = example_names[idx]
    
    col1, col2, col3, col4, col5 = st.columns(5)
    cols = [col1, col2, col3, col4, col5]
    
    for idx, col in enumerate(cols):
        with col:
            if st.button(example_names[idx + 5], key=f"btn_{idx+5}"):
                selected_polymer = example_names[idx + 5]
    
    st.write("---")
    
    main_col, side_col = st.columns([1, 1])
    
    with main_col:
        st.write("### Structure Input")
        
        # Use selected example or text input
        default_smiles = "C(C)Cc1ccccc1"
        if selected_polymer:
            default_smiles = EXAMPLE_POLYMERS[selected_polymer]
            st.success(f"✅ Loaded: **{selected_polymer}**")
        
        smiles_val = st.text_input("Repeating Unit SMILES:", default_smiles, help="Enter the monomeric structure string")
        
        predict_trigger = st.button("🧪 RUN ANALYTICAL INFERENCE", key="run_analysis")

    if smiles_val:
        features, mol = get_descriptors(smiles_val)
        
        if features:
            with side_col:
                st.write("### 2D Graphical Representation")
                img = Draw.MolToImage(mol, size=(450, 450))
                st.image(img, use_container_width=True)
            
            st.markdown('<div class="section-header">Inference Data</div>', unsafe_allow_html=True)
            
            # Metric Display
            tg_val = tg_model.predict([features])[0]
            rho_val = rho_model.predict([features])[0]
            
            res_1, res_2, res_3, res_4 = st.columns(4)
            res_1.metric("🌡️ Tg Prediction", f"{tg_val:.2f} K")
            res_2.metric("⚖️ Density", f"{rho_val:.3f} g/cm³")
            res_3.metric("⚗️ Mol. Weight", f"{features[0]:.1f}")
            res_4.metric("🧬 LogP", f"{features[1]:.2f}")
            
            # Tabular Analysis
            st.write("### Computational Descriptors")
            data = {
                "Property Parameter": ["Molecular Weight (MW)", "Lipophilicity (LogP)", "Rotatable Bonds", "TPSA"],
                "Calculated Value": [f"{features[0]:.1f}", f"{features[1]:.2f}", f"{int(features[2])}", f"{features[3]:.1f}"],
                "Standard Unit": ["g/mol", "Log Scale", "Integer Count", "Å²"]
            }
            st.table(pd.DataFrame(data))
            
            # ===== INTERPRETATION & APPLICATIONS SECTION =====
            st.markdown('<div class="section-header">📊 Interpretation & Real-World Applications</div>', unsafe_allow_html=True)
            
            characteristics, applications = get_interpretation(tg_val, rho_val, features)
            
            # Display characteristics
            st.markdown('<div class="interpretation-box">', unsafe_allow_html=True)
            st.write("**✨ Polymer Characteristics:**")
            for char in characteristics:
                st.markdown(f'<div class="app-item">✓ {char}</div>', unsafe_allow_html=True)
            
            st.write("")
            st.write("**🏭 Real-World Applications Based on Properties:**")
            for app in applications:
                st.markdown(f'<div class="app-item">→ {app}</div>', unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Tech Analysis
            st.markdown('<div class="results-container">', unsafe_allow_html=True)
            st.write("### Technical Summary")
            if tg_val > 300:
                st.info(f"**High Performance Grade:** This material (Tg = {tg_val:.1f}K) exhibits exceptional thermal stability, suitable for rigid engineering components and high-temperature applications.")
            elif tg_val > 200:
                st.info(f"**Engineering Grade:** This material (Tg = {tg_val:.1f}K) shows good balance between rigidity and processability, ideal for structural applications.")
            else:
                st.info(f"**Elastomeric Grade:** This material (Tg = {tg_val:.1f}K) shows high chain mobility, suitable for flexible products and elastic applications.")
            
            if rho_val < 0.95:
                st.success(f"**Lightweight Classification:** Density of {rho_val:.3f} g/cm³ makes this ideal for weight-critical applications like automotive and aerospace.")
            elif rho_val < 1.15:
                st.success(f"**Standard Classification:** Density of {rho_val:.3f} g/cm³ is typical for general-purpose plastics in consumer products.")
            else:
                st.success(f"**Dense Material:** Density of {rho_val:.3f} g/cm³ indicates strong chemical resistance, suitable for industrial containers.")
            st.markdown('</div>', unsafe_allow_html=True)
            
        else:
            st.error("🚨 CHEM-VALIDATION ERROR: The provided SMILES string does not follow standard IUPAC/SMILES rules.")

# Footer
st.markdown("<br><hr><p style='text-align: center; color: #94a3b8 !important;'>Analytical Model v1.0.4 | Internal Departmental Use Only</p>", unsafe_allow_html=True)