import os
import subprocess
from PIL import Image

def generate_system_architecture_diagram(output_svg_path, output_png_path):
    width = 2200
    height = 1300

    font_sans = "system-ui, -apple-system, 'Helvetica Neue', Helvetica, Arial, sans-serif"
    font_mono = "'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, monospace"

    svg_parts = []
    svg_parts.append(f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}">
  <defs>
    <style>
      .sans {{ font-family: {font_sans}; }}
      .mono {{ font-family: {font_mono}; }}
      .title {{ font-size: 26px; font-weight: 800; fill: #0f172a; letter-spacing: -0.5px; }}
      .subtitle {{ font-size: 13.5px; font-weight: 500; fill: #475569; }}
      .stage-title {{ font-size: 15px; font-weight: 700; fill: #1e293b; letter-spacing: 0.2px; }}
      .stage-subtitle {{ font-size: 12px; font-weight: 500; fill: #64748b; }}
      .card-title {{ font-size: 15.5px; font-weight: 700; }}
      .card-subtitle {{ font-size: 12px; font-weight: 500; }}
      .badge-text {{ font-size: 10.5px; font-weight: 700; }}
      .item-title {{ font-size: 12px; font-weight: 700; fill: #1e293b; }}
      .item-desc {{ font-size: 11px; font-weight: 400; fill: #334155; line-height: 1.4; }}
      .formula {{ font-family: {font_mono}; font-size: 11.5px; font-weight: 600; fill: #0f172a; }}
      .metric-label {{ font-size: 10.5px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.4px; }}
      .metric-val {{ font-family: {font_mono}; font-size: 21px; font-weight: 800; }}
      .metric-sub {{ font-size: 10px; font-weight: 500; fill: #475569; }}
    </style>

    <filter id="shadow-sm" x="-4%" y="-4%" width="108%" height="108%">
      <feDropShadow dx="0" dy="2" stdDeviation="3" flood-color="#0f172a" flood-opacity="0.04"/>
    </filter>
    <filter id="shadow-md" x="-4%" y="-4%" width="108%" height="108%">
      <feDropShadow dx="0" dy="3" stdDeviation="6" flood-color="#0f172a" flood-opacity="0.05"/>
    </filter>

    <linearGradient id="bg-grad" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" stop-color="#ffffff"/>
      <stop offset="100%" stop-color="#f8fafc"/>
    </linearGradient>

    <marker id="arrow-slate" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 1 2 L 7 5 L 1 8 z" fill="#64748b"/>
    </marker>
    <marker id="arrow-blue" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 1 2 L 7 5 L 1 8 z" fill="#2563eb"/>
    </marker>
    <marker id="arrow-purple" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 1 2 L 7 5 L 1 8 z" fill="#7c3aed"/>
    </marker>
    <marker id="arrow-emerald" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 1 2 L 7 5 L 1 8 z" fill="#059669"/>
    </marker>
  </defs>

  <!-- Background -->
  <rect width="{width}" height="{height}" fill="url(#bg-grad)"/>

  <!-- Top Title Header Area -->
  <g transform="translate(50, 24)">
    <text x="0" y="24" class="sans title">TWO-STAGE NEWS RECOMMENDATION ARCHITECTURE</text>
    <text x="0" y="47" class="sans subtitle">End-to-End Pipeline on EB-NeRD: Full-Catalogue Candidate Retrieval followed by Diversity-Oriented Reranking</text>
    
    <!-- Benchmark Meta Tag -->
    <rect x="1750" y="4" width="350" height="32" rx="16" fill="#f1f5f9" stroke="#cbd5e1" stroke-width="1.2"/>
    <circle cx="1768" cy="20" r="4.5" fill="#10b981"/>
    <text x="1782" y="24" class="mono" font-size="11.5" font-weight="700" fill="#334155">EB-NeRD BENCHMARK SPECIFICATION</text>
  </g>
''')

    # STAGE 1 CONTAINER (y: 86 to 686, height = 600)
    s1_x, s1_y, s1_w, s1_h = 50, 86, 2100, 600
    svg_parts.append(f'''
  <!-- Stage 1 Container Box -->
  <rect x="{s1_x}" y="{s1_y}" width="{s1_w}" height="{s1_h}" rx="14" fill="#ffffff" stroke="#94a3b8" stroke-width="1.8" filter="url(#shadow-md)"/>
  <rect x="{s1_x}" y="{s1_y}" width="{s1_w}" height="40" rx="14" fill="#f8fafc"/>
  <rect x="{s1_x}" y="{s1_y+28}" width="{s1_w}" height="12" fill="#f8fafc"/>
  <line x1="{s1_x}" y1="{s1_y+40}" x2="{s1_x+s1_w}" y2="{s1_y+40}" stroke="#e2e8f0" stroke-width="1.2"/>

  <!-- Stage 1 Header Elements -->
  <rect x="{s1_x+16}" y="{s1_y+9}" width="78" height="22" rx="6" fill="#dbeafe"/>
  <text x="{s1_x+55}" y="{s1_y+24}" class="sans badge-text" fill="#1e40af" text-anchor="middle">STAGE 1</text>
  <text x="{s1_x+106}" y="{s1_y+25}" class="sans stage-title">FULL-CATALOGUE CANDIDATE RETRIEVAL</text>
  <text x="{s1_x+s1_w-20}" y="{s1_y+25}" class="sans stage-subtitle" text-anchor="end">Evaluates all 20,738 eligible articles per impression request (&gt;566M candidate pairs scored on validation)</text>
''')

    # Stage 1 Input Card (y: 138 to 192, height = 54)
    in_w, in_h = 1480, 54
    in_x = s1_x + (s1_w - in_w) // 2
    in_y = s1_y + 52
    svg_parts.append(f'''
  <!-- Stage 1 Input Card -->
  <rect x="{in_x}" y="{in_y}" width="{in_w}" height="{in_h}" rx="9" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1.4"/>
  <rect x="{in_x+12}" y="{in_y+16}" width="96" height="22" rx="5" fill="#e2e8f0"/>
  <text x="{in_x+60}" y="{in_y+31}" class="sans badge-text" fill="#334155" text-anchor="middle">CATALOGUE</text>
  <text x="{in_x+120}" y="{in_y+26}" class="sans" font-size="13.5" font-weight="700" fill="#0f172a">Eligible News Catalogue (20,738 Articles)  x  User Context &amp; Sequential Interaction History</text>
  <text x="{in_x+120}" y="{in_y+44}" class="sans" font-size="11" font-weight="400" fill="#64748b">Eligibility rule: published_time &lt;= impression_time | User reading history, dwell times, scroll depths, and click gaps</text>
''')

    # Branching arrows
    lstm_cx = s1_x + 510
    xgb_cx = s1_x + s1_w - 510
    svg_parts.append(f'''
  <!-- Split Paths to Dual Retrieval Engines -->
  <path d="M {in_x + in_w//2} {in_y + in_h} L {in_x + in_w//2} {in_y + in_h + 8} L {lstm_cx} {in_y + in_h + 8} L {lstm_cx} {in_y + in_h + 22}" 
        fill="none" stroke="#64748b" stroke-width="1.6" marker-end="url(#arrow-slate)"/>
  <path d="M {in_x + in_w//2} {in_y + in_h} L {in_x + in_w//2} {in_y + in_h + 8} L {xgb_cx} {in_y + in_h + 8} L {xgb_cx} {in_y + in_h + 22}" 
        fill="none" stroke="#64748b" stroke-width="1.6" marker-end="url(#arrow-slate)"/>
''')

    # Dual Engines (y: 216 to 452, height = 236)
    eng_y = in_y + in_h + 24
    eng_h = 238
    eng_w = 995
    eng1_x = s1_x + 24
    eng2_x = s1_x + s1_w - 24 - eng_w

    # Engine 1: Target-Aware LSTM
    svg_parts.append(f'''
  <!-- Engine 1: Target-Aware LSTM Ranker -->
  <rect x="{eng1_x}" y="{eng_y}" width="{eng_w}" height="{eng_h}" rx="10" fill="#f0f7ff" stroke="#3b82f6" stroke-width="1.8" filter="url(#shadow-sm)"/>
  <rect x="{eng1_x}" y="{eng_y}" width="{eng_w}" height="35" rx="10" fill="#e0f2fe"/>
  <rect x="{eng1_x}" y="{eng_y+25}" width="{eng_w}" height="10" fill="#e0f2fe"/>
  <line x1="{eng1_x}" y1="{eng_y+35}" x2="{eng1_x+eng_w}" y2="{eng_y+35}" stroke="#bae6fd" stroke-width="1.2"/>

  <text x="{eng1_x+16}" y="{eng_y+23}" class="sans card-title" fill="#0369a1">Engine 1: Target-Aware LSTM Ranker</text>
  <rect x="{eng1_x+eng_w-190}" y="{eng_y+7}" width="176" height="21" rx="5" fill="#bae6fd"/>
  <text x="{eng1_x+eng_w-102}" y="{eng_y+21}" class="mono badge-text" fill="#0369a1" text-anchor="middle">NEURAL SEQUENCE MODEL</text>

  <g transform="translate({eng1_x+14}, {eng_y+44})">
    <!-- Row 1: Multimodal Encoder -->
    <rect x="0" y="0" width="{eng_w-28}" height="40" rx="6" fill="#ffffff" stroke="#bfdbfe" stroke-width="1"/>
    <text x="12" y="16" class="sans item-title" fill="#0284c7">145-D Multimodal Article Encoding (Linear Projection to 128-D)</text>
    <text x="12" y="32" class="sans item-desc">Learned category (16-D), article type (8-D), topic bag (16-D), Gemini text projection (64-D), ResNet image projection (32-D), dense statistics (9-D)</text>

    <!-- Row 2: Sequence & Attention -->
    <rect x="0" y="47" width="{eng_w-28}" height="44" rx="6" fill="#ffffff" stroke="#bfdbfe" stroke-width="1"/>
    <text x="12" y="63" class="sans item-title" fill="#0284c7">Sequential History Encoder with Recency-Biased QK Attention</text>
    <text x="12" y="80" class="sans item-desc">Chronological LSTM(128, 128) over clicked items + 8-D interaction features (dwell, scroll %, gap) | Query-Key attention: alpha_t = softmax((q^T k_t)/8 + w_rec * Delta_t)</text>

    <!-- Row 3: Head & Freshness -->
    <rect x="0" y="98" width="{eng_w-28}" height="44" rx="6" fill="#ffffff" stroke="#bfdbfe" stroke-width="1"/>
    <text x="12" y="114" class="sans item-title" fill="#0284c7">344-D Interaction Head with Soft Freshness Regularization</text>
    <text x="12" y="131" class="sans item-desc">Candidate (128-D) + context (128-D) + interaction products (65-D) + request features (23-D) -&gt; MLP(344-&gt;256-&gt;128-&gt;1) | Decay: s = logit - 2.0 * ln(1 + Delta_t_hours)</text>

    <!-- Row 4: Metrics -->
    <rect x="0" y="149" width="{eng_w-28}" height="32" rx="6" fill="#e0f2fe" stroke="#7dd3fc" stroke-width="1"/>
    <text x="12" y="170" class="sans" font-size="11.5" font-weight="700" fill="#0369a1">Validation Standalone Retrieval:</text>
    <text x="220" y="170" class="mono" font-size="11.5" font-weight="700" fill="#0c4a6e">Val Recall@100: 0.907  |  Val Recall@200: 0.965  |  Val nDCG@100: 0.251</text>
  </g>
''')

    # Engine 2: XGBoost GBDT
    svg_parts.append(f'''
  <!-- Engine 2: XGBoost GBDT Ranker -->
  <rect x="{eng2_x}" y="{eng_y}" width="{eng_w}" height="{eng_h}" rx="10" fill="#f0fdf4" stroke="#16a34a" stroke-width="1.8" filter="url(#shadow-sm)"/>
  <rect x="{eng2_x}" y="{eng_y}" width="{eng_w}" height="35" rx="10" fill="#dcfce7"/>
  <rect x="{eng2_x}" y="{eng_y+25}" width="{eng_w}" height="10" fill="#dcfce7"/>
  <line x1="{eng2_x}" y1="{eng_y+35}" x2="{eng2_x+eng_w}" y2="{eng_y+35}" stroke="#bbf7d0" stroke-width="1.2"/>

  <text x="{eng2_x+16}" y="{eng_y+23}" class="sans card-title" fill="#15803d">Engine 2: XGBoost GBDT Ranker</text>
  <rect x="{eng2_x+eng_w-200}" y="{eng_y+7}" width="186" height="21" rx="5" fill="#bbf7d0"/>
  <text x="{eng2_x+eng_w-107}" y="{eng_y+21}" class="mono badge-text" fill="#15803d" text-anchor="middle">POINT-IN-TIME TREE ENSEMBLE</text>

  <g transform="translate({eng2_x+14}, {eng_y+44})">
    <!-- Row 1: Point-in-Time Features -->
    <rect x="0" y="0" width="{eng_w-28}" height="40" rx="6" fill="#ffffff" stroke="#bbf7d0" stroke-width="1"/>
    <text x="12" y="16" class="sans item-title" fill="#16a34a">202 Point-in-Time Features (Strict Zero Leakage)</text>
    <text x="12" y="32" class="sans item-desc">Exact point-in-time calculation at request instant; log1p transformations on skewed counts, dwell durations, and historical aggregates</text>

    <!-- Row 2: Category Affinity & Freshness Splits -->
    <rect x="0" y="47" width="{eng_w-28}" height="44" rx="6" fill="#ffffff" stroke="#bbf7d0" stroke-width="1"/>
    <text x="12" y="63" class="sans item-title" fill="#16a34a">Short-Term Category Affinity &amp; Freshness Dynamics</text>
    <text x="12" y="80" class="sans item-desc">Rolling 72-hour category interaction affinity captures rapid reader topic shifts; explicit article_age_h and slate_relative_age_h splits</text>

    <!-- Row 3: Tree Interactions & Objective -->
    <rect x="0" y="98" width="{eng_w-28}" height="44" rx="6" fill="#ffffff" stroke="#bbf7d0" stroke-width="1"/>
    <text x="12" y="114" class="sans item-title" fill="#16a34a">Tree-Based Higher-Order Feature Interactions (31 leaves, 25 trees)</text>
    <text x="12" y="131" class="sans item-desc">Non-linear partitioning across user historical CTR, category alignment, and temporal decay; provides complementary non-neural scoring</text>

    <!-- Row 4: Metrics -->
    <rect x="0" y="149" width="{eng_w-28}" height="32" rx="6" fill="#dcfce7" stroke="#86efac" stroke-width="1"/>
    <text x="12" y="170" class="sans" font-size="11.5" font-weight="700" fill="#15803d">Validation Standalone Retrieval:</text>
    <text x="220" y="170" class="mono" font-size="11.5" font-weight="700" fill="#14532d">Val Recall@100: 0.858  |  Val Recall@200: 0.923  |  Val nDCG@100: 0.221</text>
  </g>
''')

    # Convergence Arrows to RRF
    rrf_y = eng_y + eng_h + 16
    rrf_h = 52
    rrf_w = 1480
    rrf_x = s1_x + (s1_w - rrf_w) // 2

    svg_parts.append(f'''
  <!-- Convergence Arrows to RRF -->
  <path d="M {lstm_cx} {eng_y + eng_h} L {lstm_cx} {rrf_y - 8} L {s1_x + s1_w//2 - 160} {rrf_y - 8} L {s1_x + s1_w//2 - 160} {rrf_y}"
        fill="none" stroke="#3b82f6" stroke-width="1.6" marker-end="url(#arrow-blue)"/>
  <path d="M {xgb_cx} {eng_y + eng_h} L {xgb_cx} {rrf_y - 8} L {s1_x + s1_w//2 + 160} {rrf_y - 8} L {s1_x + s1_w//2 + 160} {rrf_y}"
        fill="none" stroke="#16a34a" stroke-width="1.6" marker-end="url(#arrow-emerald)"/>

  <!-- Aggregation Engine: Weighted RRF -->
  <rect x="{rrf_x}" y="{rrf_y}" width="{rrf_w}" height="{rrf_h}" rx="9" fill="#faf5ff" stroke="#7c3aed" stroke-width="1.8" filter="url(#shadow-sm)"/>
  <rect x="{rrf_x+12}" y="{rrf_y+15}" width="160" height="22" rx="5" fill="#ede9fe"/>
  <text x="{rrf_x+92}" y="{rrf_y+30}" class="sans badge-text" fill="#6d28d9" text-anchor="middle">ENSEMBLE AGGREGATOR</text>
  <text x="{rrf_x+184}" y="{rrf_y+25}" class="sans" font-size="13.5" font-weight="700" fill="#581c87">Weighted Reciprocal Rank Fusion (RRF):  75% Target-Aware LSTM  +  25% XGBoost GBDT</text>
  <text x="{rrf_x+184}" y="{rrf_y+42}" class="mono" font-size="11" font-weight="600" fill="#4c1d95">RRF_Score(c) = 0.75 * [1 / (60 + Rank_LSTM(c))]  +  0.25 * [1 / (60 + Rank_XGB(c))]   (k = 60)</text>
''')

    # Arrow from RRF to Stage 1 Output
    out1_y = rrf_y + rrf_h + 14
    out1_h = 56
    out1_w = 1480
    out1_x = rrf_x

    svg_parts.append(f'''
  <!-- Arrow to Stage 1 Output -->
  <line x1="{s1_x + s1_w//2}" y1="{rrf_y + rrf_h}" x2="{s1_x + s1_w//2}" y2="{out1_y}" stroke="#7c3aed" stroke-width="1.6" marker-end="url(#arrow-purple)"/>

  <!-- Stage 1 Output Card -->
  <rect x="{out1_x}" y="{out1_y}" width="{out1_w}" height="{out1_h}" rx="9" fill="#eff6ff" stroke="#2563eb" stroke-width="2" filter="url(#shadow-sm)"/>
  
  <rect x="{out1_x+12}" y="{out1_y+17}" width="114" height="22" rx="5" fill="#2563eb"/>
  <text x="{out1_x+69}" y="{out1_y+32}" class="sans badge-text" fill="#ffffff" text-anchor="middle">STAGE 1 OUTPUT</text>
  
  <text x="{out1_x+138}" y="{out1_y+26}" class="sans" font-size="14.5" font-weight="800" fill="#1e3a8a">High-Recall Top-200 Candidate Shortlist</text>
  <text x="{out1_x+138}" y="{out1_y+43}" class="sans" font-size="11" font-weight="500" fill="#3b82f6">Prunes 99.04% of eligible catalogue; retains nearly all clicked items for downstream diversity reranking</text>

  <!-- Metric Badges inside Stage 1 Output -->
  <rect x="{out1_x+out1_w-420}" y="{out1_y+10}" width="195" height="36" rx="6" fill="#ffffff" stroke="#93c5fd" stroke-width="1.2"/>
  <text x="{out1_x+out1_w-322}" y="{out1_y+24}" class="sans" font-size="10" font-weight="700" fill="#64748b" text-anchor="middle">VAL RECALL@200</text>
  <text x="{out1_x+out1_w-322}" y="{out1_y+40}" class="mono" font-size="14" font-weight="800" fill="#0369a1" text-anchor="middle">0.948 (94.8%)</text>

  <rect x="{out1_x+out1_w-210}" y="{out1_y+10}" width="195" height="36" rx="6" fill="#ffffff" stroke="#93c5fd" stroke-width="1.2"/>
  <text x="{out1_x+out1_w-112}" y="{out1_y+24}" class="sans" font-size="10" font-weight="700" fill="#64748b" text-anchor="middle">VAL nDCG@100</text>
  <text x="{out1_x+out1_w-112}" y="{out1_y+40}" class="mono" font-size="14" font-weight="800" fill="#0369a1" text-anchor="middle">0.308</text>
''')

    # TRANSITION CONNECTOR (y: 686 to 726)
    trans_y1 = s1_y + s1_h
    svg_parts.append(f'''
  <!-- Transition Connector to Stage 2 -->
  <g transform="translate({s1_x + s1_w//2}, {trans_y1})">
    <line x1="0" y1="2" x2="0" y2="34" stroke="#0284c7" stroke-width="2.2" marker-end="url(#arrow-blue)"/>
    <rect x="-195" y="8" width="390" height="22" rx="11" fill="#ffffff" stroke="#cbd5e1" stroke-width="1.2" filter="url(#shadow-sm)"/>
    <text x="0" y="23" class="sans" font-size="11" font-weight="700" fill="#334155" text-anchor="middle">Passes Top-200 Candidate Shortlist to Stage 2</text>
  </g>
''')

    # STAGE 2 CONTAINER (y: 728 to 1268, height = 540)
    s2_x, s2_y, s2_w, s2_h = 50, 728, 2100, 540
    svg_parts.append(f'''
  <!-- Stage 2 Container Box -->
  <rect x="{s2_x}" y="{s2_y}" width="{s2_w}" height="{s2_h}" rx="14" fill="#ffffff" stroke="#0d9488" stroke-width="1.8" filter="url(#shadow-md)"/>
  <rect x="{s2_x}" y="{s2_y}" width="{s2_w}" height="40" rx="14" fill="#f0fdfa"/>
  <rect x="{s2_x}" y="{s2_y+28}" width="{s2_w}" height="12" fill="#f0fdfa"/>
  <line x1="{s2_x}" y1="{s2_y+40}" x2="{s2_x+s2_w}" y2="{s2_y+40}" stroke="#ccfbf1" stroke-width="1.2"/>

  <!-- Stage 2 Header Elements -->
  <rect x="{s2_x+16}" y="{s2_y+9}" width="78" height="22" rx="6" fill="#fef3c7"/>
  <text x="{s2_x+55}" y="{s2_y+24}" class="sans badge-text" fill="#92400e" text-anchor="middle">STAGE 2</text>
  <text x="{s2_x+106}" y="{s2_y+25}" class="sans stage-title">DIVERSITY RERANKING &amp; FINAL SLATE GENERATION</text>
  <text x="{s2_x+s2_w-20}" y="{s2_y+25}" class="sans stage-subtitle" text-anchor="end">Maximizes intra-list recommendation diversity while preserving top-ranked relevance accuracy</text>
''')

    # Stage 2 Left and Right columns
    s2_left_w = 1270
    s2_right_w = s2_w - s2_left_w - 48 - 24 # 758 px
    s2_left_x = s2_x + 24
    s2_right_x = s2_left_x + s2_left_w + 48

    # Left: MMR Reranker Card (y: 778 to 1040, height = 262)
    mmr_y = s2_y + 50
    mmr_h = 264
    svg_parts.append(f'''
  <!-- MMR Reranker Engine -->
  <rect x="{s2_left_x}" y="{mmr_y}" width="{s2_left_w}" height="{mmr_h}" rx="10" fill="#fffbeb" stroke="#d97706" stroke-width="1.8" filter="url(#shadow-sm)"/>
  <rect x="{s2_left_x}" y="{mmr_y}" width="{s2_left_w}" height="35" rx="10" fill="#fef3c7"/>
  <rect x="{s2_left_x}" y="{mmr_y+25}" width="{s2_left_w}" height="10" fill="#fef3c7"/>
  <line x1="{s2_left_x}" y1="{mmr_y+35}" x2="{s2_left_x+s2_left_w}" y2="{mmr_y+35}" stroke="#fde68a" stroke-width="1.2"/>

  <text x="{s2_left_x+16}" y="{mmr_y+23}" class="sans card-title" fill="#b45309">Maximal Marginal Relevance (MMR) Reranker (lambda = 0.9)</text>
  <rect x="{s2_left_x+s2_left_w-260}" y="{mmr_y+7}" width="246" height="21" rx="5" fill="#fde68a"/>
  <text x="{s2_left_x+s2_left_w-137}" y="{mmr_y+21}" class="mono badge-text" fill="#92400e" text-anchor="middle">LOCAL SUBMODULAR OPTIMIZATION</text>

  <g transform="translate({s2_left_x+14}, {mmr_y+44})">
    <!-- Row 1: Objective Formulation -->
    <rect x="0" y="0" width="{s2_left_w-28}" height="42" rx="6" fill="#ffffff" stroke="#fde68a" stroke-width="1"/>
    <text x="12" y="16" class="sans item-title" fill="#b45309">Greedy Sequential Slate Selection Objective</text>
    <text x="12" y="33" class="mono" font-size="11.5" font-weight="700" fill="#78350f">c* = arg max_{{c in R \\ S}} [ lambda * Rel_Stage1(c, u)  -  (1 - lambda) * max_{{s in S}} Sim_Semantic(c, s) ]    (lambda = 0.9)</text>

    <!-- Row 2: Gemini Embeddings Cosine Distance -->
    <rect x="0" y="49" width="{s2_left_w-28}" height="48" rx="6" fill="#ffffff" stroke="#fde68a" stroke-width="1"/>
    <text x="12" y="66" class="sans item-title" fill="#b45309">Cosine Distance over 768-D Gemini Semantic Text Embeddings</text>
    <text x="12" y="84" class="sans item-desc">Computes pairwise semantic similarity: Sim_Semantic(c, s) = cos(e_c, e_s) using pre-extracted multilingual Gemini embeddings (gemini-embedding-2); ensures slate covers diverse news themes</text>

    <!-- Row 3: Category Redundancy Penalties -->
    <rect x="0" y="104" width="{s2_left_w-28}" height="52" rx="6" fill="#ffffff" stroke="#fde68a" stroke-width="1"/>
    <text x="12" y="121" class="sans item-title" fill="#b45309">Cross-Article Category Redundancy Penalties</text>
    <text x="12" y="139" class="sans item-desc">Penalizes repetitive candidate articles sharing top-level categories with already-selected items; prevents slate monopolization by multiple stories from a single breaking topic</text>

    <!-- Row 4: Slate Construction Strategy -->
    <rect x="0" y="163" width="{s2_left_w-28}" height="44" rx="6" fill="#ffffff" stroke="#fde68a" stroke-width="1"/>
    <text x="12" y="180" class="sans item-title" fill="#b45309">Top-5 Greedy Item Extraction</text>
    <text x="12" y="198" class="sans item-desc">Begins with the highest-ranked Stage 1 candidate, iteratively appending 4 items that maximize marginal informational gain relative to previously selected items</text>
  </g>
''')

    # Left Bottom: Architectural Insights Card (y: 1054 to 1246, height = 192)
    ins_y = mmr_y + mmr_h + 12
    ins_h = 192
    svg_parts.append(f'''
  <!-- Architectural Insights Card -->
  <rect x="{s2_left_x}" y="{ins_y}" width="{s2_left_w}" height="{ins_h}" rx="10" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1.4" filter="url(#shadow-sm)"/>
  
  <text x="{s2_left_x+16}" y="{ins_y+24}" class="sans" font-size="13.5" font-weight="700" fill="#334155">PIPELINE DESIGN ADVANTAGES &amp; EMPIRICAL FINDINGS</text>
  
  <g transform="translate({s2_left_x+16}, {ins_y+36})">
    <!-- Point 1 -->
    <circle cx="6" cy="14" r="3.5" fill="#2563eb"/>
    <text x="18" y="18" class="sans" font-size="11.5" font-weight="700" fill="#0f172a">Decoupled Optimization Paradigm:</text>
    <text x="245" y="18" class="sans" font-size="11.5" font-weight="400" fill="#334155">Stage 1 focuses exclusively on global catalogue recall across 20,738 articles (Recall@200 = 0.948); Stage 2 handles local list diversity without full-catalogue computational overhead.</text>

    <!-- Point 2 -->
    <circle cx="6" cy="48" r="3.5" fill="#16a34a"/>
    <text x="18" y="52" class="sans" font-size="11.5" font-weight="700" fill="#0f172a">Neural Sequence + Tree Regularization:</text>
    <text x="295" y="52" class="sans" font-size="11.5" font-weight="400" fill="#334155">Target-Aware LSTM captures temporal interest trajectories; XGBoost tree splits anchor point-in-time freshness and short-term (72h) category affinities, mitigating neural overconfidence.</text>

    <!-- Point 3 -->
    <circle cx="6" cy="82" r="3.5" fill="#7c3aed"/>
    <text x="18" y="86" class="sans" font-size="11.5" font-weight="700" fill="#0f172a">Zero Ranking Accuracy Penalty:</text>
    <text x="250" y="86" class="sans" font-size="11.5" font-weight="400" fill="#334155">MMR reranking expands Intra-List Diversity (+12.4% relative gain) while maintaining Test nDCG@5 at 0.143 and Test Recall@5 at 0.225 (matching Stage 1 precision).</text>

    <!-- Point 4 -->
    <circle cx="6" cy="116" r="3.5" fill="#d97706"/>
    <text x="18" y="120" class="sans" font-size="11.5" font-weight="700" fill="#0f172a">Holdout Generalization Integrity:</text>
    <text x="260" y="120" class="sans" font-size="11.5" font-weight="400" fill="#334155">All transformations and vocabularies fitted strictly on training partition; verified across 29,142 holdout test impression requests with zero user overlap.</text>
  </g>
''')

    # Connector from MMR to Final Slate
    svg_parts.append(f'''
  <!-- Connector Arrow from MMR to Slate -->
  <line x1="{s2_left_x + s2_left_w}" y1="{mmr_y + mmr_h//2}" x2="{s2_right_x}" y2="{mmr_y + mmr_h//2}" stroke="#059669" stroke-width="2.2" marker-end="url(#arrow-emerald)"/>
''')

    # Right: Final Top-5 Recommendation Slate (y: 778 to 1246, height = 468)
    slate_y = mmr_y
    slate_h = 468
    slate_w = s2_right_w
    svg_parts.append(f'''
  <!-- Final Top-5 Recommendation Slate -->
  <rect x="{s2_right_x}" y="{slate_y}" width="{slate_w}" height="{slate_h}" rx="12" fill="#ecfdf5" stroke="#059669" stroke-width="2.2" filter="url(#shadow-md)"/>
  <rect x="{s2_right_x}" y="{slate_y}" width="{slate_w}" height="40" rx="12" fill="#d1fae5"/>
  <rect x="{s2_right_x}" y="{slate_y+28}" width="{slate_w}" height="12" fill="#d1fae5"/>
  <line x1="{s2_right_x}" y1="{slate_y+40}" x2="{s2_right_x+slate_w}" y2="{slate_y+40}" stroke="#a7f3d0" stroke-width="1.2"/>

  <rect x="{s2_right_x+16}" y="{slate_y+9}" width="116" height="22" rx="5" fill="#059669"/>
  <text x="{s2_right_x+74}" y="{slate_y+24}" class="sans badge-text" fill="#ffffff" text-anchor="middle">STAGE 2 OUTPUT</text>
  <text x="{s2_right_x+142}" y="{slate_y+26}" class="sans" font-size="15" font-weight="800" fill="#065f46">Final Top-5 News Slate</text>

  <!-- Subtitle -->
  <text x="{s2_right_x+16}" y="{slate_y+60}" class="sans" font-size="11.5" font-weight="500" fill="#047857">Delivered to user client feed (29,142 holdout test impression requests evaluated):</text>

  <!-- 4 Big Metric Highlight Cards (2x2 Grid) -->
  <g transform="translate({s2_right_x+16}, {slate_y+72})">
    <!-- Card 1: Test Recall@5 -->
    <rect x="0" y="0" width="{slate_w//2 - 22}" height="116" rx="8" fill="#ffffff" stroke="#6ee7b7" stroke-width="1.4" filter="url(#shadow-sm)"/>
    <text x="14" y="24" class="sans metric-label" fill="#047857">Test Recall@5</text>
    <text x="14" y="66" class="mono metric-val" fill="#065f46">0.225</text>
    <text x="14" y="90" class="sans metric-sub">High coverage across true reads</text>
    <rect x="{slate_w//2 - 95}" y="12" width="60" height="18" rx="4" fill="#d1fae5"/>
    <text x="{slate_w//2 - 65}" y="24" class="mono" font-size="9.5" font-weight="700" fill="#065f46" text-anchor="middle">22.5%</text>

    <!-- Card 2: Test nDCG@5 -->
    <rect x="{slate_w//2 - 10}" y="0" width="{slate_w//2 - 22}" height="116" rx="8" fill="#ffffff" stroke="#6ee7b7" stroke-width="1.4" filter="url(#shadow-sm)"/>
    <text x="{slate_w//2 + 4}" y="24" class="sans metric-label" fill="#047857">Test nDCG@5</text>
    <text x="{slate_w//2 + 4}" y="66" class="mono metric-val" fill="#065f46">0.143</text>
    <text x="{slate_w//2 + 4}" y="90" class="sans metric-sub">Preserves top-slate ranking precision</text>
    <rect x="{slate_w - 105}" y="12" width="68" height="18" rx="4" fill="#d1fae5"/>
    <text x="{slate_w - 71}" y="24" class="mono" font-size="9.5" font-weight="700" fill="#065f46" text-anchor="middle">TOP-5 RANK</text>

    <!-- Card 3: Test Shared ILD@5 -->
    <rect x="0" y="128" width="{slate_w//2 - 22}" height="116" rx="8" fill="#ffffff" stroke="#6ee7b7" stroke-width="1.4" filter="url(#shadow-sm)"/>
    <text x="14" y="152" class="sans metric-label" fill="#047857">Test Shared ILD@5</text>
    <text x="14" y="194" class="mono metric-val" fill="#065f46">0.707</text>
    <text x="14" y="218" class="sans metric-sub">Mean pairwise cosine semantic distance</text>
    <rect x="{slate_w//2 - 110}" y="140" width="75" height="18" rx="4" fill="#d1fae5"/>
    <text x="{slate_w//2 - 72}" y="152" class="mono" font-size="9.5" font-weight="700" fill="#065f46" text-anchor="middle">DIVERSITY</text>

    <!-- Card 4: Diversity Gain -->
    <rect x="{slate_w//2 - 10}" y="128" width="{slate_w//2 - 22}" height="116" rx="8" fill="#ffffff" stroke="#34d399" stroke-width="1.6" filter="url(#shadow-sm)"/>
    <text x="{slate_w//2 + 4}" y="152" class="sans metric-label" fill="#047857">Diversity Gain</text>
    <text x="{slate_w//2 + 4}" y="194" class="mono metric-val" fill="#047857">+12.4%</text>
    <text x="{slate_w//2 + 4}" y="218" class="sans metric-sub">Relative expansion over baseline (0.629)</text>
    <rect x="{slate_w - 95}" y="140" width="58" height="18" rx="4" fill="#a7f3d0"/>
    <text x="{slate_w - 66}" y="152" class="mono" font-size="9.5" font-weight="700" fill="#065f46" text-anchor="middle">IMPROVE</text>
  </g>

  <!-- Simulated Recommendation Slate Items Preview -->
  <g transform="translate({s2_right_x+16}, {slate_y+332})">
    <rect x="0" y="0" width="{slate_w-32}" height="114" rx="7" fill="#ffffff" stroke="#a7f3d0" stroke-width="1"/>
    <text x="12" y="18" class="sans" font-size="11" font-weight="700" fill="#065f46">TOP-5 SLATE DIVERSITY PROFILE (5 DISTINCT HIGH-LEVEL CATEGORIES):</text>
    
    <rect x="10" y="26" width="{slate_w-52}" height="17" rx="3" fill="#f0fdf4"/>
    <text x="16" y="38" class="mono" font-size="10" fill="#166534">1. [Nyheder/Politik] Klimaaftale vedtaget i Folketinget (Rel: 0.94, Cat #1)</text>
    
    <rect x="10" y="47" width="{slate_w-52}" height="17" rx="3" fill="#f0fdf4"/>
    <text x="16" y="59" class="mono" font-size="10" fill="#166534">2. [Sport/Fodbold] Champions League analyse og højdepunkter (Rel: 0.91, Cat #4)</text>

    <rect x="10" y="68" width="{slate_w-52}" height="17" rx="3" fill="#f0fdf4"/>
    <text x="16" y="80" class="mono" font-size="10" fill="#166534">3. [Penge/Erhverv] Renteudvikling og markedsreaktion (Rel: 0.88, Cat #7)</text>

    <rect x="10" y="89" width="{slate_w-52}" height="17" rx="3" fill="#f0fdf4"/>
    <text x="16" y="101" class="mono" font-size="10" fill="#166534">4. [Underholdning] Filmfestival anmeldelser og vindere (Rel: 0.85, Cat #12)</text>
  </g>
''')

    svg_parts.append('</svg>')

    full_svg = "".join(svg_parts)
    with open(output_svg_path, "w", encoding="utf-8") as f:
        f.write(full_svg)

    # Convert to PNG using resvg at 2200x1300
    subprocess.run(["resvg", output_svg_path, output_png_path, "-w", str(width), "-h", str(height)], check=True)

    # Re-save with 300 DPI metadata in PIL
    img = Image.open(output_png_path)
    img.save(output_png_path, dpi=(300, 300))
    print(f"Generated {output_png_path}: size={img.size}, dpi={img.info.get('dpi')}")

if __name__ == "__main__":
    svg_file = "/tmp/system_architecture.svg"
    png_file = "/Users/winston/Documents/School/Y4S1/BT4222/BT4222-Final-Project/figures/system_architecture.png"
    generate_system_architecture_diagram(svg_file, png_file)
