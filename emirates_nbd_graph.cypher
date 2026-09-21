// =====================================================================
// Emirates NBD Strategic Report 2025 — Knowledge Graph for Neo4j
// Extracted manually (no LLM extraction step needed on your end).
// Run with: cypher-shell -u neo4j -p <password> -f emirates_nbd_graph.cypher
// or paste into Neo4j Browser.
// =====================================================================

// ---------- Constraints ----------
CREATE CONSTRAINT org_name IF NOT EXISTS FOR (o:Organization) REQUIRE o.name IS UNIQUE;
CREATE CONSTRAINT sub_name IF NOT EXISTS FOR (s:Subsidiary) REQUIRE s.name IS UNIQUE;
CREATE CONSTRAINT seg_name IF NOT EXISTS FOR (s:Segment) REQUIRE s.name IS UNIQUE;
CREATE CONSTRAINT person_name IF NOT EXISTS FOR (p:Person) REQUIRE p.name IS UNIQUE;
CREATE CONSTRAINT country_name IF NOT EXISTS FOR (c:Country) REQUIRE c.name IS UNIQUE;
CREATE CONSTRAINT award_id IF NOT EXISTS FOR (a:Award) REQUIRE a.id IS UNIQUE;
CREATE CONSTRAINT priority_name IF NOT EXISTS FOR (p:StrategicPriority) REQUIRE p.name IS UNIQUE;
CREATE CONSTRAINT event_id IF NOT EXISTS FOR (e:Event) REQUIRE e.id IS UNIQUE;
CREATE CONSTRAINT metric_id IF NOT EXISTS FOR (m:FinancialMetric) REQUIRE m.id IS UNIQUE;

// ---------- Core organization ----------
MERGE (org:Organization {name: 'Emirates NBD Bank (P.J.S.C.)'})
  SET org.founded_context = 'Largest Dubai-based Bank, 55.8% owned by Government of Dubai (ICD & Dubai Holdings)',
      org.report_year = 2025,
      org.gcc_rank_by_assets = 4,
      org.uae_most_profitable = true;

// ---------- Subsidiaries ----------
UNWIND [
  {name:'DenizBank Anonim Sirketi', desc:'Universal bank in Türkiye, 5th largest private bank'},
  {name:'Emirates Islamic Bank (P.J.S.C.)', desc:'Third-largest Islamic bank in UAE by assets and branch network'},
  {name:'Emirates NBD Egypt S.A.E.', desc:'Egypt banking subsidiary'},
  {name:'Emirates NBD Capital', desc:'Investment banking / debt capital markets arm'},
  {name:'Emirates NBD Securities', desc:'Brokerage arm'},
  {name:'Emirates NBD Global Services', desc:'Group services subsidiary (Tanfeeth, IT, Digital Office, Retail Sales)'},
  {name:'Emirates NBD Innovation Fund', desc:'Corporate venture capital arm, USD 100mn fund'}
] AS s
MERGE (sub:Subsidiary {name: s.name})
  SET sub.description = s.desc
WITH sub
MATCH (org:Organization {name:'Emirates NBD Bank (P.J.S.C.)'})
MERGE (sub)-[:SUBSIDIARY_OF]->(org);

// ---------- Business Segments (FY2025, AED billion, YoY vs FY2024) ----------
UNWIND [
  {name:'Retail Banking and Wealth Management', code:'RBWM', income:19.7, income_yoy:11, pbt:11.8, pbt_yoy:17, advances:184, advances_yoy:25, deposits:384, deposits_yoy:18},
  {name:'Corporate and Institutional Banking', code:'C&IB', income:9.0, income_yoy:11, pbt:10.9, pbt_yoy:10, advances:340, advances_yoy:42, deposits:293, deposits_yoy:21},
  {name:'Global Markets and Treasury', code:'GM&T', income:2.3, income_yoy:-16, pbt:2.0, pbt_yoy:-17, advances:null, advances_yoy:null, deposits:null, deposits_yoy:null},
  {name:'DenizBank', code:'DENIZ', income:13.8, income_yoy:25, pbt:3.3, pbt_yoy:12, advances:97, advances_yoy:16, deposits:108, deposits_yoy:11},
  {name:'Emirates Islamic', code:'EI', income:6.0, income_yoy:11, pbt:3.9, pbt_yoy:26, advances:93, advances_yoy:24, deposits:102, deposits_yoy:33},
  {name:'International', code:'INTL', income:3.3, income_yoy:19, pbt:null, pbt_yoy:null, advances:78, advances_yoy:38, deposits:null, deposits_yoy:null}
] AS seg
MERGE (s:Segment {name: seg.name})
  SET s.code = seg.code,
      s.income_AED_bn_2025 = seg.income,
      s.income_yoy_pct = seg.income_yoy,
      s.pbt_AED_bn_2025 = seg.pbt,
      s.pbt_yoy_pct = seg.pbt_yoy,
      s.customer_advances_AED_bn_2025 = seg.advances,
      s.advances_yoy_pct = seg.advances_yoy,
      s.customer_deposits_AED_bn_2025 = seg.deposits,
      s.deposits_yoy_pct = seg.deposits_yoy
WITH s
MATCH (org:Organization {name:'Emirates NBD Bank (P.J.S.C.)'})
MERGE (s)-[:SEGMENT_OF]->(org);

// ---------- Group-level financial metrics (FY2025 vs FY2024) ----------
UNWIND [
  {id:'net_profit_before_tax', name:'Net Profit Before Tax', unit:'AED bn', value_2025:29.8, value_2024:27.1},
  {id:'net_profit', name:'Net Profit', unit:'AED bn', value_2025:24.0, value_2024:23.0},
  {id:'total_assets', name:'Total Assets', unit:'AED bn', value_2025:1164.0, value_2024:997.0},
  {id:'deposits', name:'Deposits', unit:'AED bn', value_2025:786.0, value_2024:667.0},
  {id:'gross_loans', name:'Gross Loans', unit:'AED bn', value_2025:658.0, value_2024:529.0},
  {id:'market_cap', name:'Market Cap', unit:'AED bn', value_2025:176.0, value_2024:135.0},
  {id:'dividend_per_share', name:'Dividend Per Share', unit:'fils', value_2025:100.0, value_2024:null},
  {id:'rote', name:'Return on Tangible Equity', unit:'%', value_2025:19.4, value_2024:21.8},
  {id:'cet1', name:'CET1 Capital Ratio', unit:'%', value_2025:14.4, value_2024:14.7},
  {id:'total_income', name:'Total Income', unit:'AED bn', value_2025:49.3, value_2024:44.1},
  {id:'net_interest_income', name:'Net Interest Income', unit:'AED bn', value_2025:35.5, value_2024:32.4},
  {id:'non_funded_income', name:'Non-Funded Income', unit:'AED bn', value_2025:13.8, value_2024:11.7},
  {id:'operating_expenses', name:'Operating Expenses', unit:'AED bn', value_2025:15.0, value_2024:13.8},
  {id:'npl_ratio', name:'NPL Ratio', unit:'%', value_2025:2.4, value_2024:3.3},
  {id:'coverage_ratio', name:'Impaired Loan Coverage Ratio', unit:'%', value_2025:160.0, value_2024:156.0},
  {id:'liquidity_coverage_ratio', name:'Liquidity Coverage Ratio', unit:'%', value_2025:152.0, value_2024:197.0},
  {id:'capital_adequacy_ratio', name:'Capital Adequacy Ratio', unit:'%', value_2025:16.6, value_2024:17.1},
  {id:'tier1_ratio', name:'Tier 1 Ratio', unit:'%', value_2025:15.5, value_2024:16.0},
  {id:'branches', name:'Branches (Global)', unit:'count', value_2025:787.0, value_2024:848.0},
  {id:'employees', name:'Employees', unit:'count', value_2025:35000.0, value_2024:null},
  {id:'nps', name:'Net Promoter Score', unit:'score', value_2025:56.0, value_2024:48.0},
  {id:'active_customers', name:'Active Customers', unit:'mn', value_2025:9.8, value_2024:null}
] AS m
MERGE (metric:FinancialMetric {id: m.id})
  SET metric.name = m.name,
      metric.unit = m.unit,
      metric.value_fy2025 = m.value_2025,
      metric.value_fy2024 = m.value_2024
WITH metric
MATCH (org:Organization {name:'Emirates NBD Bank (P.J.S.C.)'})
MERGE (metric)-[:METRIC_OF]->(org);

// ---------- Credit ratings ----------
UNWIND [
  {agency:'Moody\u2019s', long_term:'A1', short_term:'P-1', outlook:'Stable'},
  {agency:'Fitch', long_term:'A+', short_term:'F1', outlook:'Stable'},
  {agency:'Capital Intelligence', long_term:'A+', short_term:'A1', outlook:'Stable'}
] AS r
MATCH (org:Organization {name:'Emirates NBD Bank (P.J.S.C.)'})
MERGE (org)-[rel:RATED_BY {agency: r.agency}]->(org)
  SET rel.long_term = r.long_term, rel.short_term = r.short_term, rel.outlook = r.outlook;
// (self-relationship used to keep ratings as simple edge properties; alternative: separate :Rating nodes)

// ---------- Countries of operation ----------
UNWIND [
  {name:'UAE', branches:107, entity:'Emirates NBD'},
  {name:'Egypt', branches:64, entity:'Emirates NBD'},
  {name:'KSA', branches:22, entity:'Emirates NBD'},
  {name:'India', branches:3, entity:'Emirates NBD'},
  {name:'UK', branches:1, entity:'Emirates NBD'},
  {name:'Singapore', branches:1, entity:'Emirates NBD'},
  {name:'Türkiye', branches:575, entity:'DenizBank'},
  {name:'Austria', branches:10, entity:'DenizBank'},
  {name:'Germany', branches:3, entity:'DenizBank'},
  {name:'Bahrain', branches:1, entity:'DenizBank'},
  {name:'China', branches:0, entity:'Representative Office'},
  {name:'Indonesia', branches:0, entity:'Representative Office'}
] AS c
MERGE (country:Country {name: c.name})
  SET country.branch_count = c.branches, country.operating_entity = c.entity
WITH country
MATCH (org:Organization {name:'Emirates NBD Bank (P.J.S.C.)'})
MERGE (org)-[:OPERATES_IN]->(country);

// ---------- People / Executives ----------
UNWIND [
  {name:'H.H. Sheikh Ahmed Bin Saeed Al Maktoum', role:'Chairman'},
  {name:'Hesham Abdulla Al Qassim', role:'Vice Chairman and Managing Director'},
  {name:'Shayne Nelson', role:'Group Chief Executive Officer'},
  {name:'Patrick Sullivan', role:'Group Chief Financial Officer'},
  {name:'Ahmed Al Qassim', role:'Group Head, Wholesale Banking'},
  {name:'Marwan Hadi', role:'Group Head, Retail Banking and Wealth Management'},
  {name:'Ammar Al Haj', role:'Group Treasurer and Head of Global Markets'},
  {name:'Recep Ba\u015ftu\u011f', role:'Chief Executive Officer, DenizBank'},
  {name:'Farid AlMulla', role:'Chief Executive Officer, Emirates Islamic'},
  {name:'Aazar Ali Khwaja', role:'Group Head, International'},
  {name:'Eman Abdulrazzaq', role:'Group Chief Operating Officer / Group Chief Human Resources Officer'},
  {name:'Miguel Rio-Tinto', role:'Group Chief Digital and Information Officer'},
  {name:'Neeraj Makin', role:'Group Head, Strategy, Analytics and Venture Capital'},
  {name:'Victor Matafonov', role:'Group Chief Compliance Officer'},
  {name:'Manoj Chawla', role:'Group Chief Risk Officer'},
  {name:'Vijay Bains', role:'Chief Sustainability Officer and Group Head of ESG'}
] AS p
MERGE (person:Person {name: p.name})
  SET person.role = p.role
WITH person
MATCH (org:Organization {name:'Emirates NBD Bank (P.J.S.C.)'})
MERGE (person)-[:HOLDS_ROLE {title: person.role}]->(org);

// ---------- Strategic priorities ----------
UNWIND [
  'Customer-centric approach',
  'Leading Bank in the UAE',
  'Internationally diversified institution',
  'Investing in future potential',
  'Most innovative Bank',
  'People and sustainability'
] AS pr
MERGE (priority:StrategicPriority {name: pr})
WITH priority
MATCH (org:Organization {name:'Emirates NBD Bank (P.J.S.C.)'})
MERGE (priority)-[:PRIORITY_OF]->(org);

// ---------- Key 2025 events / milestones ----------
UNWIND [
  {id:'evt_rbl', name:'Definitive agreement to acquire majority stake in RBL Bank, India (~USD 3bn)', category:'M&A'},
  {id:'evt_sukuk', name:'Emirates Islamic issues world\u2019s first Sustainability-Linked Financing Sukuk (USD 500mn)', category:'Sustainable Finance'},
  {id:'evt_issb', name:'First bank globally to publish ISSB report aligned with IFRS S1 and S2', category:'ESG/Disclosure'},
  {id:'evt_1trillion', name:'Total assets surpass AED 1 trillion for the first time', category:'Milestone'},
  {id:'evt_dimsum', name:'Return to Dimsum bond market after a decade with CNH 1bn issuance', category:'Capital Markets'},
  {id:'evt_gold', name:'First UAE bank to offer in-house branded gold bullion (Emirates NBD Gold)', category:'Product Launch'}
] AS e
MERGE (event:Event {id: e.id})
  SET event.name = e.name, event.category = e.category, event.year = 2025
WITH event
MATCH (org:Organization {name:'Emirates NBD Bank (P.J.S.C.)'})
MERGE (event)-[:EVENT_OF]->(org);

// ---------- Awards (representative set) ----------
UNWIND [
  {id:'aw_1', name:'Middle East\u2019s Best Bank', body:'Euromoney Awards for Excellence 2025', target:'Emirates NBD Bank (P.J.S.C.)'},
  {id:'aw_2', name:'UAE\u2019s Best Bank', body:'Euromoney Awards for Excellence 2025', target:'Emirates NBD Bank (P.J.S.C.)'},
  {id:'aw_3', name:'Middle East\u2019s Best Bank for ESG', body:'Euromoney Awards for Excellence 2025', target:'Emirates NBD Bank (P.J.S.C.)'},
  {id:'aw_4', name:'Middle East\u2019s Best ESG Deal', body:'Euromoney Awards for Excellence 2025', target:'Emirates NBD Bank (P.J.S.C.)'},
  {id:'aw_5', name:'UAE\u2019s Best Bank for ESG', body:'Euromoney Awards for Excellence 2025', target:'Emirates NBD Bank (P.J.S.C.)'},
  {id:'aw_6', name:'Middle East\u2019s Best Bank for SMEs', body:'Euromoney Awards for Excellence 2025', target:'Emirates NBD Bank (P.J.S.C.)'},
  {id:'aw_7', name:'Middle East\u2019s Best Bank for Customer Experience', body:'Euromoney Awards for Excellence 2025', target:'Emirates NBD Bank (P.J.S.C.)'},
  {id:'aw_8', name:'UAE\u2019s Best Investment Bank for ECM', body:'Euromoney Awards for Excellence 2025', target:'Emirates NBD Capital'},
  {id:'aw_9', name:'Middle East\u2019s Best FX Bank', body:'Euromoney Foreign Exchange Awards 2025', target:'Emirates NBD Bank (P.J.S.C.)'},
  {id:'aw_10', name:'UAE\u2019s Best FX Bank', body:'Euromoney Foreign Exchange Awards 2025', target:'Emirates NBD Bank (P.J.S.C.)'},
  {id:'aw_11', name:'World\u2019s Best Islamic Digital Bank', body:'Euromoney Islamic Finance Awards 2025', target:'Emirates Islamic Bank (P.J.S.C.)'},
  {id:'aw_12', name:'Middle East\u2019s Best Islamic Digital Bank', body:'Euromoney Islamic Finance Awards 2025', target:'Emirates Islamic Bank (P.J.S.C.)'},
  {id:'aw_13', name:'UAE\u2019s Best Islamic Digital Bank', body:'Euromoney Islamic Finance Awards 2025', target:'Emirates Islamic Bank (P.J.S.C.)'},
  {id:'aw_14', name:'Islamic Retail Bank of the Year \u2013 Middle East', body:'The Banker\u2019s Islamic Banking Awards 2025', target:'Emirates Islamic Bank (P.J.S.C.)'},
  {id:'aw_15', name:'Best Islamic Corporate Bank in the World', body:'Global Finance \u2013 Best Islamic Financial Institutions Awards 2025', target:'Emirates Islamic Bank (P.J.S.C.)'},
  {id:'aw_16', name:'NAFIS Diamond Award for Emiratisation', body:'The NAFIS Award 2025', target:'Emirates NBD Bank (P.J.S.C.)'},
  {id:'aw_17', name:'Ranked #1 in Tier One Capital', body:'The Banker\u2019s Top 1000 Global Bank Rankings 2025', target:'Emirates NBD Bank (P.J.S.C.)'},
  {id:'aw_18', name:'Ranked #1 by Total Assets in the UAE', body:'The Banker\u2019s Top 1000 Global Bank Rankings 2025', target:'Emirates NBD Bank (P.J.S.C.)'}
] AS a
MERGE (award:Award {id: a.id})
  SET award.name = a.name, award.awarding_body = a.body, award.year = 2025
WITH award, a
OPTIONAL MATCH (org:Organization {name: a.target})
OPTIONAL MATCH (sub:Subsidiary {name: a.target})
FOREACH (_ IN CASE WHEN org IS NOT NULL THEN [1] ELSE [] END | MERGE (award)-[:AWARDED_TO]->(org))
FOREACH (_ IN CASE WHEN sub IS NOT NULL THEN [1] ELSE [] END | MERGE (award)-[:AWARDED_TO]->(sub));

// =====================================================================
// Optional: text-chunk nodes for retrieval (RAG) without any LLM.
// Add report sections as chunks; you can later embed them with any
// embedding model (even a small local sentence-transformers model,
// which is NOT a generative LLM) and index with Neo4j vector index.
// Example pattern (fill in `text` per chunk yourself or via a script):
// -----------------------------------------------------------------
// MERGE (c:Chunk {id: 'chunk_001'})
//   SET c.text = '...', c.section = 'Chairman statement', c.page = 4
// WITH c
// MATCH (org:Organization {name:'Emirates NBD Bank (P.J.S.C.)'})
// MERGE (c)-[:MENTIONS]->(org);
// =====================================================================
