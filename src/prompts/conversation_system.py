from __future__ import annotations

import json

from src.models.schemas import InjectedConflict, UserProfile
from src.pipeline.phase1_document_prep import DocumentContext
from src.prompts.base import PromptBuilder
from src.scenarios.base import BaseScenario


# ---------------------------------------------------------------------------
# Scenario-specific context descriptions
# ---------------------------------------------------------------------------

SCENARIO_DESCRIPTIONS: dict[str, str] = {
    "1": """\
## Scenario Background — Study Group: Country Research on Ficlandia

You are generating a conversation for one member of a 4-person study group researching
the fictional Republic of Ficlandia for a graduate-level comparative politics / area
studies course. The group shares access to seven comprehensive reference documents
covering Ficlandia's history, geography, demographics, economics, culture, governance,
and current challenges. Each student uses the SAME AI assistant independently — they
do NOT see each other's conversations.

### About the Documents

These seven documents contain ALL factual content for this scenario. Every historical
date, statistic, place name, institution, policy, cultural practice, and economic figure
that appears in the conversation MUST come from these documents — not from the LLM's
general knowledge. Ficlandia is entirely fictional.

1. **History (History.pdf)**: Chronicles Ficlandia from 3200 BCE (Alderi city-states) through
   the modern republic. Covers the Korath Empire, Vorennian invasions, medieval unification
   under Queen Aelindra, Golden Age crusades, constitutional crises, colonial expansion,
   Enlightenment reforms, industrial revolution, the 1919 republic, WWII occupation, and
   post-war democracy. Key events: Battle of Stormhaven (923 CE), Compromise of Valdren
   (1423), Edict of Tolerance (1520), Constitution of 1919.

2. **Geography (Geography.pdf)**: Physical and human geography. Total area 487,320 km².
   Greyhorn Mountains (Mount Alderon 4,217 m), Central Lowlands, Southern Highlands,
   Havenmere Peninsula, Stormhaven Archipelago (347 islands). River Alderen (1,180 km),
   Lake Vethara (892 km²). Climate zones, natural resources, regional landscapes. Detailed
   tables (Table 2.1 etc.) with geographic facts.

3. **Demographics (Demographics.pdf)**: Population 48,762,310 (2024 census). Age structure,
   fertility rates (1.62), life expectancy (M: 79.4, F: 84.1), ethnic composition (Ficlandian
   78.2%, Thalmorian 6.8%, etc.), languages (Ficlandian official, Thalmorian, Vorennian),
   religions, urbanization (78.3%), immigration trends. Detailed tables (Table 4.1 etc.).

4. **Economics (Economics.pdf)**: GDP $1.62 trillion (2024), GDP per capita $33,230. Social
   market economy. Sector breakdown: services 68%, industry 27%, agriculture 5%. Trade,
   fiscal policy, Central Bank, labor markets, innovation ecosystem. Key indicators in
   Table 7.1. Major companies and industrial specializations.

5. **Culture (Culture.pdf)**: The concept of verath (communal harmony). Literary tradition
   (Ficlandiad epic, Katarina Selthenberg Nobel 1978), visual arts, music (drehspel,
   Ficlandian National Orchestra), cinema, cuisine (Alderstew national dish, Greyhorn
   Raclette), festivals (Midsommar, Harvest Festival), sports. Regional cultural variations.

6. **Civil Society (Civil_Society.pdf)**: Constitutional framework (Constitution of 1919,
   amended 1998). The Valthing (bicameral: People's Chamber 350 seats, Senate 100 seats).
   President, Chancellor, Council of Ministers (18 portfolios in Table 3.1). Judicial branch,
   regional government (15 regions), electoral system (mixed-member proportional, 5% threshold).
   Eternity Clause (Article 79).

7. **Current Scenario & Future (Current_Future.pdf)**: Ficlandia in 2024. Challenges: climate
   change (Green Ficlandia Plan, net-zero by 2045), aging demographics (Demographic Strategy
   2040), digital transformation (Digital Ficlandia 2030), housing affordability, social
   cohesion. Foreign policy, defense (1.8% GDP). Ficlandia 2050 national vision.

### What Makes This Scenario Special
- All students are EQUAL PEERS — no authority hierarchy
- Each student has different expertise (history/politics vs economics vs culture vs geography)
- Students have MISCONCEPTIONS that may be corrected by the AI
- The AI must cite specific sections, tables, dates, and statistics from the documents
- Different students focus on different documents and notice different details
- Ficlandia is FICTIONAL — all facts must come from the documents, not real-world knowledge""",

    "2": """\
## Scenario Background — Municipal Committee: Solid Waste Management Reform

You are generating a conversation for one member of a 4-person municipal committee
reviewing FicTown's solid waste management crisis. The committee has a clear authority
hierarchy reflecting municipal governance. Each member uses the SAME AI assistant
independently to analyze the crisis from their domain perspective.

### About the Documents

These six documents represent FicTown Municipal Corporation's internal reports on its
solid waste management operations. Every statistic, budget figure, health metric,
contractor KPI, citizen complaint, and policy proposal in the conversation MUST come
from these documents.

1. **Municipal Ground Audit (Municipal_Ground_Audit.pdf)**: Comprehensive field assessment.
   FicTown generates 3,420 MTPD of waste. Collection efficiency: 78.3% (target: 95%).
   Segregation at source: 41%. Three active landfills with 28 months remaining capacity.
   Cost-per-tonne: INR 3,102. Total SWM budget: INR 387.4 crore. 12 zones, 186 wards.

2. **Public Health Impact Study (Public_Health_Impact_Study.pdf)**: Epidemiological
   assessment linking waste sites to health outcomes. Residents within 1 km of Pallavaram
   landfill show 2.7x respiratory illness and 3.1x waterborne disease increase.
   Economic burden: INR 48.6 crore/year. Data from 42 PHCs, 8 CHCs, 3 hospitals.

3. **Contractor Performance Review (Contractor_Performance_Review.pdf)**: Evaluates three
   private contractors (CleanSweep, EcoCollect, GreenHaul) covering 65% of city area.
   Performance scorecards with KPIs. Mixed results: one meets expectations, two underperform.
   Covers fleet compliance, manpower, citizen complaints, and penalty assessments.

4. **Citizen Grievance Summary (Citizen_Grievance_Summary.pdf)**: 22,660 complaints in
   FY 2025-26 (28.4% increase). Multi-channel: helpline, mobile app (142,000 users),
   web portal, ward offices. Category-wise breakdown, zone-wise hotspots, resolution
   performance, qualitative analysis with representative complaints, social media monitoring.

5. **Zonal Budget Utilization (Zonal_Budget_Utilization.pdf)**: Budget analysis for SWM
   across 12 zones. Total SWM budget: INR 412 crore (14.8% of FMC total). Allocated vs
   actual expenditure by head and zone. Revenue recovery, variance analysis, inter-city
   benchmarking, fiscal recommendations. High recurring costs, capital investment shortfalls.

6. **Proposed Technical Upgrade (Proposed_Technical_Upgrade.pdf)**: Proposal for a 2,000-TPD
   Waste-to-Energy plant and 600-TPD composting facility. Would raise processing capacity
   from 22% to ~98%. Technology selection, site assessment, financial analysis, environmental
   impact, implementation timeline. References findings from all other documents.

### What Makes This Scenario Special
- HIERARCHICAL authority reflecting municipal governance roles
- Cross-references between documents reveal systemic failures (health ↔ landfill ↔ budget ↔ contractor)
- Data-heavy documents with specific INR figures, tonnages, percentages, and zone-level breakdowns
- The Proposed Technical Upgrade document synthesizes evidence from all others
- Real tensions between cost constraints, public health urgency, and contractor accountability""",

    "3": """\
## Scenario Background — Product Strategy: Ficduct Enterprise Platform Review

You are generating a conversation for one member of a 4-person cross-functional team
reviewing the full product dossier for Ficduct, a next-generation enterprise integration
platform (iPaaS). Each reviewer is a domain expert examining the product from their
perspective. Each uses the SAME AI assistant independently.

### About the Documents

These seven documents constitute Ficduct's complete product and business dossier. Every
metric, feature, architecture detail, financial figure, and strategic claim in the
conversation MUST come from these documents.

1. **Product Overview (Product_Overview.pdf)**: Comprehensive introduction to Ficduct v1.0.
   Core capabilities: Connect, Flow, Stream, Insight, Vault. Use cases, competitive landscape
   (vs MuleSoft, Workato, Boomi), vision and outlook. Launched Q1 2026.

2. **Technical Architecture (Technical_Architecture.pdf)**: Microservices on Kubernetes.
   gRPC inter-service communication. Data pipeline design, security & compliance (SOC 2,
   HIPAA), API spec, observability & monitoring, performance & reliability metrics,
   engineering roadmap.

3. **Marketing & Sales (Marketing_Sales.pdf)**: iPaaS market valued at $6.8B (2025),
   CAGR 18.2%. Brand positioning, go-to-market strategy, sales pipeline & metrics,
   campaign performance, content & thought leadership, channel strategy, customer
   acquisition funnel, marketing roadmap.

4. **Financial Report (Financial_Report.pdf)**: ARR $14.2M (210% YoY growth). Q1 2026
   revenue $3.8M ($3.55M subscription, $250K services). Total raised: $42M. Revenue
   breakdown, cost structure, unit economics, cash flow & burn rate, financial projections,
   key risks.

5. **Customer Feedback (Customer_Feedback.pdf)**: NPS analysis, satisfaction by product
   area, qualitative feedback themes, customer testimonials, competitive perception,
   churn analysis, Customer Advisory Board insights, recommendations.

6. **Strategy & Operations (Strategy_Operations.pdf)**: Four strategic pillars for 2026-2027.
   Organizational structure, product lifecycle management, operational KPIs & dashboards,
   process & governance, talent & culture, risk management framework, operational roadmap.

7. **V2 Planning (Future_Planning.pdf)**: Ficduct v2 vision for GA in Q1 2027. V1
   retrospective, architecture evolution, UX & design improvements, AI capabilities,
   new feature roadmap, migration & compatibility, timeline & milestones.

### What Makes This Scenario Special
- CROSS-FUNCTIONAL authority: each expert owns their domain's assessment
- Technical, financial, market, and customer perspectives may conflict on priorities
- No single expert can override another's domain assessment
- Documents cross-reference each other (e.g., customer feedback informs v2 planning)
- Mix of quantitative metrics and qualitative assessments""",

    "4": """\
## Scenario Background — M&A Due Diligence: Project Ficguard

You are generating a conversation for one member of a 4-person M&A team conducting
due diligence on the acquisition of Veloxra Labs Corp. by Ficguard Industries Inc.
(Project Ficguard). There are two adversarial sides: Buyer team (Ficguard) vs Seller
team (Veloxra). Each person uses the SAME AI assistant independently to prepare their
position. The AI must NEVER leak one side's strategy to the other.

### About the Documents

These eight documents are the deal materials for Project Ficguard. Every valuation
figure, legal term, financial metric, and strategic claim in the conversation MUST come
from these documents. Each side will SELECTIVELY cite data that supports their position.

1. **Term Sheet / Letter of Intent (Term_Sheet_Letter_of_Intent.pdf)**: Non-binding LOI
   dated Sep 22, 2025. Base purchase price: $485M (stock purchase). Working capital
   adjustment, earnout components, escrow provisions, exclusivity period, conditions.

2. **Draft Acquisition Agreement (Draft_Acquisition_Agreement.pdf)**: Stock Purchase
   Agreement v2.4. Articles I-XI covering definitions, purchase/sale, reps & warranties
   (Seller Article III, Buyer Article IV), covenants, indemnification, closing conditions.

3. **Independent Valuation Report (Independent_Valuation_Report.pdf)**: By Drexmont Hale
   Advisory. Fair market value assessment of Veloxra. Veloxra: founded 2018, 412 FTEs,
   FY2024 revenue $106.2M, ARR $118.4M. DCF analysis, comparable companies, precedent
   transactions, scenario analysis. Enterprise AI/ML platform market TAM $68B by 2028.

4. **Financial Due Diligence (Financial_Due_Diligence_Report.pdf)**: By Hargrove & Sinclair.
   Revenue analysis (42% CAGR 2022-2024), customer concentration risk, revenue recognition
   practices. FY2024 revenue $106.2M, YTD Sep 2025 $98.7M.

5. **Legal Counsel Memorandum (Legal_Counsel_Memorandum.pdf)**: By Pemberton & Ashcroft.
   Legal due diligence findings: active patent litigation ($4.5M exposure), open-source
   compliance gaps, key-person IP dependencies, change-of-control provisions, missing
   non-compete agreements.

6. **Negotiation Strategy Memo (Negotiation_Strategy_Memo.pdf)**: Internal Ficguard strategy.
   Build-vs-buy: organic $85-120M over 30-36 months. Talent value: $25-35M in signing
   bonuses. Cross-sell opportunity: $45-60M over 36 months. Enterprise value range:
   Buyer $440-548M vs estimated Seller $550-650M.

7. **IP & Technology Assessment (IP_Technology_Assessment.pdf)**: Technical due diligence
   by Ficguard CTO Office + Ironclad Advisors. 2.4M LOC across modules (DataForge,
   ModelForge, etc.). Code coverage, tech debt grades, architecture review, security
   penetration testing.

8. **Regulatory & Compliance Review (Regulatory_Compliance_Review.pdf)**: HSR filing
   required ($280K fee). Antitrust analysis, data privacy, export controls, government
   contract regulations, cross-border considerations.

### What Makes This Scenario Special
- ADVERSARIAL dual sides: buyer vs seller, NEVER merged
- Both sides access the SAME documents but SELECTIVELY cite favorable data
- The AI must help each side build THEIR case, not a neutral analysis
- Internal strategy documents (Negotiation Strategy Memo) should inform buyer-side analysis
- Legal, financial, technical, and regulatory perspectives all feed into deal positions""",

    "5": """\
## Scenario Background — Incident Response: Cascading Payment Failures

You are generating a conversation for one of four analysts handling a major incident
(INC-2026-04417) at Meridian Financial Technologies. The incident involves cascading
payment-processing failures across the settlement engine, gateway routing layer, and
merchant reconciliation pipeline. Analysts work SEQUENTIALLY across shifts. Each analyst
uses the SAME AI assistant independently during their shift.

### About the Documents

These five documents represent the full incident record. Every log entry, metric, alert,
configuration value, and diagnosis in the conversation MUST come from these documents.
The evidence accumulates across shifts — later analysts have access to more packets.

1. **Main Incident Report (Main_Incident_Report.pdf)**: Defines the full scenario, benchmark
   rules, analyst workflow, and evidence model. Meridian serves 2,400+ merchants. Incident
   opened 2026-03-15 02:14 UTC, closed 2026-03-16 19:48 UTC (~41.5 hours). Severity
   escalated from SEV-1 to SEV-0.

2. **Log Packet 1 — Shift A (Log_Packet1.pdf)**: First evidence (02:14-08:00 UTC, Mar 15).
   Initial PagerDuty alerts, settlement engine p99 latency spike to 4,870ms. ~412K
   transaction timeouts, 71.3% success rate at nadir. PgBouncer pool-size increase deployed
   (CHG-88921) with partial improvement then regression. 6 red flags raised (A-RF1 to A-RF6).

3. **Log Packet 2 — Shift B (Log_Packet2.pdf)**: New evidence from Shift B (08:00-16:00
   UTC, Mar 15). DB-pool theory weakens as non-database service shows symptoms. TLS
   handshake failures emerge. Connection leak mechanism traced. Success rate ~88% but below
   SLA. 3 Shift A flags resolved, 3 new flags raised (B-RF1 to B-RF3).

4. **Log Packet 3 — Shift C (Log_Packet3.pdf)**: Shift C evidence (16:00-00:00 UTC, Mar
   15/16). ROOT CAUSE IDENTIFIED: CLS SAN-mismatch in mTLS cert rotation
   (cls-rotate-20260315-0100). 119/312 certificates (38.1%) deployed with wrong SAN
   ('*.mesh.internal' instead of '*.mesh.local'). SAN validation was in soft-fail mode.
   Incident escalated to SEV-0. Emergency re-rotation initiated (CHG-89104).

5. **Log Packet 4 — Shift D (Log_Packet4.pdf)**: Final evidence (00:00-19:48 UTC, Mar 16).
   Remediation completes: all 119 certs replaced. Root cause code change identified
   (PR #4781 race condition). Soft-fail mode origin traced. All 11 red flags resolved.
   Incident formally closed with CONFIRMED status.

### What Makes This Scenario Special
- SEQUENTIAL handoffs: Analyst 1 → 2 → 3 → 4 (expanding evidence corpus each shift)
- Diagnosis EVOLVES: DB pool exhaustion (Shift A) → TLS anomaly (Shift B) → CLS SAN mismatch (Shift C) → PR #4781 race condition (Shift D)
- Each analyst reads ALL previous packets plus their new one
- Red flags are tracked across shifts (A-RF1 through C-RF2, all resolved by Shift D)
- Temporal reasoning: earlier interpretations must be updated as new evidence arrives
- The full audit trail of evolving diagnosis must be preserved""",
}


# ---------------------------------------------------------------------------
# Document description builders (scenario-specific)
# ---------------------------------------------------------------------------

DOCUMENT_DESCRIPTIONS: dict[str, dict[str, str]] = {
    "1": {
        "History of Ficlandia": (
            "Comprehensive history from 3200 BCE to the modern republic. Covers Alderi "
            "city-states, Korath Empire (Thalvorian Codex), Vorennian invasions, medieval "
            "unification (Queen Aelindra, Battle of Stormhaven 923 CE), Golden Age, "
            "constitutional crises (Compromise of Valdren 1423), Wars of Faith, colonial "
            "expansion, Enlightenment, industrialization, 1919 republic, WWII occupation "
            "(1940-1943), and post-war democracy. Cite specific dates, rulers, and events."
        ),
        "Geography of Ficlandia": (
            "Physical and human geography. 487,320 km² total area. Greyhorn Mountains "
            "(Mount Alderon 4,217 m), Central Lowlands, Southern Highlands, Havenmere "
            "Peninsula, Stormhaven Archipelago (347 islands). River Alderen 1,180 km. "
            "Climate zones, natural resources, infrastructure. Key data in Table 2.1."
        ),
        "Demographics of Ficlandia": (
            "Population 48,762,310 (2024). Median age 41.8, fertility rate 1.62. Ethnic "
            "groups (Ficlandian 78.2%, Thalmorian 6.8%, etc.), languages, religions. "
            "Urbanization 78.3%. Immigration trends, age structure, vital statistics. "
            "Key data in Tables 4.1 and 4.2."
        ),
        "Economics of Ficlandia": (
            "GDP $1.62 trillion, per capita $33,230. Social market economy. Services 68%, "
            "industry 27%, agriculture 5%. Trade, fiscal policy, Central Bank, labor markets, "
            "innovation. Macroeconomic indicators in Table 7.1. Sector breakdown, major "
            "companies, and industrial specializations."
        ),
        "Culture of Ficlandia": (
            "Cultural identity rooted in verath (communal harmony). Literary tradition: "
            "Ficlandiad epic (1340 CE), Nobel laureate Selthenberg (1978). Visual arts, "
            "music (drehspel, kulning), cinema, cuisine (Alderstew, Greyhorn Raclette), "
            "festivals (Midsommar, Harvest Festival), sports. Regional variations."
        ),
        "Civil Society of Ficlandia": (
            "Constitution of 1919 (amended 1998). Valthing: People's Chamber (350 seats, "
            "mixed-member proportional) + Senate (100 seats). President (ceremonial), "
            "Chancellor (head of government), Council of Ministers (18 portfolios, Table 3.1). "
            "Judiciary, regional government (15 regions). Eternity Clause (Article 79)."
        ),
        "Current Scenario and Future of Ficlandia": (
            "Ficlandia today: HDI rank 9th globally. Current government: center-left coalition "
            "led by Chancellor Korvaldsen. Challenges: climate change (Green Ficlandia Plan, "
            "net-zero by 2045), aging society (Demographic Strategy 2040), digital transformation "
            "(Digital Ficlandia 2030), housing affordability, social cohesion, foreign policy. "
            "Ficlandia 2050 national vision."
        ),
    },
    "2": {
        "Municipal Ground Audit": (
            "Field assessment of FicTown's solid waste operations. 3,420 MTPD waste generated. "
            "Collection efficiency 78.3% (target 95%). Segregation at source 41%. Three "
            "active landfills, 28 months remaining capacity. Cost-per-tonne INR 3,102. "
            "12 zones, 186 wards. Detailed tables and zone-wise breakdowns."
        ),
        "Public Health Impact Study": (
            "Epidemiological study linking waste sites to health outcomes. 2.7x respiratory "
            "illness and 3.1x waterborne disease increase within 1 km of Pallavaram landfill. "
            "Economic burden INR 48.6 crore/year. Data from 42 PHCs, 8 CHCs, 3 hospitals. "
            "Exposed vs control population comparison."
        ),
        "Contractor Performance Review": (
            "Evaluates three private contractors (CleanSweep, EcoCollect, GreenHaul) covering "
            "65% of city area. Performance scorecards against contractual KPIs. Fleet and "
            "manpower compliance, citizen complaint correlation, penalty assessments. "
            "One meets expectations, two underperform."
        ),
        "Citizen Grievance Summary": (
            "22,660 complaints in FY 2025-26 (28.4% increase YoY). Multi-channel collection: "
            "helpline, mobile app (142,000 users), web portal, ward offices. Category-wise "
            "and zone-wise breakdown, resolution performance, representative complaints, "
            "social media monitoring."
        ),
        "Zonal Budget Utilization": (
            "Budget analysis: INR 412 crore total SWM budget (14.8% of FMC total). Allocated "
            "vs actual by head and zone. Revenue recovery, variance analysis, inter-city "
            "benchmarking. High recurring cost growth with capital investment shortfalls. "
            "Contractor payments: INR 142.8 crore (36.9%)."
        ),
        "Proposed Technical Upgrade": (
            "Proposal for 2,000-TPD Waste-to-Energy plant + 600-TPD composting facility. "
            "Would raise processing from 22% to ~98%. Technology selection, site assessment, "
            "financial analysis, environmental impact, timeline. Synthesizes findings from "
            "all companion documents to justify the investment."
        ),
    },
    "3": {
        "Product Overview": (
            "Comprehensive introduction to Ficduct v1.0 enterprise integration platform. "
            "Core capabilities: Connect, Flow, Stream, Insight, Vault. Use cases across "
            "industries. Competitive landscape (MuleSoft, Workato, Boomi). Vision and outlook. "
            "Launched Q1 2026."
        ),
        "Technical Architecture": (
            "Microservices on Kubernetes with gRPC. Data pipeline design, security & compliance "
            "(SOC 2, HIPAA), API specification, observability & monitoring stack, performance "
            "& reliability metrics, engineering roadmap. Infrastructure and deployment details."
        ),
        "Marketing & Sales": (
            "iPaaS market $6.8B (2025), CAGR 18.2%. Brand positioning, go-to-market strategy, "
            "sales pipeline & metrics, campaign performance, content strategy, channel strategy, "
            "customer acquisition funnel, marketing roadmap."
        ),
        "Financial Report": (
            "ARR $14.2M (210% YoY growth). Q1 2026 revenue $3.8M. Total raised $42M. Revenue "
            "breakdown, cost structure, unit economics, cash flow & burn rate, fundraising "
            "history, financial projections, key risks & mitigations."
        ),
        "Customer Feedback": (
            "NPS analysis, satisfaction by product area, qualitative feedback themes, customer "
            "testimonials, competitive perception, churn analysis, Customer Advisory Board "
            "insights, and actionable recommendations."
        ),
        "Strategy & Operations": (
            "Four strategic pillars for 2026-2027. Organizational structure, product lifecycle "
            "management, operational KPIs & dashboards, process & governance, talent & culture, "
            "risk management framework, operational roadmap."
        ),
        "V2 Planning": (
            "Ficduct v2 vision for GA in Q1 2027. V1 retrospective, architecture evolution, "
            "UX & design improvements, AI capabilities, new feature roadmap, migration & "
            "compatibility planning, timeline & milestones."
        ),
    },
    "4": {
        "Term Sheet / Letter of Intent": (
            "Non-binding LOI dated Sep 22, 2025. Stock purchase of 100% of Veloxra equity. "
            "Base price $485M with working capital adjustment, earnout, escrow. Exclusivity "
            "period, closing conditions, binding provisions (Sections 10-14)."
        ),
        "Draft Acquisition Agreement": (
            "Stock Purchase Agreement v2.4 between Ficguard (Buyer) and Veloxra (Seller). "
            "Articles I-XI: definitions, purchase/sale, reps & warranties (Seller Art. III, "
            "Buyer Art. IV), covenants (Art. V), indemnification, closing conditions."
        ),
        "Independent Valuation Report": (
            "By Drexmont Hale Advisory. Fair market value of Veloxra as of Sep 30, 2025. "
            "DCF analysis, comparable public companies, precedent transactions. Veloxra: "
            "FY2024 revenue $106.2M, ARR $118.4M, 412 FTEs, 340+ customers."
        ),
        "Financial Due Diligence Report": (
            "By Hargrove & Sinclair. Revenue analysis (42% CAGR 2022-2024), customer "
            "concentration risk, revenue recognition practices. Historical performance: "
            "FY2022 $52.4M, FY2023 $74.8M, FY2024 $106.2M, YTD Sep 2025 $98.7M."
        ),
        "Legal Counsel Memorandum": (
            "By Pemberton & Ashcroft. Legal due diligence: active patent litigation ($4.5M "
            "exposure), open-source compliance gaps, key-person IP dependencies, "
            "change-of-control provisions in customer contracts, missing non-competes."
        ),
        "Negotiation Strategy Memo": (
            "Internal Ficguard strategy. Build-vs-buy: organic $85-120M over 30-36 months. "
            "Target talent: $25-35M signing bonus value. Cross-sell: $45-60M over 36 months. "
            "EV range: Buyer $440-548M vs estimated Seller $550-650M."
        ),
        "IP & Technology Assessment": (
            "Technical due diligence by Ficguard CTO + Ironclad Advisors. 2.4M LOC review "
            "across modules (DataForge, ModelForge, etc.). Code coverage, tech debt grades, "
            "architecture sessions, infrastructure audit, security pen testing."
        ),
        "Regulatory & Compliance Review": (
            "HSR filing required ($280K fee, 30-day review). Antitrust analysis shows "
            "minimal competitive overlap. Data privacy, export controls, government "
            "contract regulations, cross-border considerations."
        ),
    },
    "5": {
        "Main Incident Report": (
            "Defines the full scenario for INC-2026-04417 at Meridian Financial Technologies. "
            "Cascading payment failures across settlement engine, gateway routing layer, and "
            "merchant reconciliation pipeline. 2,400+ merchants. SEV-1 to SEV-0. Opened "
            "2026-03-15 02:14 UTC, closed 2026-03-16 19:48 UTC (~41.5 hours)."
        ),
        "Log Packet 1 — Shift A": (
            "Shift A evidence (02:14-08:00 UTC, Mar 15). PagerDuty alerts, settlement engine "
            "p99 latency 4,870ms. ~412K transaction timeouts, 71.3% success rate nadir. "
            "PgBouncer pool-size increase (CHG-88921): partial improvement then regression. "
            "6 red flags raised (A-RF1 through A-RF6)."
        ),
        "Log Packet 2 — Shift B": (
            "Shift B evidence (08:00-16:00 UTC, Mar 15). DB-pool theory weakens as "
            "non-database service symptomatic. TLS handshake failures emerge. Connection "
            "leak mechanism traced. Success rate ~88%. 3 Shift A flags resolved, 3 new "
            "raised (B-RF1 to B-RF3)."
        ),
        "Log Packet 3 — Shift C": (
            "Shift C evidence (16:00-00:00 UTC, Mar 15/16). ROOT CAUSE: CLS SAN-mismatch "
            "in mTLS cert rotation. 119/312 certs (38.1%) got '*.mesh.internal' instead of "
            "'*.mesh.local'. Soft-fail mode allowed deployment. SEV-0 escalation. Emergency "
            "re-rotation CHG-89104 initiated."
        ),
        "Log Packet 4 — Shift D": (
            "Shift D evidence (00:00-19:48 UTC, Mar 16). All 119 certs replaced. Root cause "
            "code: PR #4781 race condition. Soft-fail mode origin traced. All 11 red flags "
            "(A-RF1-6, B-RF1-3, C-RF1-2) resolved. Incident CONFIRMED and closed."
        ),
    },
}


def _get_scenario_description(scenario_id: str) -> str:
    """Return the scenario-specific context description."""
    return SCENARIO_DESCRIPTIONS.get(scenario_id, "")


def _get_document_descriptions(scenario_id: str, doc_context: DocumentContext) -> str:
    """Build annotated document descriptions for the scenario."""
    descriptions = DOCUMENT_DESCRIPTIONS.get(scenario_id, {})
    lines = ["## Document Descriptions\n"]
    lines.append(
        "The following documents are the SOLE SOURCE OF TRUTH for this conversation. "
        "Read the description of each document carefully — it explains what the document "
        "contains and what to look for.\n"
    )
    for doc_name in doc_context.documents:
        desc = descriptions.get(doc_name, "")
        token_count = doc_context.token_counts.get(doc_name, 0)
        lines.append(f"### {doc_name} (~{token_count} tokens)")
        if desc:
            lines.append(desc)
        lines.append("")
    return "\n".join(lines)


def build_conversation_system_prompt(
    user: UserProfile,
    scenario: BaseScenario,
    session_number: int,
    session_timestamp: str,
    prior_summaries: str,
    target_conflicts: list[InjectedConflict],
    doc_context: DocumentContext,
    turns_this_session: int | None = None,
) -> str:
    config = scenario.config
    if turns_this_session is None:
        turns_this_session = config.get_turns_for_session(session_number)
    persona_section = PromptBuilder.build_user_persona_section(user)
    authority_section = PromptBuilder.build_authority_section(
        user, scenario.get_authority_context()
    )
    document_section = PromptBuilder.build_document_section(doc_context)

    # Scenario description and document descriptions
    scenario_description = _get_scenario_description(config.scenario_id)
    document_descriptions = _get_document_descriptions(
        config.scenario_id, doc_context
    )

    # Session-specific behavior
    session_key = f"session_{session_number}"
    session_behavior = user.session_evolution.get(session_key, "Continue naturally.")

    # Session arc (scenario-level description of what this session is about)
    session_arc_desc = ""
    if hasattr(config, "session_arc") and config.session_arc:
        arc_key = f"session_{session_number}"
        session_arc_desc = config.session_arc.get(arc_key, "")

    # Format conflict seeding
    conflict_section = ""
    if target_conflicts:
        conflict_lines = []
        for c in target_conflicts:
            conflict_lines.append(
                f"- Topic: {c.topic}\n"
                f"  Nature: {c.nature}\n"
                f"  Resolution approach: {c.resolution}"
            )
        conflict_section = "\n".join(conflict_lines)

    prompt = f"""You are generating a synthetic conversation between a human user and an AI assistant
for a research benchmark dataset. The conversation must be realistic, persona-consistent,
and — most critically — GROUNDED ENTIRELY in the provided source documents.

{scenario_description}

{document_descriptions}

# USER PERSONA

{persona_section}

{authority_section}

# TEMPORAL CONTEXT

Current date/time: {session_timestamp}
This is Session {session_number} of {config.sessions_per_user}.
The user has been working on this material since {config.timeline.start_date}.

## Session Arc (what this session is about in the overall narrative)
{session_arc_desc if session_arc_desc else "Continue the ongoing work."}

## Session-Specific Behavior for This User
{session_behavior}

# PRIOR SESSION CONTEXT

{prior_summaries if prior_summaries else "This is the FIRST session. No prior context exists. The user is approaching the material for the first time."}

# CONFLICT SEEDING (INTERNAL — do not reference directly)

The following disagreements should emerge NATURALLY from the user's persona, biases,
and how they read the documents. Do NOT force these or mention them meta-textually.
Let them arise organically from the user asking questions or making claims consistent
with their expertise, biases, and document reading pattern:
{conflict_section if conflict_section else "No specific conflicts to seed in this session."}

# CONTINUITY REQUIREMENTS

- The user may reference things they discussed in prior sessions.
- The user's understanding should BUILD across sessions — not restart from scratch.
- Opinions and positions should EVOLVE: the user can change their mind when shown evidence.
- If the user held a position in a prior session, they should reference it
  (reinforcing, updating, or correcting it).
- The user's emotional state and confidence should match the session_evolution description.

# CRITICAL RULES — DOCUMENT GROUNDING

This is the MOST IMPORTANT requirement. Violations will invalidate the generated data.

## For AI ASSISTANT responses:
1. EVERY factual claim must be traceable to a specific location in the provided documents.
2. Use EXPLICIT citations: reference specific section numbers, theorem numbers, table
   numbers, page references, slide numbers, clause numbers, log timestamps, or other
   document-internal identifiers that appear in the source text.
3. When explaining concepts, quote or closely paraphrase the document text — do NOT
   rephrase using general LLM knowledge.
4. If the documents contain specific numbers, formulas, parameter names, or technical
   details, use the EXACT values from the documents.
5. Do NOT introduce facts, examples, or explanations that are not in the documents.
   Do not bring in external knowledge or real-world comparisons unless those specific
   items are mentioned in the documents.
6. If the user asks about something not covered in the documents, the AI should say so:
   "The documents don't cover that topic" or "I don't see that addressed in our materials."
7. When the documents contain errors or inconsistencies (these are INTENTIONAL), the AI
   should faithfully reflect the document content. If asked to cross-reference, the AI
   should surface the discrepancy.

## For USER messages:
1. The user's questions and statements must reference the documents as their source.
   Users should ask "What does Section X say about..." or "According to the textbook..."
   — NOT "What is [topic]?" in the abstract.
2. When the user states a fact, opinion, or misconception, it should be framed as
   coming from their reading of the documents (or from prior knowledge/courses as
   defined in their persona).
3. The user's reference_style defines HOW they cite documents (formal vs casual vs
   cross-referencing). Follow this strictly.
4. The user's document_reading_pattern defines WHICH parts of the documents they have
   read and which they skip. Respect this — users should NOT ask about sections they
   wouldn't read.
5. Users should show realistic engagement with document content: quoting passages,
   asking about specific figures or tables, noticing details, misreading things, etc.

# GENERATION INSTRUCTIONS

Generate a realistic conversation of exactly {turns_this_session} turns
(1 turn = 1 user message + 1 AI response).

## Conversation Quality Requirements:
1. USER messages MUST match their communication_style, reference_style, and example_utterances.
   Read the example_utterances — they define the user's VOICE, tone, and typical phrasing.
2. USER messages must reflect their knowledge_gaps and misconceptions where relevant.
   If the user has a misconception, they should state it confidently (until corrected).
3. AI ASSISTANT responses must be GROUNDED in the provided documents with specific
   citations. Generic explanations without document references are NOT acceptable.
4. Conversations must feel natural — tangents, follow-ups, confusion, topic changes are
   good. Not every turn needs to be a deep question; some can be clarifications, reactions,
   or meta-comments about the material.
5. The user's reaction_to_corrections should be realistic when the AI pushes back.
6. Session-specific behavior (from session_evolution) should shape the conversation arc.
7. Each message should have a realistic length: user messages vary from 1-3 sentences
   (casual questions) to short paragraphs (detailed analysis); AI responses should be
   substantive but not exhaustive (1-3 paragraphs typically).

## What NOT to do:
- Do NOT generate conversations that read like a textbook or encyclopedia article.
- Do NOT have the AI give long, generic explanations that could come from anywhere.
- Do NOT have the user ask vague questions like "What is X?" without document context.
- Do NOT introduce information from outside the provided documents.
- Do NOT have every turn be a deep technical question — include natural conversation flow.
- Do NOT have the AI volunteer information the user didn't ask about.
- Do NOT break character from the user's defined persona.

# FULL DOCUMENT TEXT

The complete text of all documents follows. Use this as the SOLE source of facts,
numbers, and technical content for the conversation.

{document_section}

# OUTPUT FORMAT

Output a JSON object with a "turns" key containing an array of turn objects.
Each turn object has four fields: turn (integer), role ("user" or "assistant"),
content (string), and timestamp (ISO 8601 string).
Timestamps should advance by approximately 2-5 minutes between messages.

{{"turns": [
  {{"turn": 1, "role": "user", "content": "...", "timestamp": "{session_timestamp}T09:00:00Z"}},
  {{"turn": 1, "role": "assistant", "content": "...", "timestamp": "..."}},
  {{"turn": 2, "role": "user", "content": "...", "timestamp": "..."}},
  {{"turn": 2, "role": "assistant", "content": "...", "timestamp": "..."}},
  ...
]}}"""

    return prompt
