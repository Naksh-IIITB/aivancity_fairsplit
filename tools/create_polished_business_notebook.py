from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "ipynb_submission" / "00_Business_Startup_Deliverable.ipynb"


def cell(cell_type: str, text: str) -> dict:
    return {
        "cell_type": cell_type,
        "metadata": {},
        "source": text.strip().splitlines(keepends=True),
    }


def md(text: str) -> dict:
    return cell("markdown", text)


notebook = {
    "cells": [
        md(
            """
            # FairPrice AI

            ## Business Startup Deliverable

            **Project context:** House Prices - Advanced Regression Techniques  
            **Course:** IIIT Bangalore x Aivancity Hands-On AI Projects  
            **Startup direction:** Explainable house-price prediction for buyers

            FairPrice AI turns a trained house-price prediction model into a buyer-facing decision product. The app estimates a property's fair value, compares it with the asking price, and explains the main factors behind the estimate.
            """
        ),
        md(
            """
            ## 1. Executive Summary

            | Item | Description |
            | --- | --- |
            | Startup name | FairPrice AI |
            | Product type | AI-powered property valuation assistant |
            | Primary customer | First-time home buyers |
            | Core problem | Buyers do not know whether a listing is fairly priced, underpriced, or overpriced |
            | AI solution | Predict fair sale price using property features and explain the value drivers |
            | Main output | Predicted price, pricing verdict, confidence band, and feature explanations |
            | Revenue model | Freemium app, premium valuation reports, B2B agent seats, lender/fintech API |
            | Differentiator | Explainable valuation and buyer decision guidance, not just a raw estimate |
            """
        ),
        md(
            """
            ## 2. Problem and Opportunity

            | Area | Details |
            | --- | --- |
            | Buyer pain point | Home buyers often rely on listing prices set by sellers and agents, which may not reflect fair market value |
            | Decision risk | Overpaying for a house can create long-term financial stress; missing an underpriced listing can mean losing an opportunity |
            | Market gap | Existing valuation tools often feel like black boxes and may not explain why a price is high or low |
            | AI opportunity | A model trained on historical property data can estimate fair value and identify the main drivers of price |
            | Product opportunity | Convert model predictions into simple buying guidance: underpriced, fair, or overpriced |
            """
        ),
        md(
            """
            ## 3. Product Concept

            | Feature | User Benefit |
            | --- | --- |
            | Fair value prediction | Gives the user an independent estimate of what the house should sell for |
            | Pricing verdict | Converts a numerical prediction into a clear label: underpriced, fair, or overpriced |
            | Confidence band | Helps users understand uncertainty instead of treating the estimate as exact |
            | Value-driver explanation | Shows why the model predicts the price, using features such as quality, area, garage, basement, age, and neighborhood |
            | Saved comparisons | Lets buyers compare multiple listings before deciding which homes to visit or bid on |
            | Report export | Produces a shareable report for personal review, family discussion, or agent negotiation |
            """
        ),
        md(
            """
            ## 4. Target Customers

            | Segment | Need | Why They Would Use FairPrice AI |
            | --- | --- | --- |
            | First-time buyers | Avoid overpaying and understand pricing | They lack experience and need simple guidance |
            | Relocating professionals | Compare homes in unfamiliar areas | They need fast valuation support in a new market |
            | Buyer agents | Support client recommendations | They can use reports to justify offer strategy |
            | Mortgage advisors | Pre-check property value risk | They can flag listings where price may be unrealistic |
            | Property investors | Identify undervalued homes | They can screen listings for potential opportunities |
            """
        ),
        md(
            """
            ## 5. Business Model Canvas

            | Canvas Block | FairPrice AI Plan |
            | --- | --- |
            | Customer Segments | First-time buyers, relocating professionals, buyer agents, mortgage advisors, property investors |
            | Value Proposition | Independent, explainable fair-value estimates that help users avoid overpaying and negotiate smarter |
            | Channels | Mobile web app, browser extension, real estate agent partnerships, lender/fintech integrations |
            | Customer Relationships | Freemium self-service, saved watchlists, automated alerts, premium reports, B2B account support |
            | Revenue Streams | Monthly subscriptions, paid valuation reports, B2B agent seats, API fees from lenders or fintech partners |
            | Key Resources | Trained ML model, property dataset, feature engineering pipeline, explainability layer, valuation reports |
            | Key Activities | Data cleaning, model retraining, model monitoring, UX design, partner integrations, compliance reviews |
            | Key Partners | Listing portals, real estate agencies, lenders, data providers, insurance/climate-risk data vendors |
            | Cost Structure | Data licensing, cloud hosting, model development, compliance, customer support, marketing |
            """
        ),
        md(
            """
            ## 6. Competitor Analysis

            | Competitor | Strengths | Weaknesses | FairPrice AI Advantage |
            | --- | --- | --- | --- |
            | Zillow Zestimate | Large brand, large dataset, widely recognized | Often perceived as a black box; strongest in supported US markets | Focus on explainability and buyer decision support |
            | Redfin Estimate | Integrated with listings and brokerage services | Tied closely to Redfin ecosystem; not always available everywhere | Independent valuation across portals |
            | Realtor.com tools | Large consumer reach and simple user experience | Less emphasis on model explanations and negotiation advice | Turns valuation into practical offer guidance |
            | HouseCanary | Strong analytics for professionals | More B2B-oriented and less accessible for everyday buyers | Consumer-first interface with simple verdicts |
            | Local agents | Market knowledge and human judgment | Incentives may not always align with buyer caution | Independent second opinion before making an offer |
            """
        ),
        md(
            """
            ## 7. Positioning

            | Positioning Element | Statement |
            | --- | --- |
            | Category | Explainable AI property valuation assistant |
            | For whom | Home buyers who want confidence before making an offer |
            | Core promise | Know whether a listing is underpriced, fair, or overpriced |
            | Key proof | Machine learning model trained on property attributes and validated with RMSLE |
            | Emotional benefit | Buyers feel more confident, less rushed, and better prepared to negotiate |
            | Tagline | Fair value before you make an offer |
            """
        ),
        md(
            """
            ## 8. PESTEL Analysis

            | Factor | Opportunity | Risk / Challenge | Response |
            | --- | --- | --- | --- |
            | Political | Governments care about housing affordability and market transparency | Housing markets are politically sensitive | Position the product as decision support, not market manipulation |
            | Economic | High interest rates make buyers more cautious and price-sensitive | Housing downturns may reduce transaction volume | Serve both buyers and investors looking for value |
            | Social | First-time buyers want simple, trustworthy tools | Users may over-trust AI predictions | Show confidence bands and clear disclaimers |
            | Technological | Tabular ML, explainability, and AI agents improve valuation tools | Model drift can reduce accuracy over time | Retrain regularly and monitor performance |
            | Environmental | Climate risk can become a major property-value factor | Missing climate data can understate long-term risk | Add flood, heat, wildfire, and insurance-risk features later |
            | Legal | Transparent valuation can help consumers make informed decisions | Data privacy, fair-lending, and appraisal-related regulations may apply | Use licensed data, audit bias, and avoid presenting estimates as official appraisals |
            """
        ),
        md(
            """
            ## 9. SWOT Analysis

            | Strengths | Weaknesses |
            | --- | --- |
            | Clear AI engine from the Kaggle project | Initial dataset is limited to Ames, Iowa |
            | Explainable value proposition | Real deployment needs fresher and broader data |
            | Simple buyer-facing verdict | Prediction uncertainty must be communicated carefully |
            | Low marginal cost per prediction | Trust must be earned against established competitors |

            | Opportunities | Threats |
            | --- | --- |
            | Add climate-aware valuation | Incumbents have stronger data access |
            | Build renovation ROI recommendations | Regulatory scrutiny of automated valuation models |
            | Sell API access to fintechs and lenders | Biased or stale data can harm users |
            | Expand into underserved international markets | Real estate platforms may copy the feature |
            """
        ),
        md(
            """
            ## 10. Ethics and Data Governance

            | Governance Area | Risk | Mitigation |
            | --- | --- | --- |
            | Prediction uncertainty | Users may treat estimates as exact prices | Show confidence bands and explain uncertainty |
            | Bias and fairness | Errors may be larger in some neighborhoods or price segments | Audit model errors by geography and price range |
            | Protected attributes | Direct or proxy variables may create discriminatory outcomes | Avoid protected attributes and test proxy effects |
            | Data privacy | User-entered property or financial information may be sensitive | Minimize collection, encrypt storage, and define retention periods |
            | Model drift | Housing markets change over time | Monitor performance and retrain with recent data |
            | Transparency | Black-box predictions can reduce trust | Provide feature explanations and model documentation |
            """
        ),
        md(
            """
            ## 11. Regulation and Compliance

            | Topic | Requirement / Concern | FairPrice AI Approach |
            | --- | --- | --- |
            | Appraisal boundaries | The product should not claim to be an official appraisal | Use language such as estimate, decision support, and fair-value signal |
            | Consumer protection | Users must not be misled by overly confident claims | Display uncertainty and assumptions clearly |
            | Data licensing | Property and listing data must be legally usable | Use licensed datasets and document data sources |
            | Privacy | Personal or financial data may trigger privacy obligations | Apply GDPR-style consent, access, deletion, and minimization controls |
            | Fair lending | If used by lenders, valuation outputs may affect high-stakes decisions | Add stronger bias audits and human review before lender deployment |
            """
        ),
        md(
            """
            ## 12. Go-To-Market Plan

            | Phase | Goal | Actions | Success Metric |
            | --- | --- | --- | --- |
            | MVP | Validate buyer interest | Launch a simple web app with manual property input and fair/overpriced verdicts | Users complete valuation reports |
            | Pilot | Build trust with real users | Partner with small buyer-agent teams | Repeat usage by agents |
            | Expansion | Improve data and coverage | Add more cities, richer listing data, and climate-risk inputs | Lower prediction error and broader coverage |
            | B2B | Monetize at scale | Offer agent dashboards and lender/fintech API | Paid seats and API usage |
            """
        ),
        md(
            """
            ## 13. Technical Differentiator

            | Technical Component | Business Value |
            | --- | --- |
            | Log-target training | Aligns training with Kaggle RMSLE and reduces outlier impact |
            | Feature engineering | Creates interpretable signals such as total area, house age, bathrooms, and quality-area interaction |
            | Regularized models | Provide strong accuracy with stable behavior on one-hot encoded data |
            | Ensemble / stacking | Improves robustness by combining multiple model families |
            | Explainability layer | Converts predictions into customer-facing reasons |
            | Monitoring plan | Keeps the model trustworthy as housing markets change |
            """
        ),
        md(
            """
            ## 14. Final Recommendation

            FairPrice AI should launch first as a buyer-facing valuation assistant focused on **confidence before making an offer**. The initial product should use the trained regression model to estimate fair value, then present the result through simple verdicts and explanations.

            The strongest next step is to connect the model to a lightweight web app where users enter property details and receive:

            | Output | Purpose |
            | --- | --- |
            | Predicted fair price | Gives the user an independent benchmark |
            | Underpriced / fair / overpriced verdict | Makes the model output easy to act on |
            | Confidence band | Prevents overconfidence |
            | Top value drivers | Builds trust and supports negotiation |
            | Downloadable report | Creates a useful paid feature |
            """
        ),
    ],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}


def main() -> None:
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
