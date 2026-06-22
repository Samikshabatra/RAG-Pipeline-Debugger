"""Test corpus + labeled test queries.

The corpus is a fictional company ("Northwind") knowledge base, written
specifically to create *retrieval confusion*:

  * Near-duplicate topics that embed close together (vacation vs. sick vs.
    parental leave; primary vs. secondary caregiver; Pro vs. Pro Plus plan;
    API v1 vs. v2 limits; refund vs. return vs. warranty).
  * Several queries where the CORRECT chunk exists but is lexically/semantically
    out-shadowed by a similar WRONG chunk -- so a naive retriever ranks the
    wrong one first. These are the genuine failure cases the debugger exists to
    catch (Phases 2-3).

Each test query carries the gold `expected_doc_id` and a short `trap` note
explaining the confusion, so the analyzer's verdicts can be checked by hand.
"""
from __future__ import annotations

from pydantic import BaseModel

from .models import Document

DOCUMENTS: list[Document] = [
    # ----- Leave policies (heavily confusable cluster) -----
    Document(
        doc_id="leave-vacation",
        title="Paid Vacation Policy",
        category="hr",
        text=(
            "Full-time employees accrue 20 days of paid vacation per year, "
            "accrued monthly. Unused vacation days carry over up to a maximum "
            "of 10 days into the next calendar year. Vacation requests must be "
            "submitted at least two weeks in advance through the HR portal."
        ),
    ),
    Document(
        doc_id="leave-sick",
        title="Paid Sick Leave Policy",
        category="hr",
        text=(
            "Employees receive 10 days of paid sick leave per year. Sick leave "
            "does not carry over and resets each January. A doctor's note is "
            "required for any sick absence longer than three consecutive days."
        ),
    ),
    Document(
        doc_id="leave-parental-primary",
        title="Parental Leave — Primary Caregiver",
        category="hr",
        text=(
            "The primary caregiver of a new child is entitled to 16 weeks of "
            "fully paid parental leave, which must be taken within 12 months of "
            "the birth or adoption. Primary caregiver status applies regardless "
            "of gender."
        ),
    ),
    Document(
        doc_id="leave-parental-secondary",
        title="Parental Leave — Secondary Caregiver",
        category="hr",
        text=(
            "The secondary caregiver of a new child is entitled to 4 weeks of "
            "fully paid parental leave, to be taken within 6 months of the birth "
            "or adoption. This is the applicable allowance for most new fathers "
            "unless they are designated as the primary caregiver."
        ),
    ),
    Document(
        doc_id="leave-bereavement",
        title="Bereavement Leave Policy",
        category="hr",
        text=(
            "Employees may take up to 5 days of paid bereavement leave following "
            "the death of an immediate family member, and up to 2 days for an "
            "extended family member."
        ),
    ),

    # ----- Pricing plans (near-duplicate names) -----
    Document(
        doc_id="plan-pro",
        title="Pro Plan",
        category="pricing",
        text=(
            "The Pro plan costs $29 per user per month. It includes up to 5 "
            "projects, 50 GB of storage, and email support with a 24-hour "
            "response time."
        ),
    ),
    Document(
        doc_id="plan-pro-plus",
        title="Pro Plus Plan",
        category="pricing",
        text=(
            "The Pro Plus plan costs $59 per user per month. It includes "
            "unlimited projects, 500 GB of storage, priority chat support, and "
            "advanced analytics. Pro Plus is our most popular plan for growing "
            "teams."
        ),
    ),
    Document(
        doc_id="plan-enterprise",
        title="Enterprise Plan",
        category="pricing",
        text=(
            "The Enterprise plan is custom-priced and billed annually. It adds "
            "single sign-on (SSO), a dedicated account manager, a 99.9% uptime "
            "SLA, and unlimited storage. Contact sales for a quote."
        ),
    ),
    Document(
        doc_id="plan-free",
        title="Free Plan",
        category="pricing",
        text=(
            "The Free plan costs nothing and includes 1 project, 2 GB of "
            "storage, and community-forum support only. It is intended for "
            "individuals evaluating the product."
        ),
    ),

    # ----- API rate limits (v1 vs v2 — classic version confusion) -----
    Document(
        doc_id="api-v1-limits",
        title="API v1 Rate Limits",
        category="api",
        text=(
            "The legacy v1 REST API is limited to 60 requests per minute per "
            "API key. The v1 API is deprecated and will be shut down at the end "
            "of the year; new integrations should not use it."
        ),
    ),
    Document(
        doc_id="api-v2-limits",
        title="API v2 Rate Limits",
        category="api",
        text=(
            "The current v2 REST API allows 600 requests per minute per API "
            "key, with short bursts of up to 1,000 requests per minute. "
            "Exceeding the limit returns an HTTP 429 response with a "
            "Retry-After header."
        ),
    ),
    Document(
        doc_id="api-auth",
        title="API Authentication",
        category="api",
        text=(
            "All API requests must include a bearer token in the Authorization "
            "header. Tokens are created in the dashboard under Settings > API "
            "Keys and can be scoped to read-only or read-write access."
        ),
    ),
    Document(
        doc_id="api-webhooks",
        title="Webhooks",
        category="api",
        text=(
            "Webhooks deliver events to your endpoint via HTTP POST. Failed "
            "deliveries are retried with exponential backoff for up to 24 "
            "hours. Each webhook payload is signed so you can verify "
            "authenticity."
        ),
    ),

    # ----- Money-back policies (refund vs return vs warranty) -----
    Document(
        doc_id="policy-refund",
        title="Subscription Refund Policy",
        category="support",
        text=(
            "Software subscriptions can be refunded within 14 days of purchase "
            "for a full refund, no questions asked. After 14 days, "
            "subscriptions are non-refundable but can be cancelled to stop "
            "future billing."
        ),
    ),
    Document(
        doc_id="policy-return",
        title="Hardware Return Policy",
        category="support",
        text=(
            "Physical hardware can be returned within 30 days of delivery for a "
            "full refund, provided it is in its original packaging. Return "
            "shipping is paid by the customer unless the item arrived damaged."
        ),
    ),
    Document(
        doc_id="policy-warranty",
        title="Hardware Warranty",
        category="support",
        text=(
            "All hardware comes with a 1-year limited warranty covering "
            "manufacturing defects. The warranty does not cover accidental "
            "damage, water damage, or normal wear and tear."
        ),
    ),

    # ----- Cancellation (near-duplicate pair; the WRONG doc carries the
    #       literal phrase "cancellation notice period", so the reranker ranks
    #       it first — a reliable failure trap at tight top_n) -----
    Document(
        doc_id="cancel-monthly",
        title="Cancelling a Monthly Plan",
        category="billing",
        text=(
            "Monthly subscriptions can be cancelled at any time and stop at the "
            "end of the current billing month. No advance notice is required to "
            "cancel a monthly plan."
        ),
    ),
    Document(
        doc_id="cancel-annual",
        title="Cancelling an Annual Plan",
        category="billing",
        text=(
            "Annual subscriptions require 30 days advance written notice to "
            "cancel. Without the required cancellation notice period, the annual "
            "plan automatically renews for another full year."
        ),
    ),

    # ----- Phone support (near-duplicate pair; "phone support response time"
    #       lives in the Enterprise doc = wrong answer, so it out-ranks the
    #       Free-plan doc that *negates* phone support) -----
    Document(
        doc_id="phone-free",
        title="Free Plan Support",
        category="support",
        text=(
            "The Free plan does not include phone support; only community "
            "forums are available for help."
        ),
    ),
    Document(
        doc_id="phone-enterprise",
        title="Enterprise Support",
        category="support",
        text=(
            "Enterprise customers get a dedicated phone support line with a "
            "1-hour response time, available 24/7."
        ),
    ),

    # ----- Security / misc (distractors) -----
    Document(
        doc_id="sec-2fa",
        title="Two-Factor Authentication",
        category="security",
        text=(
            "Two-factor authentication (2FA) can be enabled per account under "
            "security settings. We support authenticator apps (TOTP) and "
            "hardware security keys. SMS-based 2FA is not supported for "
            "security reasons."
        ),
    ),
    Document(
        doc_id="sec-data-retention",
        title="Data Retention",
        category="security",
        text=(
            "Deleted projects are retained in a recoverable state for 30 days, "
            "after which they are permanently erased. Account audit logs are "
            "retained for 12 months."
        ),
    ),
    Document(
        doc_id="onboarding-setup",
        title="Getting Started",
        category="general",
        text=(
            "After signing up, create your first project from the dashboard, "
            "invite teammates by email, and connect your data sources. Most "
            "teams are up and running within an hour."
        ),
    ),
    Document(
        doc_id="support-hours",
        title="Support Hours",
        category="support",
        text=(
            "Standard email support is available Monday to Friday, 9am to 6pm "
            "Pacific Time. Priority and Enterprise customers have access to "
            "24/7 chat support."
        ),
    ),
]


class TestQuery(BaseModel):
    """A labeled test query for evaluating / demoing the pipeline.

    `expected_doc_id` is the gold answer chunk. `trap` documents *why* the
    query is hard, so the analyzer's verdicts can be sanity-checked. `expect_fail`
    flags queries engineered so a naive retriever picks the wrong chunk first --
    these are the cases the debugger should catch as retrieval failures.
    """

    query: str
    expected_doc_id: str
    trap: str = ""
    expect_fail: bool = False


TEST_QUERIES: list[TestQuery] = [
    # --- Clean(ish) cases: the pipeline should get these right ---
    TestQuery(
        query="How many vacation days do full-time employees get per year?",
        expected_doc_id="leave-vacation",
        trap="Competes with sick/parental leave docs in the same cluster.",
    ),
    TestQuery(
        query="What is the rate limit on the current API?",
        expected_doc_id="api-v2-limits",
        trap="v1 doc is lexically similar; 'current' should favor v2.",
    ),
    TestQuery(
        query="Can I get my money back on a software subscription?",
        expected_doc_id="policy-refund",
        trap="'money back' is closer to refund than the hardware return doc.",
    ),
    TestQuery(
        query="How much does the Pro Plus plan cost?",
        expected_doc_id="plan-pro-plus",
        trap="Pro vs Pro Plus near-duplicate names.",
    ),

    # --- Engineered traps: correct chunk likely out-ranked by a similar wrong one ---
    TestQuery(
        query="How many days of paid parental leave do new fathers get?",
        expected_doc_id="leave-parental-secondary",
        trap=(
            "The primary-caregiver doc (16 weeks) embeds strongly to 'parental "
            "leave' and out-shadows the secondary-caregiver doc, which is the "
            "correct answer for most fathers."
        ),
        expect_fail=True,
    ),
    TestQuery(
        query="What are the rate limits for the legacy API?",
        expected_doc_id="api-v1-limits",
        trap=(
            "The v2 doc dominates retrieval for 'rate limits'; 'legacy' is the "
            "only signal pointing at v1."
        ),
        expect_fail=True,
    ),
    TestQuery(
        query="If my new laptop arrives broken, how long do I have to send it back?",
        expected_doc_id="policy-return",
        trap=(
            "'broken' pulls the warranty doc and 'send it back' overlaps the "
            "subscription refund doc; the hardware *return* doc is correct."
        ),
        expect_fail=True,
    ),
    TestQuery(
        query="How much storage and how many projects come with the cheaper paid plan?",
        expected_doc_id="plan-pro",
        trap=(
            "Pro Plus is labeled 'most popular' and mentions storage/projects "
            "prominently, out-ranking the plain Pro plan ('cheaper paid')."
        ),
        expect_fail=True,
    ),
    TestQuery(
        query="How much notice must I give to cancel my monthly subscription?",
        expected_doc_id="cancel-monthly",
        trap=(
            "The annual-cancellation doc literally contains 'cancellation notice "
            "period' (= 30 days, the WRONG answer) so the reranker ranks it #1. "
            "At tight top_n the correct monthly doc is dropped -> retrieval "
            "failure; the system then refuses or answers '30 days'."
        ),
        expect_fail=True,
    ),
    TestQuery(
        query="What is the phone support response time on the Free plan?",
        expected_doc_id="phone-free",
        trap=(
            "'phone support' + 'response time' live in the Enterprise doc "
            "(1-hour, the WRONG answer for Free), which out-ranks the Free-plan "
            "doc that says phone support isn't offered. Dropped at tight top_n."
        ),
        expect_fail=True,
    ),
]

