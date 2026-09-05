# ML Feature Taxonomy & Explanations

## 🕒 What are "Temporal Features"?

**Temporal Features** (Time-based features) are features derived from **timestamps**, **time deltas (differences)**, and **rolling time windows**.

### Why are Temporal Features Critical for Fraud Detection?
1. **Human vs. Bot Speeds**: Honest customers take minutes or hours to browse, read terms, and checkout. Automated scripts or sybil promo hoarders execute operations (registration $\rightarrow$ coupon claim $\rightarrow$ checkout) in **milliseconds or seconds**.
2. **Velocity Spikes**: Fraud syndicates redeem dozens of coupons in short bursts (e.g. 10 redemptions in 1 hour) before fraud defense systems block their IP or device.
3. **Sequence & Recency**: Measuring the exact age of an account or how quickly a promo code was claimed relative to account creation is often the single highest-predictive signal for abuse.

---

## 🔬 Master Feature Reference Table

Here are the key features engineered in our system, categorized by feature type:

| Feature Name | Feature Type | Mathematical / Logical Concept | Why It Detects Fraud |
| :--- | :--- | :--- | :--- |
| **`signup_to_redemption_sec`** | **Temporal (Delta)** | $T_{\text{redemption}} - T_{\text{signup}}$ | Abusers redeem welcome promos within seconds of creating an account. |
| **`cluster_redemptions_1h`** | **Temporal (Window)** | Count of redemptions across linked cluster in rolling 1-hour window | Detects coordinated bot attacks redeeming coupons simultaneously across multiple fake accounts. |
| **`account_age_days`** | **Temporal (Age)** | $T_{\text{as\_of}} - T_{\text{signup}}$ | Sybil throwaway accounts have an age near zero when redeeming welcome offers. |
| **`min_account_creation_delta_minutes`** | **Temporal (Burst)** | Min time gap between account creations in the same identity graph | Identifies bulk account creation scripts creating 50 accounts 30 seconds apart. |
| **`unique_connected_customers`** | **Graph / Network** | Number of distinct customer nodes connected via shared entities | Identifies multi-account rings (e.g., 8 different names sharing 1 laptop). |
| **`max_device_user_count`** | **Graph / Entity** | Max count of accounts tied to the customer's device | Measures device sharing degree (single device operating multiple accounts). |
| **`max_payment_user_count`** | **Graph / Entity** | Max count of accounts tied to the customer's credit card | Identifies stolen credit cards or single organizers funding multiple throwaway accounts. |
| **`max_ip_user_count`** | **Graph / Entity** | Max count of accounts tied to the customer's IP address | Detects residential proxy networks or localized fraud farms. |
| **`shared_entity_ratio`** | **Graph / Density** | $\frac{\text{Shared Entities}}{\text{Total Entities Used}}$ | Percentage of a user's digital footprint that overlaps with existing users. |
| **`cluster_size`** | **Graph / Topological** | Size of connected component $G(V, E)$ | Total size of the sybil syndicate network the user belongs to. |
| **`spend_to_discount_ratio`** | **Financial Ratio** | $\frac{\text{Total Dollars Spent}}{\text{Total Discount Received}}$ | Abusers maximize discount while minimizing real cash spend (near zero ratio). |
| **`high_value_promo_ratio`** | **Behavioral** | $\frac{\text{Redemptions for High Value Offers}}{\text{Total Redemptions}}$ | Promo hoarders ignore standard 5% discounts and exclusively target high-value \$50+ promos. |
| **`order_redemption_rate`** | **Behavioral Ratio** | $\frac{\text{Orders with Promo Code}}{\text{Total Orders}}$ | Honest buyers order without promos; abusers only purchase when a promo code is active ($100\%$ rate). |

---

## 🎯 Pitch Summary Sheet (What to Say)

- **Temporal Features** $\rightarrow$ *"We capture time deltas like registration-to-redemption speed and 1-hour cluster velocity to catch automated bot scripts."*
- **Graph Features** $\rightarrow$ *"We build real-time entity graphs (Device, IP, Address, Payment) to link hidden multi-account networks even when users fake their name or email."*
- **Behavioral Ratios** $\rightarrow$ *"We calculate financial ratios like `spend_to_discount_ratio` to spot one-and-done promo hoarders."*
