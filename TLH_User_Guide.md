# 🧾 Tax-Loss Harvesting Simulator: Start-to-Finish Guide

Welcome to the **Tax-Loss Harvesting Simulator**! This tool is designed to help you explore how selling investments at a loss can strategically offset the capital gains taxes owed on investments you sell at a profit.

This guide will walk you through each section of the simulator, how to enter your data correctly, and how to interpret the results to maximize your tax savings.

---

## 1. Getting Started

When you launch the app, you will see a radio button in the left sidebar labeled **Active Tool**. 
- Select **🧾 Tax-Loss Harvesting Simulator**. 
- Ensure that the main screen tab is also set to **🧾 Tax-Loss Harvesting Simulator**.

---

## 2. Holdings (Batch Mode)

"Batch Mode" is a powerful feature that allows you to input *all* the assets you are considering selling, regardless of whether they are currently up (gains) or down (losses). The tool will automatically fetch live prices and rank every possible "loss vs. gain" pairing to find you the most tax-efficient move.

### How to use Batch Mode:
1. **Enter Your First Holding**: Type the ticker symbol (e.g., `AAPL`), your original purchase price per share, and how many shares you own.
2. **Select Holding Period**: 
   - **< 1 Year (Short-Term)**: You bought this asset less than 365 days ago.
   - **≥ 1 Year (Long-Term)**: You bought this asset over 365 days ago.
   *(Note: This is critical, as short-term gains are taxed at much higher rates than long-term gains).*
3. **Add More Holdings**: Click the **➕ Add Holding** button in the sidebar. You can add as many holdings as you want! Put in your biggest losers and your biggest winners.
4. **Remove Holdings**: Click **🗑️ Remove Last** if you added too many blank fields.

> [!TIP]
> **Batch Mode Strategy**
> Don't just enter one loss and one gain. Enter 3-4 assets you want to sell at a loss, and 3-4 assets you want to sell at a gain. The simulator will do the heavy math of pairing them up to show you exactly which combination saves you the most money.

---

## 3. Tax Configuration

To get accurate estimates, you need to tell the simulator your tax situation.

### Marginal Federal Tax Bracket (%)
This applies primarily to **short-term capital gains**, which are taxed as ordinary income. 

**Examples based on 2024 Single Filer brackets:**
- **12%**: If your total taxable income is ~$11,600 to ~$47,150 (e.g., earning $45,000/year).
- **22%**: If your total taxable income is ~$47,150 to ~$100,500 (e.g., earning $65,000/year). 
- **24%**: If your total taxable income is ~$100,500 to ~$191,950 (e.g., earning $120,000/year).
- **32%**: If your total taxable income is ~$191,950 to ~$243,700.

### Long-Term Capital Gains Rate (%)
This applies to assets you've held for **≥ 1 Year**. Long-term rates are heavily discounted by the IRS to encourage long-term investing.

**Examples based on 2024 Single Filer brackets:**
- **0%**: If your total taxable income is under ~$47,025.
- **15% (Most Common)**: If your total taxable income is between ~$47,025 and ~$518,900 (e.g., earning $65,000/year or $150,000/year).
- **20%**: If your total taxable income is over ~$518,900.

### State Tax Selection
Select your state from the dropdown menu. The tool will automatically populate the estimated state capital gains tax rate. 
- *Example*: If you live in a tax-free state like **Texas, Florida, or Nevada**, the rate will automatically set to **0%**. 
- If you know your specific effective state rate is different from the auto-populated average, you can manually type over the number.

### Include NIIT (3.8%)
The Net Investment Income Tax (NIIT) is an extra 3.8% surcharge applied to investment income for high earners.
- **Leave Unchecked**: If you earn less than $200,000 (Single) or $250,000 (Married Filing Jointly).
- **Check the Box**: If your income exceeds those thresholds.

---

## 4. Running the Analysis & Interpreting Results

Once your holdings and tax brackets are set, click the **🔍 Analyze All Pairings** button.

### 📊 Your Holdings Summary
The tool will fetch live prices from Yahoo Finance and show you a dashboard of your entered assets. You can quickly see which assets are currently at a **📈 GAIN** and which are at a **📉 LOSS**.

### 🏆 Optimal Pairing Rankings
If you entered multiple gains and losses, this table is the magic of Batch Mode. It pairs every single loss asset against every single gain asset and sorts them by **💰 Savings**.
- **Rank 1** is the pairing that yields the highest absolute dollar tax savings.

### 📋 Detailed Breakdown
This section focuses entirely on the #1 ranked pairing. It shows you a side-by-side comparison:
- **Without Harvesting**: What you would owe in taxes if you *only* sold the profitable asset.
- **With Harvesting**: What you owe if you sell *both* the profitable asset and the losing asset together.

### 💡 Key Metrics
- **💰 Maximum Tax Savings**: The exact dollar amount you are keeping out of the IRS's hands by selling the losing asset.
- **📉 Effective Rate Reduction**: How much your overall tax percentage dropped on the sale.
- **📊 Total Proceeds After Tax**: The total cash you will walk away with (Sale of Gain Asset + Sale of Loss Asset - Taxes Owed).

---

## Important Reminders

> [!WARNING]
> **The Wash-Sale Rule**
> The IRS prevents you from claiming a tax loss if you buy a "substantially identical" asset within 30 days before or after the sale. If you use this tool to harvest a loss on `PFE`, do not buy `PFE` again for at least 31 days, otherwise the tax benefit is erased.

> [!CAUTION]
> **Not Official Tax Advice**
> This simulator is a powerful educational and planning tool, but it uses simplified math. It does not account for complex multi-lot cost bases (buying the same stock at 5 different prices over time) or carry-forward losses beyond the current year. Always verify large transactions with a CPA.
