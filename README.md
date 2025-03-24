# 🏡 Mumbai House Price Prediction  

**Leveraging machine learning to predict house prices in Mumbai based on key real estate factors.**  

---  

## 📊 About the Dataset  

### 📍 Context  
Mumbai, the **financial and cultural capital** of India, has one of the most dynamic and **expensive real estate markets** in the country. With a population exceeding **20 million**, housing demand varies significantly across **different localities, property types, and price ranges**.  

This dataset provides valuable insights into **residential property prices in Mumbai**, making it a great resource for **predictive modeling and investment analysis**.  

### 🏠 Dataset Overview  
The dataset contains **detailed information on residential properties**, including **sale prices, property types, locality, area, and construction status**. It enables **data-driven decision-making** for buyers, sellers, investors, and real estate analysts by predicting house prices based on relevant features.  

### 🔑 Key Features in the Dataset  

| **Column Name** | **Description** |
|---------------|----------------|
| **bhk** | Number of bedrooms, hall, and kitchen (e.g., 2BHK, 3BHK). |
| **type** | Type of property: `apartment`, `villa`, `independent house`, `studio apartment`. |
| **locality** | The specific neighborhood or area where the house is located. |
| **area** | Total area of the house in **square feet**. |
| **price** | The selling price of the property. |
| **price_unit** | Price representation unit: `L` (Lakh) or `Cr` (Crore). |
| **region** | The broader region within Mumbai where the property is located. |
| **status** | Construction status: `Ready to move` or `Under Construction`. |
| **age** | Indicates whether the property is `New` or `Resale`. |

---

## 🎯 Why is This Dataset Important?  

- 📊 Helps **buyers & sellers** make informed decisions.  
- 💰 Enables **investors** to identify **profitable real estate opportunities**.  
- 🏘️ Assists **real estate analysts** in tracking **market trends & price fluctuations**.  
- 🤖 Supports **machine learning models** for **accurate house price predictions**.  

---

## 📌 Real-World Applications  

- **🏠 House Price Prediction Models** → Estimate future property values.  
- **📈 Real Estate Market Analysis** → Identify **high-demand** areas.  
- **💰 Investment Strategy Planning** → Detect **profitable locations** for investment.  
- **📊 Rental Price Estimation** → Predict rental values based on similar features.  

---

## 🔑 Key Highlights  

- ✅ **Accuracy:** Achieved an **R-squared score of 85%**, indicating strong predictive power.  
- 📉 **Error Rate:** Low **Mean Absolute Error (MAE)**, ensuring minimal deviation from actual prices.  
- 🏢 **Location Insights:** Identified **high-value neighborhoods** based on price per square foot.  
- 📊 **Feature Importance:** Ranked **area** and **location** as the most influential factors in price determination.  

---

## 💻 Technologies Used  

- **Programming:** Python  
- **Libraries:** `pandas`, `numpy`, `scikit-learn`, `seaborn`, `matplotlib`  
- **Machine Learning Models:** Linear Regression, Decision Tree, Random Forest  
- **Feature Engineering:** Handling missing values, one-hot encoding, feature scaling  
- **Deployment:** Flask API (optional)  

---

## 🚀 Getting Started  

### 🔧 Installation Steps  

1️⃣ Clone the repository  
```bash
git clone https://github.com/21ds_Skyai/mumbai-house-price-prediction.git
cd mumbai-house-price-prediction
